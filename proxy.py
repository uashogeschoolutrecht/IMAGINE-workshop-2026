import json
import logging
import re
import uuid
from copy import deepcopy
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

UPSTREAM = "http://127.0.0.1:8000"

app = FastAPI(title="llama.cpp OpenAI compatibility proxy")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("llama-proxy")


# ---------------------------------------------------------------------------
# Tool-call cache
# ---------------------------------------------------------------------------

TOOL_CALL_CACHE: dict[str, dict[str, Any]] = {}
LAST_TOOL_CALL: dict[str, Any] | None = None


def json_string(value: Any) -> str:
    """OpenAI function.arguments must be a JSON string."""
    if isinstance(value, str):
        return value

    if value is None:
        return "{}"

    return json.dumps(
        value,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def get_declared_tools(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return declared function tools indexed by function name."""
    result: dict[str, dict[str, Any]] = {}

    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue

        if tool.get("type") != "function":
            continue

        function = tool.get("function")

        if not isinstance(function, dict):
            continue

        name = function.get("name")

        if isinstance(name, str) and name:
            result[name] = function

    return result


def cache_tool_call(tool_call: dict[str, Any]) -> None:
    global LAST_TOOL_CALL

    tc = deepcopy(tool_call)

    call_id = tc.get("id")

    if isinstance(call_id, str) and call_id:
        TOOL_CALL_CACHE[call_id] = tc

    LAST_TOOL_CALL = tc


def recover_tool_call(
    raw_call: dict[str, Any],
    declared_tools: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Normalize/recover an incoming historical assistant tool call.

    Recovery priority:
      1. exact cached call ID
      2. most recently generated proxy tool call
      3. infer function name if exactly one tool is declared
      4. generate missing ID
      5. use {} for irrecoverable missing arguments
    """

    call_id = raw_call.get("id")

    raw_function = raw_call.get("function")
    if not isinstance(raw_function, dict):
        raw_function = {}

    cached: dict[str, Any] | None = None

    # Exact call-id match.
    if isinstance(call_id, str) and call_id:
        cached = TOOL_CALL_CACHE.get(call_id)

    # Fall back to the most recently emitted tool call.
    if cached is None and LAST_TOOL_CALL is not None:
        cached = LAST_TOOL_CALL

    # ------------------------------------------------------------------
    # Function name
    # ------------------------------------------------------------------

    name = raw_function.get("name")

    if (not isinstance(name, str) or not name) and cached:
        cached_function = cached.get("function") or {}
        cached_name = cached_function.get("name")

        if isinstance(cached_name, str) and cached_name:
            name = cached_name

    # If there is exactly one possible function, inference is unambiguous.
    if (not isinstance(name, str) or not name) and len(declared_tools) == 1:
        name = next(iter(declared_tools))

    # ------------------------------------------------------------------
    # Arguments
    # ------------------------------------------------------------------

    arguments = raw_function.get("arguments")

    if arguments is None and cached:
        cached_function = cached.get("function") or {}
        cached_arguments = cached_function.get("arguments")

        if cached_arguments is not None:
            arguments = cached_arguments

    # If we cannot recover the original arguments, keep the historical
    # call structurally valid. This is preferable to sending null into
    # llama.cpp.
    if arguments is None:
        arguments = {}

    # ------------------------------------------------------------------
    # Call ID
    # ------------------------------------------------------------------

    if (not isinstance(call_id, str) or not call_id) and cached:
        cached_id = cached.get("id")

        if isinstance(cached_id, str) and cached_id:
            call_id = cached_id

    if not isinstance(call_id, str) or not call_id:
        call_id = f"call_{uuid.uuid4().hex}"

    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json_string(arguments),
        },
    }


def normalize_incoming_messages(body: dict[str, Any]) -> None:
    """
    Normalize client/VS Code messages before forwarding them to llama.cpp.
    """

    messages = body.get("messages")

    if not isinstance(messages, list):
        return

    declared_tools = get_declared_tools(body)

    preceding_tool_calls: list[dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role")

        # ------------------------------------------------------------------
        # Assistant tool-call message
        # ------------------------------------------------------------------

        if role == "assistant":
            raw_calls = message.get("tool_calls")

            if isinstance(raw_calls, list) and raw_calls:
                # Some llama.cpp builds reject content:null.
                if message.get("content") is None:
                    message["content"] = ""

                normalized_calls = []

                for raw_call in raw_calls:
                    if not isinstance(raw_call, dict):
                        continue

                    tc = recover_tool_call(
                        raw_call,
                        declared_tools,
                    )

                    normalized_calls.append(tc)

                message["tool_calls"] = normalized_calls
                preceding_tool_calls = normalized_calls

        # ------------------------------------------------------------------
        # Tool result
        # ------------------------------------------------------------------

        elif role == "tool":
            content = message.get("content")

            if content is None:
                message["content"] = ""

            elif not isinstance(content, str):
                message["content"] = json.dumps(
                    content,
                    ensure_ascii=False,
                )

            tool_call_id = message.get("tool_call_id")

            if not isinstance(tool_call_id, str) or not tool_call_id:
                # Best source: immediately preceding assistant tool call.
                if len(preceding_tool_calls) == 1:
                    message["tool_call_id"] = preceding_tool_calls[0]["id"]

                # Otherwise use the last call generated by this proxy.
                elif LAST_TOOL_CALL is not None:
                    last_id = LAST_TOOL_CALL.get("id")

                    if isinstance(last_id, str) and last_id:
                        message["tool_call_id"] = last_id

                # Last resort: make it structurally valid.
                else:
                    message["tool_call_id"] = f"call_{uuid.uuid4().hex}"

        # ------------------------------------------------------------------
        # Ordinary message
        # ------------------------------------------------------------------

        else:
            if "content" in message and message["content"] is None:
                message["content"] = ""


def validate_incoming_messages(body: dict[str, Any]) -> None:
    """
    Check that no remaining null function names reach llama.cpp.
    """

    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue

        if message.get("role") != "assistant":
            continue

        for tc in message.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue

            function = tc.get("function") or {}
            name = function.get("name")

            if not isinstance(name, str) or not name:
                raise ValueError(
                    "Unable to determine function name for tool call. "
                    "Multiple tools were supplied and the original call "
                    "contained no function name."
                )


def parse_tool_object(
    obj: Any,
    allowed_tools: set[str],
) -> dict[str, Any] | None:
    """
    Convert a textual/pseudo tool-call JSON object emitted by the model into
    OpenAI tool_calls format.
    """

    if not isinstance(obj, dict):
        return None

    if isinstance(obj.get("function"), dict):
        obj = obj["function"]

    name = obj.get("name")

    if not isinstance(name, str) or not name:
        return None

    # Do not accidentally promote arbitrary JSON into a tool call.
    if name not in allowed_tools:
        return None

    arguments = obj.get("arguments", {})

    tc = {
        "id": f"call_{uuid.uuid4().hex}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json_string(arguments),
        },
    }

    cache_tool_call(tc)

    return tc


def try_json(text: str) -> Any | None:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError, TypeError:
        return None


def extract_textual_tool_call(
    content: str | None,
    allowed_tools: set[str],
) -> dict[str, Any] | None:
    """
    Recognize tool-call formats we have observed from Qwen:

      <function_call>
      {"name":"list_files","arguments":{"path":"."}}
      </function_call>

      <tool_call>
      {...}
      </tool_call>

      ```json
      {...}
      ```

      or a bare JSON object.
    """

    if not isinstance(content, str):
        return None

    text = content.strip()

    if not text:
        return None

    candidates: list[str] = []

    # <function_call>...</function_call>
    # <tool_call>...</tool_call>
    for tag in ("function_call", "tool_call"):
        pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"

        for match in re.finditer(
            pattern,
            text,
            flags=re.DOTALL | re.IGNORECASE,
        ):
            candidates.append(match.group(1))

    # ```json ... ```
    for match in re.finditer(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        candidates.append(match.group(1))

    # Bare JSON.
    candidates.append(text)

    for candidate in candidates:
        obj = try_json(candidate)

        if obj is None:
            continue

        tc = parse_tool_object(
            obj,
            allowed_tools,
        )

        if tc is not None:
            return tc

    return None


def normalize_existing_outgoing_tool_calls(
    message: dict[str, Any],
) -> None:
    """
    Normalize real tool_calls already returned by llama.cpp.
    """

    raw_calls = message.get("tool_calls")

    if not isinstance(raw_calls, list):
        return

    normalized = []

    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            continue

        call_id = raw_call.get("id")

        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_{uuid.uuid4().hex}"

        function = raw_call.get("function")

        if not isinstance(function, dict):
            function = {}

        name = function.get("name")
        arguments = function.get("arguments", {})

        tc = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json_string(arguments),
            },
        }

        normalized.append(tc)
        cache_tool_call(tc)

    message["tool_calls"] = normalized


async def upstream(
    method: str,
    path: str,
    json_body: dict[str, Any] | None = None,
) -> httpx.Response:
    async with httpx.AsyncClient(timeout=None) as client:
        return await client.request(
            method,
            f"{UPSTREAM}{path}",
            json=json_body,
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "status": "ok",
        "proxy": "llama.cpp OpenAI compatibility proxy",
        "upstream": UPSTREAM,
    }


@app.get("/health", response_model=None)
async def health() -> dict[str, str | int] | JSONResponse:
    try:
        r = await upstream("GET", "/health")

        return {
            "proxy": "ok",
            "upstream_status": r.status_code,
        }

    except httpx.HTTPError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "proxy": "ok",
                "upstream": "unreachable",
                "error": str(exc),
            },
        )


@app.get("/v1/models")
async def models() -> JSONResponse:
    try:
        r = await upstream("GET", "/v1/models")
        return JSONResponse(
            status_code=r.status_code,
            content=r.json(),
        )

    except httpx.HTTPError as exc:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": str(exc),
                    "type": "upstream_error",
                }
            },
        )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    try:
        body = await request.json()

        if not isinstance(body, dict):
            raise TypeError("Request body must be a JSON object")

    except (json.JSONDecodeError, TypeError) as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Invalid request: {exc}",
                    "type": "invalid_request_error",
                }
            },
        )

    # ------------------------------------------------------------------
    # Normalize VS Code/client -> llama.cpp
    # ------------------------------------------------------------------

    normalize_incoming_messages(body)

    try:
        validate_incoming_messages(body)

    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                }
            },
        )

    # Streaming normalization is deliberately disabled for now.
    body["stream"] = False

    declared_tools = get_declared_tools(body)
    allowed_tools = set(declared_tools)

    log.info(
        "request: model=%s messages=%d tools=%s",
        body.get("model"),
        len(body.get("messages") or []),
        sorted(allowed_tools),
    )

    # Useful while debugging:
    log.debug(
        "normalized upstream request:\n%s",
        json.dumps(
            body,
            indent=2,
            ensure_ascii=False,
        ),
    )

    # ------------------------------------------------------------------
    # Call llama.cpp
    # ------------------------------------------------------------------

    try:
        r = await upstream(
            "POST",
            "/v1/chat/completions",
            body,
        )

    except httpx.HTTPError as exc:
        log.exception("Could not contact llama.cpp")

        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": f"Unable to reach llama.cpp: {exc}",
                    "type": "upstream_connection_error",
                }
            },
        )

    try:
        data = r.json()

    except json.JSONDecodeError:
        return JSONResponse(
            status_code=r.status_code,
            content={
                "error": {
                    "message": r.text,
                    "type": "upstream_error",
                }
            },
        )

    if r.status_code >= 400:
        return JSONResponse(
            status_code=r.status_code,
            content=data,
        )

    # ------------------------------------------------------------------
    # Normalize llama.cpp -> VS Code/client
    # ------------------------------------------------------------------

    for choice in data.get("choices") or []:
        if not isinstance(choice, dict):
            continue

        message = choice.get("message")

        if not isinstance(message, dict):
            continue

        # llama.cpp already returned proper tool_calls.
        if message.get("tool_calls"):
            normalize_existing_outgoing_tool_calls(message)

            if message.get("content") == "":
                message["content"] = None

            choice["finish_reason"] = "tool_calls"
            continue

        # Otherwise detect Qwen's textual pseudo-calls.
        tc = extract_textual_tool_call(
            message.get("content"),
            allowed_tools,
        )

        if tc is not None:
            log.info(
                "converted textual tool call: %s %s",
                tc["function"]["name"],
                tc["function"]["arguments"],
            )

            message["content"] = None
            message["tool_calls"] = [tc]
            choice["finish_reason"] = "tool_calls"

    return JSONResponse(
        status_code=r.status_code,
        content=data,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=9000,
    )
