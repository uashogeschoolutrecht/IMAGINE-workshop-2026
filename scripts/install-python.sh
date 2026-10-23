#!/usr/bin/env bash
set -Eeuo pipefail

# Install uv, pyenv, and Python 3.14 without sudo or administrator access.
#
# The operating system must already provide a compiler, standard build tools,
# and Python's development libraries. Installing those system prerequisites is
# deliberately outside this script because doing so requires administrator
# privileges on a clean macOS or Linux installation.

readonly PYENV_GIT_TAG="${PYENV_GIT_TAG:-v2.6.25}"
readonly UV_VERSION="${UV_VERSION:-0.11.14}"
export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
export PATH="$PYENV_ROOT/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

log() {
    printf '\n==> %s\n' "$*"
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

require_command() {
    local command_name="$1"
    command -v "$command_name" >/dev/null 2>&1 || \
        fail "Required command '$command_name' is missing. Ask the system administrator to install the build prerequisites."
}

check_linux_headers() {
    local header source_file missing_headers=()
    local -a headers=(
        bzlib.h
        ffi.h
        lzma.h
        openssl/ssl.h
        readline/readline.h
        sqlite3.h
        zlib.h
    )

    source_file="$(mktemp "${TMPDIR:-/tmp}/python-prerequisite.XXXXXX.c")"

    for header in "${headers[@]}"; do
        printf '#include <%s>\n' "$header" >"$source_file"
        if ! cc -E "$source_file" >/dev/null 2>&1; then
            missing_headers+=("$header")
        fi
    done
    rm -f "$source_file"

    if ((${#missing_headers[@]} > 0)); then
        printf 'Error: required development headers are missing:\n' >&2
        printf '  - %s\n' "${missing_headers[@]}" >&2
        printf 'Ask the system administrator to install the Python build dependencies.\n' >&2
        exit 1
    fi
}

check_prerequisites() {
    local tool

    log "Checking unprivileged build prerequisites"
    for tool in cc curl git make; do
        require_command "$tool"
    done

    # command -v can find Apple's developer-tool stubs even when the Command
    # Line Tools are absent, so xcrun must validate the actual toolchain.
    case "$(uname -s)" in
        Darwin)
            require_command xcrun
            xcrun --find clang >/dev/null 2>&1 || \
                fail "Xcode Command Line Tools are unavailable. Ask the system administrator to install them."
            xcrun --show-sdk-path >/dev/null 2>&1 || \
                fail "The macOS SDK is unavailable. Ask the system administrator to install Xcode Command Line Tools."
            ;;
        Linux)
            check_linux_headers
            ;;
        *)
            fail "Unsupported operating system: $(uname -s)"
            ;;
    esac

    curl --version >/dev/null
    git --version >/dev/null
    make --version >/dev/null
    cc --version >/dev/null
}

install_pyenv() {
    if command -v pyenv >/dev/null 2>&1; then
        log "pyenv is already installed: $(pyenv --version)"
        return
    fi

    if [[ -e "$PYENV_ROOT" ]]; then
        fail "$PYENV_ROOT already exists but does not contain an executable pyenv installation."
    fi

    log "Installing pyenv $PYENV_GIT_TAG in $PYENV_ROOT"
    git clone --branch "$PYENV_GIT_TAG" --depth 1 \
        https://github.com/pyenv/pyenv.git "$PYENV_ROOT"

    command -v pyenv >/dev/null 2>&1 || \
        fail "pyenv was cloned but is not available on PATH."
}

configure_pyenv_for_this_process() {
    eval "$(pyenv init - bash)"
}

append_shell_setup_to() {
    local rc_file="$1"
    local shell_name="$2"
    local marker="# >>> install-python.sh user-local setup >>>"
    local pyenv_command_dir
    pyenv_command_dir="$(dirname "$(command -v pyenv)")"

    touch "$rc_file"
    if grep -Fq "$marker" "$rc_file"; then
        return
    fi

    log "Adding user-local tool setup to $rc_file"
    {
        printf '\n%s\n' "$marker"
        printf 'export PYENV_ROOT=%q\n' "$PYENV_ROOT"
        # These variables must expand when the shell later reads its rc file.
        # shellcheck disable=SC2016
        printf '%s\n' 'export PATH="$PYENV_ROOT/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"'
        if [[ "$pyenv_command_dir" != "$PYENV_ROOT/bin" ]]; then
            # Keep a pre-existing pyenv installation reachable in new shells.
            # shellcheck disable=SC2016
            printf 'export PATH="%s:$PATH"\n' "$pyenv_command_dir"
        fi
        printf '%s\n' "eval \"\$(pyenv init - $shell_name)\""
        printf '%s\n' '# <<< install-python.sh user-local setup <<<'
    } >>"$rc_file"
}

configure_login_shell() {
    local shell_name
    shell_name="$(basename "${SHELL:-bash}")"

    case "$shell_name" in
        zsh)
            append_shell_setup_to "$HOME/.zshrc" zsh
            ;;
        bash)
            # Bash login shells and non-login interactive shells read different
            # files, so configure both without assuming one sources the other.
            append_shell_setup_to "$HOME/.bash_profile" bash
            append_shell_setup_to "$HOME/.bashrc" bash
            ;;
        *)
            log "Skipping automatic shell setup for unsupported shell '$shell_name'"
            printf 'Add %s/bin and %s/.local/bin to PATH, then initialize pyenv manually.\n' \
                "$PYENV_ROOT" "$HOME"
            ;;
    esac
}

install_uv() {
    if command -v uv >/dev/null 2>&1; then
        log "uv is already installed: $(uv --version)"
        return
    fi

    log "Installing uv $UV_VERSION in the current user's home directory"
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | \
        env UV_NO_MODIFY_PATH=1 sh

    command -v uv >/dev/null 2>&1 || \
        fail "uv was installed but is not available on PATH."
}

latest_python_314() {
    pyenv install --list | sed -nE 's/^[[:space:]]+(3\.14\.[0-9]+)$/\1/p' | tail -n 1
}

ensure_python_definition() {
    local python_version
    python_version="$(latest_python_314)"

    if [[ -n "$python_version" ]]; then
        printf '%s\n' "$python_version"
        return
    fi

    fail "The installed pyenv does not know a stable Python 3.14 release. Update pyenv or rerun with a newer PYENV_GIT_TAG."
}

install_python() {
    local python_version
    python_version="$(ensure_python_definition)"

    log "Installing Python $python_version"
    pyenv install --skip-existing "$python_version"
    pyenv global "$python_version"

    log "Installation complete"
    printf 'uv:     %s\n' "$(uv --version)"
    printf 'pyenv:  %s\n' "$(pyenv --version)"
    printf 'Python: %s\n' "$(PYENV_VERSION="$python_version" pyenv exec python --version)"
    printf '\nRestart your shell, or run: exec %s -l\n' "${SHELL:-bash}"
}

check_prerequisites
install_pyenv
configure_pyenv_for_this_process
configure_login_shell
install_uv
install_python
