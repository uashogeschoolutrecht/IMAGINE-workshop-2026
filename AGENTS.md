# Project Guidelines

## Python Environment

This project uses `uv` to manage its local virtual environment. Run all Python
programs, tests, and tools through `uv run ...` rather than activating the
environment directly.

`pytest` is available in the local environment. Every executable Python script
(any file with an `if __name__ == "__main__":` block) and every Python module
whose filename does not start with an underscore must have a matching test file
at `tests/test_<module_name>.py`, where `<module_name>` is the filename without
`.py`. If two modules share a filename, mirror their package paths under
`tests/` instead, for example `tests/pkg_a/test_utils.py`.
Only `__init__.py` files are exempt. Other files whose names begin with `__`,
including `__main__.py`, require a matching test file.

The `scripts/` directory is reserved for Bash scripts. Add new non-Bash scripts
to the repository root.
