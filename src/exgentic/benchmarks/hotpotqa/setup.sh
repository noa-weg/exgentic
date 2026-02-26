#!/bin/bash
pip_install() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install "$@"
  else
    python -m pip install "$@"
  fi
}

pip_install datasets
pip_install fastmcp
pip_install wikipedia-mcp


