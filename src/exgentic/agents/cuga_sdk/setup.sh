#!/bin/bash
# Install CUGA SDK agent dependencies

set -euo pipefail

echo "Installing CUGA SDK Agent..."

# Resolve the exgentic project root relative to this script's location.
# setup.sh lives at: src/exgentic/agents/cuga_sdk/setup.sh
# Project root is 4 levels up from this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
    echo "Error: could not find pyproject.toml at $PROJECT_ROOT" >&2
    exit 1
fi

# The venv is created with `uv venv` (no pip). Use `uv pip install` which
# respects the VIRTUAL_ENV env var set by exgentic's run_setup_sh helper.
uv pip install -e "$PROJECT_ROOT"

echo "CUGA SDK Agent installed successfully"
