#!/bin/bash
# Install Codex Agent dependencies

echo "Installing Codex Agent..."

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo "Error: setup.sh must be run from the root directory of the Exgentic project"
    exit 1
fi

# Install the base dependencies with CLI extra
if command -v uv >/dev/null 2>&1; then
    echo "Using uv for installation..."
    uv pip install -e ".[cli]"
else
    echo "Using pip for installation..."
    python -m pip install -e ".[cli]"
fi

if [ $? -eq 0 ]; then
    echo "Codex Agent installed successfully"
    echo "Note: Additional Codex-specific configuration may be required"
else
    echo "Codex Agent installation failed"
    exit 1
fi

# Made with Bob
