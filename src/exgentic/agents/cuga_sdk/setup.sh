#!/bin/bash
# Install CUGA SDK agent dependencies

echo "Installing CUGA SDK Agent..."

if [ ! -f "pyproject.toml" ]; then
    echo "Error: setup.sh must be run from the root directory of the Exgentic project"
    exit 1
fi

if command -v uv >/dev/null 2>&1; then
    echo "Using uv for installation..."
    uv pip install -e ".[cuga]"
else
    echo "Using pip for installation..."
    python -m pip install -e ".[cuga]"
fi

if [ $? -eq 0 ]; then
    echo "CUGA SDK Agent installed successfully"
else
    echo "CUGA SDK Agent installation failed"
    exit 1
fi
