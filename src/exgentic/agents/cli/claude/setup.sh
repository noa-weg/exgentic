#!/bin/bash
# Install Claude Code Agent dependencies

echo "Installing Claude Code Agent..."

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

if [ $? -ne 0 ]; then
    echo "Claude Code Agent installation failed"
    exit 1
fi

# Install Claude Code CLI
echo "Installing Claude Code CLI..."
if command -v npm >/dev/null 2>&1; then
    npm install -g @anthropic-ai/claude-code
    if [ $? -eq 0 ]; then
        echo "Claude Code CLI installed successfully"
    else
        echo "Warning: Claude Code CLI installation failed"
        echo "You may need to install it manually with: npm install -g @anthropic-ai/claude-code"
    fi
else
    echo "Warning: npm not found. Please install Node.js and npm to use Claude Code CLI"
    echo "After installing npm, run: npm install -g @anthropic-ai/claude-code"
fi

echo "Claude Code Agent setup complete"
echo "Note: Additional Claude-specific configuration may be required"

# Made with Bob
