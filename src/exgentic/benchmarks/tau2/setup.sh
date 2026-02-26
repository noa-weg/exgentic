#!/bin/bash
# Install TAU-2-bench - least intrusive approach

# Install directly from GitHub using pip
cd src/exgentic/benchmarks/tau2
orig_cwd=$CWD
if [ $? -ne 0 ]; then
    echo "TAU-2-bench setup script must be run from root directory of Exgentic project"
    exit 1
fi
rm -fr installation
mkdir -p installation
cd installation
git clone --branch v0.1.3 --depth 1 https://github.com/sierra-research/tau2-bench.git
cd tau2-bench

if command -v uv >/dev/null 2>&1; then
    uv pip install -e .
else
    python -m pip install -e .
fi

cd $orig_cwd

# Verify installation
tau2 --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "TAU-2-bench installed successfully"
else
    echo "TAU-2-bench installation failed or tau2 command not found"
    exit 1
fi
