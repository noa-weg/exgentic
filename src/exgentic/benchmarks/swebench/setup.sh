#!/usr/bin/env bash
set -euo pipefail

# Require Git LFS for bundled assets
if ! git lfs version >/dev/null 2>&1; then
  echo "Error: Git LFS not found. Install Git LFS and re-run." >&2
  exit 1
fi

SWEBENCH_REPO="https://github.com/SWE-bench/SWE-bench.git"
SWEBENCH_TAG="v4.1.0"
pip_install() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install "$@"
  else
    python -m pip install "$@"
  fi
}

pip_install "git+${SWEBENCH_REPO}@${SWEBENCH_TAG}"

MINI_SWE_AGENT_REPO="https://github.com/SWE-agent/mini-swe-agent.git"
MINI_SWE_AGENT_TAG="v1.17.0"
pip_install "git+${MINI_SWE_AGENT_REPO}@${MINI_SWE_AGENT_TAG}"

python - <<'PY'
import importlib, sys
mod = importlib.import_module("swebench")
print("swebench imported from:", mod.__file__)
mod = importlib.import_module("minisweagent")
print("minisweagent imported from:", mod.__file__)
PY
