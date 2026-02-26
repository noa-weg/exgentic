# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

## CLI usage:
# exgentic evaluate --benchmark gsm8k --agent codex_cli --num-tasks 3 \
#   --model gpt-4o
## Python API usage:
# from exgentic import RunConfig, evaluate
# evaluate(RunConfig(benchmark="gsm8k", agent="codex_cli", num_tasks=3,
#   model="gpt-4o"))
## Direct class usage (this script):
# GSM8kBenchmark + CodexAgent

from exgentic import CodexAgent, GSM8kBenchmark, evaluate


def main() -> None:
    benchmark = GSM8kBenchmark()
    agent = CodexAgent(model="gpt-4o")
    evaluate(benchmark=benchmark, agent=agent, output_dir="./outputs", num_tasks=3)


if __name__ == "__main__":
    main()
