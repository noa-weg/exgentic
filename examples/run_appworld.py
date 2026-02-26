# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

## CLI usage:
# exgentic evaluate --benchmark appworld --agent tool_calling --subset test_normal --num-tasks 3 \
#   --model gpt-4o
## Python API usage:
# from exgentic import RunConfig, evaluate
# evaluate(RunConfig(benchmark="appworld", agent="tool_calling", subset="test_normal", num_tasks=3,
#   model="gpt-4o"))
## Direct class usage (this script):
# AppWorldBenchmark + LiteLLMToolCallingAgent

from exgentic import AppWorldBenchmark, LiteLLMToolCallingAgent, evaluate


def main() -> None:
    benchmark = AppWorldBenchmark(subset="test_normal")
    agent = LiteLLMToolCallingAgent(model="gpt-4o")
    evaluate(benchmark=benchmark, agent=agent, output_dir="./outputs", num_tasks=3)


if __name__ == "__main__":
    main()
