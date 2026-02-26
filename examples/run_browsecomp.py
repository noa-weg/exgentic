# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

## CLI usage:
# exgentic evaluate --benchmark browsecompplus --agent tool_calling --subset main --num-tasks 3 \
#   --model gpt-4o
## Python API usage:
# from exgentic import RunConfig, evaluate
# evaluate(RunConfig(benchmark="browsecompplus", agent="tool_calling", subset="main", num_tasks=3,
#   model="gpt-4o"))
## Direct class usage (this script):
# BrowseCompPlusBenchmark + LiteLLMToolCallingAgent

from exgentic import BrowseCompPlusBenchmark, LiteLLMToolCallingAgent, evaluate


def main() -> None:
    benchmark = BrowseCompPlusBenchmark()
    agent = LiteLLMToolCallingAgent(model="gpt-4o")
    evaluate(benchmark=benchmark, agent=agent, output_dir="./outputs", num_tasks=3)


if __name__ == "__main__":
    main()
