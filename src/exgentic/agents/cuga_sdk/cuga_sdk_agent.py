# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from typing import Any, ClassVar

from ...core.agent import Agent
from ...core.types import ModelSettings


class CUGASDKAgent(Agent):
    display_name: ClassVar[str] = "CUGA SDK"
    slug_name: ClassVar[str] = "cuga_sdk"

    # Run in-process (same Python env as the main exgentic process / .venv312).
    # This avoids a separate subprocess venv and uses whatever CUGA version is
    # installed in the invoking environment.
    runner: str = "direct"

    model: str = "groq/openai/gpt-oss-120b"
    max_steps: int = 150
    model_settings: ModelSettings | None = None
    # CUGA execution mode. Options: "fast" (no task decomposition, no reflection),
    # "balanced", "accurate" (full planning + reflection).
    cuga_mode: str = "accurate"
    # Total seconds allowed for CUGA's full multi-turn loop.  With many tools
    # (e.g. AppWorld ~468) CUGA uses find_tools semantic search which requires
    # extra LLM calls; set this high enough to cover planning + execution.
    invoke_timeout: float = 3600.0

    @property
    def model_name(self) -> str:  # type: ignore[override]
        return str(self.model).split("/")[-1]

    def get_models_names(self) -> list[str]:  # type: ignore[override]
        return [str(self.model)]

    @classmethod
    def _get_instance_class(cls):
        from .instance import CUGASDKAgentInstance

        return CUGASDKAgentInstance

    def _get_instance_kwargs(self, session_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "model": self.model,
            "max_steps": self.max_steps,
            "model_settings": self.model_settings,
            "cuga_mode": self.cuga_mode,
            "invoke_timeout": self.invoke_timeout,
        }
