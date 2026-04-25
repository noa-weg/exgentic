# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from typing import Any, ClassVar

from ...core.agent import Agent
from ...core.types import ModelSettings


class CUGASDKAgent(Agent):
    """Exgentic agent that drives a task using the CUGA SDK.

    CUGA (Cognitive Unified General Agent) is a multi-turn agentic framework
    that orchestrates planning, tool selection, execution, and reflection
    internally.  This class is the thin configuration layer that exgentic uses
    to instantiate a ``CUGASDKAgentInstance`` per task session.

    Unlike step-based agents that call ``react()`` once per turn, CUGA runs its
    entire loop in a single ``run_code_agent()`` call.  The benchmark tools are
    passed as real LangChain ``StructuredTool`` callables so CUGA receives live
    observations from the environment rather than simulated responses.

    Model string format
    ~~~~~~~~~~~~~~~~~~~
    ``model`` uses a ``"<provider>/<native_model>"`` convention so exgentic can
    derive both the LiteLLM cost-lookup string and the model name CUGA should
    receive.  The provider prefix is stripped before being set as ``MODEL_NAME``
    in the environment (CUGA's config key).

    Example: ``"groq/openai/gpt-oss-120b"``
      - provider  → ``"groq"``  (selects settings.groq.toml)
      - MODEL_NAME → ``"openai/gpt-oss-120b"``  (passed to CUGA)
      - model_name → ``"gpt-oss-120b"``  (used for cost display)
    """

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
        """Short model name for display (last component of the model string)."""
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
