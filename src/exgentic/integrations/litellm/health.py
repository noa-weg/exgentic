# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Minimal LiteLLM model accessibility check."""

from __future__ import annotations


async def acheck_model_accessible(model: str) -> None:
    """Raise if LiteLLM cannot access the configured model."""
    import litellm

    result = await litellm.ahealth_check(
        {
            "model": model,
            "messages": [{"role": "user", "content": "test from litellm"}],
            "max_tokens": 5,
        },
        mode="chat",
    )
    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError(f"LiteLLM health check failed: {result['error']}")
