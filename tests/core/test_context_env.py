# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import os

from exgentic.core import context as context_mod
from exgentic.core.context import (
    Context,
    Role,
    context_env,
    context_env_scope,
    set_context,
)


def test_context_env_empty_when_no_context():
    token = context_mod._CONTEXT.set(None)
    try:
        assert context_env() == {}
    finally:
        context_mod._CONTEXT.reset(token)


def test_context_env_scope_applies_and_restores():
    ctx = Context(
        run_id="run-1",
        output_dir="/tmp/out",
        cache_dir="/tmp/cache",
        session_id="sess-1",
        task_id="task-1",
        role=Role.AGENT,
    )
    set_context(ctx)

    key = "EXGENTIC_CTX_RUN_ID"
    prev = os.environ.get(key)
    assert key not in os.environ

    with context_env_scope():
        assert os.environ.get(key) == "run-1"

    assert os.environ.get(key) == prev
    os.environ.pop(key, None)
