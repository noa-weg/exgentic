# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
from typing import Any, Callable, List

from ...adapters.agents.code_agent import CodeAgentInstance
from ...core.types import ModelSettings
from ...utils.cost import CostReport, LiteLLMCostReport
from ...utils.sync import run_sync


class _TokenAccumulator:
    """LangChain BaseCallbackHandler that sums token usage across all LLM calls.

    CUGA runs multiple internal LLM calls (planner, tool selector, executor,
    reflector, final-answer).  Each fires ``on_llm_end`` with token counts in
    ``response.llm_output``.  We accumulate them here so ``get_cost()`` can
    report the total for the entire task.

    We inherit lazily so the import of langchain_core is deferred until CUGA is
    actually used — keeping exgentic importable without langchain installed.
    """

    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self._handler: Any = None

    def _ensure_handler(self) -> Any:
        if self._handler is not None:
            return self._handler
        try:
            from langchain_core.callbacks.base import BaseCallbackHandler
        except Exception as exc:
            raise RuntimeError(
                "langchain_core is required for CUGA cost tracking "
                "(install optional `cuga` extra)."
            ) from exc

        accumulator = self

        class _Handler(BaseCallbackHandler):
            def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # type: ignore[override]
                llm_output = getattr(response, "llm_output", None) or {}
                usage = (
                    llm_output.get("token_usage")
                    or llm_output.get("usage")
                    or {}
                )
                accumulator.input_tokens += usage.get("prompt_tokens", 0)
                accumulator.output_tokens += usage.get("completion_tokens", 0)

        self._handler = _Handler()
        return self._handler

    @property
    def handler(self) -> Any:
        return self._ensure_handler()


class MultiAppToolProvider:
    """Splits a flat list of LangChain StructuredTools into per-app groups.

    Each tool must have its ``metadata["server_name"]`` set to the app it belongs to
    (e.g. ``"phone"``, ``"file_system"``).  Tools without a ``server_name`` fall back
    to the ``"runtime_tools"`` bucket — this is the normal path for TAU2 and other
    benchmarks with flat tool namespaces.

    Note: tool names exposed to CUGA use single ``_`` (e.g. ``phone_login``) because
    CUGA rejects ``__`` as a dunder-security violation.  App grouping must therefore
    rely on ``metadata["server_name"]``, not on splitting the tool name.
    """

    def __init__(self, tools: list[Any]) -> None:
        self._apps: dict[str, list[Any]] = {}
        for t in tools:
            app = (t.metadata or {}).get("server_name", "runtime_tools")
            self._apps.setdefault(app, []).append(t)
        self.initialized: bool = False

    async def initialize(self) -> None:
        self.initialized = True

    async def get_apps(self) -> list[Any]:
        try:
            from cuga.backend.cuga_graph.nodes.cuga_lite.tool_provider_interface import AppDefinition
        except Exception as exc:
            raise RuntimeError("CUGA SDK is required for MultiAppToolProvider.") from exc
        return [AppDefinition(name=app_name, description=app_name) for app_name in self._apps]

    async def get_tools(self, app_name: str) -> list[Any]:
        app_tools = self._apps.get(app_name, [])
        # Always include runtime_tools alongside the requested app.
        # "runtime_tools" holds tools with no app prefix (e.g. the finish/
        # complete-task tool in AppWorld whose action name is just "finish").
        # CUGA searches by app name and would miss these tools otherwise.
        # For flat-namespace benchmarks (TAU2) all tools are already in
        # runtime_tools, so this is a no-op.
        if app_name != "runtime_tools":
            seen_names = {t.name for t in app_tools}
            runtime_extras = [
                t for t in self._apps.get("runtime_tools", [])
                if t.name not in seen_names
            ]
            return app_tools + runtime_extras
        return app_tools

    async def get_all_tools(self) -> list[Any]:
        return [t for tools in self._apps.values() for t in tools]


class CUGASDKAgentInstance(CodeAgentInstance):
    """CUGA SDK agent using the CodeAgentInstance pattern.

    Runs CUGA's full multi-turn loop once inside ``run_code_agent()``.  Each tool
    callable is a real executable that blocks until the benchmark environment returns
    an observation, giving CUGA accurate results to reason over and preserving full
    conversation context inside CUGA's MemorySaver checkpointer.
    """

    def __init__(
        self,
        session_id: str,
        model: str = "groq/openai/gpt-oss-120b",
        max_steps: int = 150,
        model_settings: ModelSettings | None = None,
        cuga_mode: str = "accurate",
        invoke_timeout: float = 600.0,
    ) -> None:
        super().__init__(session_id)
        self.model_id = model
        self.max_steps = max_steps
        self.model_settings = model_settings
        self.cuga_mode = cuga_mode
        self.invoke_timeout = invoke_timeout
        self._token_accumulator: _TokenAccumulator | None = None

    # ------------------------------------------------------------------
    # CodeAgentInstance interface
    # ------------------------------------------------------------------

    def run_code_agent(self, functions: List[Callable]) -> None:
        # Configure CUGA env vars before the first `import cuga`.
        # cuga/config.py reads AGENT_SETTING_CONFIG at module-import time, so
        # these must be set before _get_cuga_agent_class() triggers the import.
        # The agent subprocess only receives forwarded env vars (API keys, OTEL)
        # from the parent; CUGA-specific config vars must be set here explicitly.
        import os as _os
        _provider = self.model_id.split("/")[0].lower() if "/" in self.model_id else ""
        if "AGENT_SETTING_CONFIG" not in _os.environ:
            # Select CUGA settings file based on the provider prefix in model_id
            # (e.g. "groq/openai/gpt-oss-120b" → provider "groq").
            # Fall back to the old GROQ_API_KEY heuristic for bare model strings.
            if _provider == "groq" or _os.environ.get("GROQ_API_KEY"):
                _os.environ["AGENT_SETTING_CONFIG"] = "settings.groq.toml"
        if "MODEL_NAME" not in _os.environ and _provider:
            # Strip provider prefix so CUGA receives its native model string.
            # "groq/openai/gpt-oss-120b" → "openai/gpt-oss-120b"
            _os.environ["MODEL_NAME"] = self.model_id[len(_provider) + 1:]
        # Always apply the configured cuga_mode (allows overriding env default).
        _os.environ["DYNACONF_FEATURES__CUGA_MODE"] = self.cuga_mode
        if "DYNACONF_FEATURES__CHAT" not in _os.environ:
            _os.environ["DYNACONF_FEATURES__CHAT"] = "false"

        try:
            from langchain_core.tools import StructuredTool
        except Exception as exc:
            raise RuntimeError(
                "langchain_core is required for the CUGA tool bridge "
                "(install optional `cuga` extra)."
            ) from exc

        # >>>>>>>>>> FINISH-BRIDGE BEGIN (remove when CUGA calls finish natively) <<<<<<<<<<
        # # Track whether the finish action was called during CUGA's run.
        # # CUGA's FinalAnswerAgent routes to END via plain text — it never calls
        # # tools itself.  We need to call the is_finish action explicitly after
        # # agent.invoke() returns.  The flag prevents a double-call on the rare
        # # occasion that CUGA does invoke the finish tool directly.
        # _finish_called: list[bool] = [False]
        # >>>>>>>>>> FINISH-BRIDGE END <<<<<<<<<<

        # Build StructuredTools using ActionType schema + real callables.
        fn_by_name = {fn.__name__: fn for fn in functions}
        all_tools: list[Any] = []
        for action_type in self.actions:
            # Don't expose hidden tools to the LLM.
            if action_type.is_hidden:
                continue
            fn_name = action_type.name.replace(".", "__")
            fn = fn_by_name.get(fn_name)
            if fn is None:
                continue
            app_name = fn_name.split("__")[0] if "__" in fn_name else "runtime_tools"

            # >>>>>>>>>> FINISH-BRIDGE BEGIN (remove when CUGA calls finish natively) <<<<<<<<<<
            # # Wrap the finish callable so we know if CUGA called it itself.
            # if action_type.is_finish:
            #     import inspect as _inspect
            #     _orig_fn = fn
            #
            #     def _make_tracked(f: Callable) -> Callable:
            #         def _tracked(*args: Any, **kwargs: Any) -> Any:
            #             _finish_called[0] = True
            #             return f(*args, **kwargs)
            #
            #         _tracked.__name__ = f.__name__
            #         _tracked.__doc__ = f.__doc__
            #         _tracked.__annotations__ = getattr(f, "__annotations__", {})
            #         try:
            #             _tracked.__signature__ = _inspect.signature(f)
            #         except (ValueError, TypeError):
            #             pass
            #         return _tracked
            #
            #     fn = _make_tracked(_orig_fn)
            #     fn_by_name[fn_name] = fn
            # >>>>>>>>>> FINISH-BRIDGE END <<<<<<<<<<

            # CUGA's sandbox rejects any Python code containing "__" as a suspected
            # dunder-method security violation.  Exgentic uses "__" as the app/tool
            # separator (e.g. "supervisor__show_account_passwords"), so we replace
            # every "__" with a single "_" in the tool name exposed to CUGA.
            cuga_tool_name = fn_name.replace("__", "_")
            base_desc = action_type.description or fn_name
            if action_type.is_finish:
                base_desc = f"{base_desc} [TASK COMPLETION TOOL]"
            elif action_type.is_message:
                base_desc = f"{base_desc} [MESSAGE TOOL]"
            tool = StructuredTool.from_function(
                name=cuga_tool_name,
                description=base_desc,
                func=fn,
                args_schema=action_type.arguments,  # type: ignore[arg-type]
            )
            tool.metadata = {"server_name": app_name}
            all_tools.append(tool)

        # threshold=35: CUGA default — find_tools activates for large tool sets.
        shortlist_threshold = 35

        provider = MultiAppToolProvider(tools=all_tools)
        cuga_agent_cls = self._get_cuga_agent_class()
        agent = cuga_agent_cls(tool_provider=provider)

        self._token_accumulator = _TokenAccumulator()
        prompt = self._build_prompt()
        run_sync(
            agent.invoke(
                prompt,
                thread_id=self.session_id,
                track_tool_calls=True,
                config={
                    "configurable": {"shortlisting_tool_threshold": shortlist_threshold},
                    "callbacks": [self._token_accumulator.handler],
                },
            ),
            timeout=self.invoke_timeout,
        )

        # >>>>>>>>>> FINISH-BRIDGE BEGIN (remove when CUGA calls finish natively) <<<<<<<<<<
        # # agent.invoke() is a single blocking call that runs CUGA's entire
        # # multi-turn loop.  When it returns, CUGA is completely done and
        # # InvokeResult.answer holds the final text written by FinalAnswerAgent.
        # # We now call the benchmark's finish/submit action on CUGA's behalf.
        # if not _finish_called[0]:
        #     self._call_finish_action(result, fn_by_name)
        # >>>>>>>>>> FINISH-BRIDGE END <<<<<<<<<<

    def close(self) -> None:
        super().close()

    def get_cost(self) -> CostReport:
        if self._token_accumulator is None:
            return LiteLLMCostReport.initialize_empty(model_name=self.model_id)
        return LiteLLMCostReport.from_token_counts(
            model_name=self.model_id,
            input_tokens=self._token_accumulator.input_tokens,
            output_tokens=self._token_accumulator.output_tokens,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # >>>>>>>>>> FINISH-BRIDGE BEGIN (remove when CUGA calls finish natively) <<<<<<<<<<
    def _call_finish_action(self, result: Any, fn_by_name: dict[str, Callable]) -> None:
        """Call the benchmark's is_finish action using CUGA's final answer."""
        finish_action_type = next((a for a in self.actions if a.is_finish), None)
        if finish_action_type is None:
            return

        fn_name = finish_action_type.name.replace(".", "__")
        finish_fn = fn_by_name.get(fn_name)
        if finish_fn is None:
            print(
                f"[CUGA finish] No callable found for finish action '{fn_name}' — skipping.",
                flush=True,
            )
            return

        raw_answer: str | None = getattr(result, "answer", None)

        kwargs: dict[str, Any] = {}
        try:
            fields = finish_action_type.arguments.model_fields
        except Exception:
            fields = {}

        if "answer" in fields:
            answer_value: Any = None
            if raw_answer and raw_answer.strip():
                try:
                    answer_value = json.loads(raw_answer.strip())
                except (json.JSONDecodeError, ValueError):
                    answer_value = None
            kwargs["answer"] = answer_value
        if "status" in fields:
            kwargs["status"] = "success"

        print(
            f"[CUGA finish] Calling '{fn_name}' with kwargs={kwargs!r} "
            f"(cuga_answer={raw_answer!r})",
            flush=True,
        )
        try:
            finish_fn(**kwargs)
        except Exception as exc:
            print(f"[CUGA finish] Finish action raised: {exc}", flush=True)
    # >>>>>>>>>> FINISH-BRIDGE END <<<<<<<<<<

    def _build_prompt(self) -> str:
        prompt = ""
        if self.context:
            try:
                prompt += f"Context: {json.dumps(self.context, ensure_ascii=False, default=str)}\n\n"
            except TypeError:
                prompt += f"Context: {self.context}\n\n"

        prompt += (
            "Complete this task using the available tools. Each tool corresponds to an action "
            "you can take in the environment. Do not respond or ask clarification questions "
            "unless done through a dedicated tool, and only if such tool exists. "
            "Any plain message that is not a tool call will end the run in failure.\n"
        )

        if self.initial_observation is not None:
            text = str(self.initial_observation).strip()
            if text and text.lower() != "none":
                prompt += f"\nFirst Observation: {text}\n"

        prompt += f"\nTask: {self.task}\n"

        # >>>>>>>>>> FINISH-HINT BEGIN <<<<<<<<<<
        # Instruct CUGA to call the completion tool at the end of the task.
        # Placed in the task prompt (not the system prompt) so it is read once
        # as part of the goal — not repeated every turn, which caused CUGA to
        # rush to finish before completing the actual work.
        finish_action = next((a for a in self.actions if a.is_finish), None)
        if finish_action is not None:
            # >>>>>>>>>> FINISH-HINT-EXACT-NAME BEGIN <<<<<<<<<<
            # Use the exact tool name rather than the "[TASK COMPLETION TOOL]" marker.
            # CUGA ignores the marker and follows its own "return text when done" rule,
            # so it never calls the finish tool unless given the precise name.
            # Remove this block (and restore the original string below) if CUGA gains
            # native awareness of the completion tool without name-hinting.
            fn_name = finish_action.name.replace(".", "__")
            cuga_finish_name = fn_name.replace("__", "_")
            prompt += (
                f"\nWhen you have completed all required actions, call `{cuga_finish_name}` "
                "as your final step. "
                "Pass `answer=None` if the task only required performing actions, or pass "
                "the specific value if the task asked you to find or return something "
                "(a name, a number, a date, etc.). "
                "Always pass `status=\"success\"` if the tool accepts it."
            )
            # >>>>>>>>>> FINISH-HINT-EXACT-NAME END <<<<<<<<<<
        # >>>>>>>>>> FINISH-HINT END <<<<<<<<<<

        return prompt

    @staticmethod
    def _get_cuga_agent_class() -> Any:
        try:
            from cuga import CugaAgent
        except Exception as exc:
            raise RuntimeError(
                "CUGA SDK is not installed. Install optional deps with `pip install -e '.[cuga]'` "
                "or run `exgentic setup --agent cuga_sdk`."
            ) from exc
        return CugaAgent
