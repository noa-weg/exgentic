# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""CUGA SDK agent instance — CodeAgentInstance implementation.

Architecture overview
---------------------
CUGA is a multi-turn agentic system that runs its own internal planning,
tool-selection, execution, and reflection loop.  This module bridges CUGA
into exgentic's ``CodeAgentInstance`` pattern so CUGA runs correctly inside
the benchmark evaluation harness.

Threading model
~~~~~~~~~~~~~~~
exgentic's ``AgentCoordinator`` spawns a dedicated agent thread and drives
it by passing observations one at a time.  ``CodeAgentInstance`` subclasses
work differently: ``run_code_agent()`` is called **once** on the agent thread
and is expected to run the agent's full loop to completion before returning.
Tool calls inside that loop are real callables that block on
``self.execute(action)`` — which posts the action to the coordinator thread,
waits for the benchmark environment to execute it, and returns the
observation.  This gives CUGA accurate, live results to reason over.

    env thread                     agent thread
    ----------                     ------------
    coordinator.react(obs) ──────► run_code_agent() starts
                                   agent.invoke(prompt, ...)
                                     CUGA internal loop begins
                                     CUGA calls tool fn (StructuredTool.func)
                                       └─ fn calls self.execute(action)
                                            └─ posts action, waits on condition
    condition notified ◄───────────  benchmark executes action, returns obs
    obs returned        ──────────►  condition.wait() returns with obs
                                     tool fn returns obs string to CUGA
                                     CUGA continues next turn …
                                   agent.invoke() returns
                                   run_code_agent() returns
    coordinator done    ◄──────────  execute(None) signals done

Tool naming
~~~~~~~~~~~
exgentic uses ``"."`` as the action-name separator (e.g. ``phone.login``) and
``"__"`` as the StructuredTool name separator (e.g. ``phone__login``).  CUGA's
sandbox rejects any identifier containing ``"__"`` as a suspected dunder
security violation, so every ``"__"`` is replaced with a single ``"_"`` in
the tool name exposed to CUGA (e.g. ``phone_login``).  App grouping therefore
cannot rely on splitting the tool name — it uses ``metadata["server_name"]``
instead (set explicitly on each StructuredTool before passing to the provider).
"""

from __future__ import annotations

import json
from typing import Any, Callable, List

from ...adapters.agents.code_agent import CodeAgentInstance
from ...core.types import ModelSettings
from ...utils.cost import CostReport, LiteLLMCostReport
from ...utils.sync import run_sync


class _TokenAccumulator:
    """Accumulates token usage across all LLM calls made during a CUGA run.

    CUGA executes multiple internal LLM calls per task (planner, tool selector,
    executor, reflector, final-answer agent).  Each call fires LangChain's
    ``on_llm_end`` callback with token counts in ``response.llm_output``.  This
    class sums those counts so ``CUGASDKAgentInstance.get_cost()`` can report
    the total cost for the entire task as a single ``CostReport``.

    The LangChain ``BaseCallbackHandler`` subclass is created lazily on first
    access of ``handler`` so that ``langchain_core`` is only imported when CUGA
    is actually used — keeping exgentic importable without langchain installed.
    """

    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self._handler: Any = None

    def _ensure_handler(self) -> Any:
        """Create and cache the LangChain callback handler on first use."""
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
        """The LangChain BaseCallbackHandler instance, created lazily."""
        return self._ensure_handler()


class MultiAppToolProvider:
    """Groups LangChain StructuredTools by app and exposes them to CUGA.

    CUGA's tool-provider interface lets it discover tools per application
    (e.g. ``phone``, ``spotify``) rather than receiving a single flat list.
    This is important for large tool sets (e.g. AppWorld's ~468 tools): when
    the total count exceeds CUGA's ``shortlisting_tool_threshold`` (default 35),
    CUGA switches to ``find_tools`` mode and queries this provider by app name
    to narrow the candidate set before each action.

    Grouping strategy
    ~~~~~~~~~~~~~~~~~
    Each tool must have ``metadata["server_name"]`` set to its app name before
    being passed here.  Tools without a ``server_name`` fall back to the
    ``"runtime_tools"`` bucket — this covers flat-namespace benchmarks (e.g.
    TAU2) where tools have no app prefix.

    The ``finish`` tool problem
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    AppWorld's task-completion action is named ``"finish"`` with no app prefix,
    so it lands in ``"runtime_tools"``.  CUGA searches for tools by app name,
    so if it searches ``"supervisor"`` it would miss ``"finish"`` entirely.
    ``get_tools()`` therefore always appends the ``"runtime_tools"`` bucket to
    every per-app result, ensuring finish (and any other app-less tools) are
    always visible regardless of which app CUGA queries.
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
        # Always append runtime_tools alongside the requested app's tools.
        # runtime_tools holds tools that have no app prefix (e.g. AppWorld's
        # "finish" action).  Without this, CUGA would never find "finish" when
        # searching any named app, causing the task to end without a completion
        # signal (action_count=0 symptom).
        # For flat-namespace benchmarks (TAU2) all tools are already in
        # runtime_tools, so this is a no-op for those cases.
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
    """CUGA SDK agent instance — runs CUGA's full multi-turn loop for one task.

    This class implements the ``CodeAgentInstance`` interface, which means
    ``run_code_agent()`` is called once per task with a list of real callable
    functions (one per action type).  It wraps those functions as LangChain
    ``StructuredTool`` objects, hands them to a ``MultiAppToolProvider``, and
    invokes CUGA's ``CugaAgent.invoke()`` which runs the entire planning +
    execution + reflection loop internally.

    Each tool callable, when invoked by CUGA, calls ``self.execute(action)``
    which synchronises with the benchmark environment via the
    ``AgentCoordinator`` thread bridge (see module docstring).

    Cost tracking
    ~~~~~~~~~~~~~
    A ``_TokenAccumulator`` callback handler is registered with CUGA's
    LangGraph invocation so token counts from every internal LLM call are
    summed and exposed via ``get_cost()``.
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
        """Run CUGA's full multi-turn loop for the current task.

        Sets required environment variables, wraps each action as a
        LangChain ``StructuredTool``, creates a ``MultiAppToolProvider``,
        and calls ``CugaAgent.invoke()`` which blocks until CUGA has
        completed (or exhausted) its internal loop.
        """
        import os as _os

        # --- Environment variable setup ---
        # CUGA reads configuration at module-import time via dynaconf, so
        # these must be set before the first ``import cuga`` statement
        # (triggered below by ``_get_cuga_agent_class()``).

        # Select CUGA's settings file based on the provider prefix in model_id
        # (e.g. "groq/openai/gpt-oss-120b" → provider "groq" → settings.groq.toml).
        # Only set if not already overridden in the environment.
        _provider = self.model_id.split("/")[0].lower() if "/" in self.model_id else ""
        if "AGENT_SETTING_CONFIG" not in _os.environ:
            if _provider == "groq" or _os.environ.get("GROQ_API_KEY"):
                _os.environ["AGENT_SETTING_CONFIG"] = "settings.groq.toml"

        # Strip the provider prefix so CUGA receives its native model string.
        # e.g. "groq/openai/gpt-oss-120b" → MODEL_NAME = "openai/gpt-oss-120b"
        # Without this, CUGA falls back to the Groq platform default (gpt-oss-20b)
        # which is too small for complex tool-use tasks.
        if "MODEL_NAME" not in _os.environ and _provider:
            _os.environ["MODEL_NAME"] = self.model_id[len(_provider) + 1:]

        # Always honour the configured cuga_mode (overrides any env default).
        _os.environ["DYNACONF_FEATURES__CUGA_MODE"] = self.cuga_mode

        # Disable CUGA's chat mode — exgentic drives the agent programmatically,
        # not through an interactive chat interface.
        if "DYNACONF_FEATURES__CHAT" not in _os.environ:
            _os.environ["DYNACONF_FEATURES__CHAT"] = "false"

        try:
            from langchain_core.tools import StructuredTool
        except Exception as exc:
            raise RuntimeError(
                "langchain_core is required for the CUGA tool bridge "
                "(install optional `cuga` extra)."
            ) from exc

        # --- Build LangChain StructuredTools from exgentic ActionTypes ---
        # Each ActionType has a Pydantic args_schema and a matching callable
        # (produced by action_type_to_function()).  We expose these to CUGA as
        # StructuredTools so CUGA can call them like normal Python functions.
        fn_by_name = {fn.__name__: fn for fn in functions}
        all_tools: list[Any] = []
        for action_type in self.actions:
            # Hidden actions (e.g. internal signals) are not exposed to the LLM.
            if action_type.is_hidden:
                continue
            fn_name = action_type.name.replace(".", "__")
            fn = fn_by_name.get(fn_name)
            if fn is None:
                continue

            # Derive the app name from the function name for provider grouping.
            # "phone__login" → app "phone"; "finish" → app "runtime_tools".
            app_name = fn_name.split("__")[0] if "__" in fn_name else "runtime_tools"

            # Replace "__" with "_" in the tool name exposed to CUGA.
            # CUGA's sandbox treats any identifier containing "__" as a dunder
            # security violation and rejects the generated code.
            cuga_tool_name = fn_name.replace("__", "_")

            # Annotate the finish and message tools so CUGA knows their roles.
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
            # Store the app name in metadata so MultiAppToolProvider can group
            # tools by app without relying on the sanitized tool name.
            tool.metadata = {"server_name": app_name}
            all_tools.append(tool)

        # --- Create provider and agent ---
        # shortlist_threshold=35 matches CUGA's default: when tool count exceeds
        # this, CUGA activates find_tools mode and queries the provider by app
        # name rather than loading all tools at once.
        shortlist_threshold = 35

        provider = MultiAppToolProvider(tools=all_tools)
        cuga_agent_cls = self._get_cuga_agent_class()
        agent = cuga_agent_cls(tool_provider=provider)

        # --- Invoke CUGA ---
        # Register the token accumulator as a LangChain callback so every
        # internal LLM call contributes to cost reporting.
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

    def close(self) -> None:
        super().close()

    def get_cost(self) -> CostReport:
        """Return total LLM cost accumulated across all of CUGA's internal calls."""
        if self._token_accumulator is None:
            return LiteLLMCostReport.initialize_empty(model_name=self.model_id)
        return LiteLLMCostReport.from_token_counts(
            model_name=self.model_id,
            input_tokens=self._token_accumulator.input_tokens,
            output_tokens=self._token_accumulator.output_tokens,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self) -> str:
        """Build the initial task prompt passed to CUGA's agent.invoke().

        The prompt includes:
        - Optional context (JSON-serialised) from the benchmark environment
        - A standing instruction reminding CUGA to only communicate via tools
        - The initial observation, if the benchmark provided one at task start
        - The task description
        - An explicit instruction to call the finish tool by name when done,
          including guidance on how to pass the ``answer`` and ``status`` args

        The finish-tool hint uses the exact tool name (e.g. ``supervisor_finish``)
        rather than the ``[TASK COMPLETION TOOL]`` description marker because
        CUGA follows its own "return text when done" rule by default and only
        calls the finish tool when given the precise function name.

        The hint is placed in the task prompt (not in a system prompt) so it is
        read once as part of the goal statement.  Putting it in the system
        prompt caused CUGA to rush to finish before completing the actual work.
        """
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

        finish_action = next((a for a in self.actions if a.is_finish), None)
        if finish_action is not None:
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

        return prompt

    @staticmethod
    def _get_cuga_agent_class() -> Any:
        """Import and return the CugaAgent class from the cuga package.

        Raises a clear RuntimeError if the cuga package is not installed,
        pointing the user to the setup command.
        """
        try:
            from cuga import CugaAgent
        except Exception as exc:
            raise RuntimeError(
                "CUGA SDK is not installed. Install optional deps with `pip install -e '.[cuga]'` "
                "or run `exgentic setup --agent cuga_sdk`."
            ) from exc
        return CugaAgent
