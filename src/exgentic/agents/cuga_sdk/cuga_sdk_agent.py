# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
import re
from typing import Any, Callable, ClassVar, List

from ...adapters.agents.code_agent import CodeAgentInstance
from ...core.agent import Agent
from ...core.agent_instance import AgentInstance
from ...core.types import (
    ActionType,
    ModelSettings,
)
from ...utils.sync import run_sync


class MultiAppToolProvider:
    """Splits a flat list of LangChain StructuredTools into per-app groups.

    Tools named ``app__operation`` (double-underscore separator, as used by AppWorld)
    are grouped under the ``app`` key so that CUGA's ``find_tools("...", "phone")``
    resolves correctly.  Tools without a ``__`` in their name fall back to the
    "runtime_tools" bucket — this is the normal path for TAU2 and other benchmarks
    with flat tool namespaces.
    """

    def __init__(self, tools: list[Any]) -> None:
        self._apps: dict[str, list[Any]] = {}
        for t in tools:
            app = t.name.split("_")[0] if "_" in t.name else "runtime_tools"
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
        return self._apps.get(app_name, [])

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
        task: str,
        context: dict[str, Any] | None = None,
        actions: list[ActionType] | None = None,
        max_steps: int = 150,
        model_settings: ModelSettings | None = None,
        max_tools: int = 60,
    ) -> None:
        super().__init__(session_id, task, context or {}, list(actions or []))
        self.max_steps = max_steps
        self.model_settings = model_settings
        self.max_tools = max_tools

    # ------------------------------------------------------------------
    # CodeAgentInstance interface
    # ------------------------------------------------------------------

    def run_code_agent(self, functions: List[Callable]) -> None:
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

        # TEMPORARILY DISABLED: _select_tools() pre-selection bypassed to observe
        # raw CUGA behaviour with all tools exposed.
        # selected_tools = self._select_tools(all_tools)
        # threshold = len(selected_tools) + 10
        selected_tools = all_tools

        # Debug: print selected tools per app.
        from collections import Counter as _Counter
        _app_counts = _Counter(
            (t.name.split("_")[0] if "_" in t.name else "runtime_tools")
            for t in selected_tools
        )
        print(
            f"[CUGA tool selection] {len(selected_tools)}/{len(all_tools)} tools selected; "
            f"apps={dict(_app_counts)}; names={[t.name for t in selected_tools]}",
            flush=True,
        )

        # Use CUGA's default threshold so find_tools mode activates naturally
        # when tool count is large (e.g. AppWorld ~480 tools).
        threshold = 35

        provider = MultiAppToolProvider(tools=selected_tools)

        cuga_agent_cls = self._get_cuga_agent_class()
        agent = cuga_agent_cls(tool_provider=provider)

        prompt = self._build_prompt()
        run_sync(
            agent.invoke(
                prompt,
                thread_id=self.session_id,
                track_tool_calls=True,
                config={"configurable": {"shortlisting_tool_threshold": threshold}},
            ),
            timeout=600.0,
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_tools(self, tools: list[Any]) -> list[Any]:
        """Return a relevance-scored subset of tools up to ``self.max_tools``.

        App relevance is determined by whether the app's *name* appears as a
        substring in the task keywords (or vice versa).  This prevents generic
        keywords like "reset" from accidentally marking every app as relevant
        (because each app has a reset_password tool).

        The budget is distributed evenly across the relevant apps.  Within each
        app, tools are ranked by how many task keywords appear in their name or
        description.  Finish-type tools are always included.
        """
        if len(tools) <= self.max_tools:
            return tools

        # Build keyword set from task text only.
        blob = self.task.lower()
        keywords = {w for w in re.findall(r"\w+", blob) if len(w) > 2}

        finish_tools: list[Any] = []
        candidate_tools: list[Any] = []
        # Tools are registered with sanitized names (single "_"), so build the
        # lookup using the same sanitization: "." → "__" → "_".
        action_types_by_name = {a.name.replace(".", "_"): a for a in self.actions}

        for t in tools:
            at = action_types_by_name.get(t.name)
            if at is not None and at.is_finish:
                finish_tools.append(t)
            else:
                candidate_tools.append(t)

        def tool_score(t: Any) -> int:
            desc = f"{t.name} {t.description or ''}".lower()
            return sum(1 for w in keywords if w in desc)

        # Group candidates by app prefix.
        app_tools: dict[str, list[Any]] = {}
        for t in candidate_tools:
            # Tool names use single "_" separator; first component is the app.
            app = t.name.split("_")[0] if "_" in t.name else "runtime_tools"
            app_tools.setdefault(app, []).append(t)

        # Score each app by whether its name appears in the task keyword set.
        # Using name-level matching avoids generic keywords (e.g. "reset") from
        # making every app appear relevant just because each has a reset_password tool.
        def app_name_score(app: str) -> int:
            return sum(1 for w in keywords if w in app or app in w)

        app_score = {app: app_name_score(app) for app in app_tools}

        relevant_apps = [a for a in app_tools if app_score[a] > 0]
        relevant_apps.sort(key=lambda a: app_score[a], reverse=True)

        budget = max(0, self.max_tools - len(finish_tools))

        if not relevant_apps:
            # No app names matched any task keyword — fall back to flat top-N
            # by individual tool description score.
            candidate_tools.sort(key=tool_score, reverse=True)
            return candidate_tools[:budget] + finish_tools

        # Distribute budget evenly across relevant apps.
        per_app = max(1, budget // len(relevant_apps))
        selected: list[Any] = []
        remainder: list[Any] = []

        for app in relevant_apps:
            app_ts = sorted(app_tools[app], key=tool_score, reverse=True)
            take = min(len(app_ts), per_app)
            selected.extend(app_ts[:take])
            remainder.extend(app_ts[take:])

        # Fill remaining slots from leftover tools, highest tool score first.
        remaining = budget - len(selected)
        if remaining > 0:
            remainder.sort(key=tool_score, reverse=True)
            selected.extend(remainder[:remaining])

        return selected + finish_tools

    # >>>>>>>>>> FINISH-BRIDGE BEGIN (remove when CUGA calls finish natively) <<<<<<<<<<
    def _call_finish_action(self, result: Any, fn_by_name: dict[str, Callable]) -> None:
        """Call the benchmark's is_finish action using CUGA's final answer.

        CUGA's ``FinalAnswerAgent`` writes a plain-text answer and routes to END
        — it never calls tools.  This method bridges that gap: after
        ``agent.invoke()`` returns we find the ``is_finish=True`` action, build
        kwargs from the args schema, and call the real function so the benchmark
        receives a proper completion signal.

        The answer value is taken from ``InvokeResult.answer`` (the text written
        by FinalAnswerAgent).  An empty string is normalised to ``None`` because
        most AppWorld tasks expect a JSON-null answer for action-only tasks.
        """
        finish_action_type = next((a for a in self.actions if a.is_finish), None)
        if finish_action_type is None:
            # Benchmark has no finish action (e.g. TAU-2 which uses message).
            return

        fn_name = finish_action_type.name.replace(".", "__")
        finish_fn = fn_by_name.get(fn_name)
        if finish_fn is None:
            print(
                f"[CUGA finish] No callable found for finish action '{fn_name}' — skipping.",
                flush=True,
            )
            return

        # Extract answer text from InvokeResult (or any object with .answer).
        raw_answer: str | None = getattr(result, "answer", None)

        # Build kwargs from the finish action's Pydantic arguments schema.
        kwargs: dict[str, Any] = {}
        try:
            fields = finish_action_type.arguments.model_fields
        except Exception:
            fields = {}

        if "answer" in fields:
            # CUGA's FinalAnswerAgent always produces natural‑language summaries
            # ("I have created the playlist…") rather than bare JSON values.
            # AppWorld action‑only tasks expect a JSON‑null answer; a prose string
            # would fail the `assert answers match` test.  We therefore try to
            # parse the text as JSON: if it succeeds (e.g. the agent output "42"
            # or '"Invisible Chains"'), we use the parsed value; if parsing fails
            # (i.e. it is a prose summary), we fall back to None which serialises
            # as JSON null — the correct answer for action‑only tasks.
            answer_value: Any = None
            if raw_answer and raw_answer.strip():
                import json as _json
                try:
                    answer_value = _json.loads(raw_answer.strip())
                except (_json.JSONDecodeError, ValueError):
                    # Natural‑language output → treat as null (action‑only).
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
            prompt += (
                "\nWhen you have completed all required actions, call the tool marked "
                "[TASK COMPLETION TOOL] as your final step. "
                "Pass `answer=None` if the task only required performing actions, or pass "
                "the specific value if the task asked you to find or return something "
                "(a name, a number, a date, etc.). "
                "Always pass `status=\"success\"` if the tool accepts it."
            )
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


class CUGASDKAgent(Agent):
    display_name: ClassVar[str] = "CUGA SDK"
    slug_name: ClassVar[str] = "cuga_sdk"

    max_steps: int = 150
    model_settings: ModelSettings | None = None
    # Maximum number of tools to expose per CUGA invocation.  Tools are scored
    # by relevance to the task; finish tools are always included.  Setting this
    # below CUGA's find_tools threshold (default 35) ensures all selected tools
    # appear directly in the model's prompt rather than behind find_tools.
    max_tools: int = 60

    def assign(
        self,
        task: str,
        context: dict[str, Any],
        actions: list[ActionType],
        session_id: str,
    ) -> AgentInstance:
        return CUGASDKAgentInstance(
            session_id=session_id,
            task=task,
            context=context,
            actions=actions,
            max_steps=self.max_steps,
            model_settings=self.model_settings,
            max_tools=self.max_tools,
        )
