# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import json
import re
import uuid
from typing import Any, ClassVar

from ...core.actions import build_unknown_action
from ...core.agent import Agent
from ...core.agent_instance import AgentInstance
from ...core.types import (
    Action,
    ActionType,
    Message,
    MessageAction,
    MessageObservation,
    MessagePayload,
    ModelSettings,
    Observation,
    ParallelAction,
    SingleAction,
)
from ...utils.sync import run_sync
from ..litellm_tool_calling.utils import ToolCall, ToolsActionsRegistry


def _normalize_cuga_tool_entry(entry: dict[str, Any]) -> ToolCall | None:
    """Map CUGA / LangChain-style tool call dicts to Exgentic ToolCall."""
    name = entry.get("name") or entry.get("tool_name") or entry.get("function", {}).get("name")
    if not name:
        return None
    name = str(name)
    raw_id = entry.get("id") or entry.get("tool_call_id") or f"call_{uuid.uuid4().hex[:24]}"
    args: Any = (
        entry.get("arguments")
        or entry.get("args")
        or entry.get("input")
        or entry.get("function", {}).get("arguments")
    )
    if isinstance(args, str):
        arg_str = args
    else:
        try:
            arg_str = json.dumps(args, ensure_ascii=False, default=str)
        except TypeError:
            arg_str = str(args)
    return ToolCall(name=name, arguments=arg_str, id=str(raw_id))


def _dedupe_tool_calls(calls: list[ToolCall]) -> list[ToolCall]:
    seen: set[tuple[str, str]] = set()
    out: list[ToolCall] = []
    for c in calls:
        key = (c["name"], c["arguments"])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


class CUGASDKAgentInstance(AgentInstance):
    """CUGA SDK-backed agent instance.

    Primary bridge: LangChain StructuredTool objects built from benchmark ActionType definitions
    (same schema path as ``ToolsActionsRegistry``), ``CugaAgent(tools=...)``, and
    ``InvokeResult.tool_calls`` / capture-buffer normalization into Exgentic ``Action``.

    Fallback: legacy JSON-in-text parsing from ``InvokeResult.answer`` when no tool calls appear.
    """

    def __init__(
        self,
        session_id: str,
        task: str,
        context: dict[str, Any] | None = None,
        actions: list[ActionType] | None = None,
        max_steps: int = 150,
        enable_action_shortlisting: bool = True,
        max_selected_actions: int = 50,
        use_legacy_json_fallback: bool = True,
    ) -> None:
        super().__init__(session_id)
        self.task = task
        self.context = context or {}
        self._action_types = list(actions or [])
        self.max_steps = max_steps
        self._step_count = 0
        self._transcript_lines: list[str] = []
        self._registry = ToolsActionsRegistry(self._action_types)
        self._actions_by_name = {a.name: a for a in self._action_types}
        self._finish_names = {a.name for a in self._action_types if a.is_finish}
        self.enable_action_shortlisting = enable_action_shortlisting
        self.max_selected_actions = max_selected_actions
        self.use_legacy_json_fallback = use_legacy_json_fallback
        self._tool_capture_buffer: list[ToolCall] = []

        env_action_types = [a for a in self._action_types if not a.is_message]
        self._non_message_action_types = env_action_types

    def _shortlist_action_types(self, observation_text: str) -> list[ActionType]:
        """Heuristic shortlist when the tool count exceeds ``max_selected_actions``."""
        if not self.enable_action_shortlisting:
            return self._non_message_action_types
        cap = max(1, self.max_selected_actions)
        if len(self._non_message_action_types) <= cap:
            return self._non_message_action_types

        blob = f"{self.task}\n{observation_text}".lower()
        words = {w for w in re.findall(r"\w+", blob) if len(w) > 2}
        scored: list[tuple[int, ActionType]] = []
        for a in self._non_message_action_types:
            if a.is_finish:
                continue
            desc = f"{a.name} {a.description}".lower()
            score = sum(1 for w in words if w in desc)
            scored.append((score, a))
        scored.sort(key=lambda x: (-x[0], x[1].name))

        finish_actions = [a for a in self._non_message_action_types if a.is_finish]
        non_finish_budget = max(0, cap - len(finish_actions))
        picked: list[ActionType] = [a for _, a in scored[:non_finish_budget]]
        for fa in finish_actions:
            if fa not in picked:
                picked.append(fa)
        return picked[:cap]

    def _build_langchain_tools(self, action_subset: list[ActionType]) -> list[Any]:
        try:
            from langchain_core.tools import StructuredTool
        except Exception as exc:
            raise RuntimeError(
                "langchain_core is required for CUGA tool bridge (install optional `cuga` extra)."
            ) from exc

        tools: list[Any] = []
        for action_type in action_subset:
            at = action_type

            def _make_impl(at_ref: ActionType):
                def _impl(**kwargs: Any) -> str:
                    try:
                        arg_str = json.dumps(kwargs, ensure_ascii=False, default=str)
                    except TypeError:
                        arg_str = str(kwargs)
                    tid = f"call_{uuid.uuid4().hex[:24]}"
                    self._tool_capture_buffer.append(
                        ToolCall(name=at_ref.name, arguments=arg_str, id=tid),
                    )
                    return json.dumps({"recorded": at_ref.name, "id": tid}, ensure_ascii=False)

                return _impl

            tools.append(
                StructuredTool.from_function(
                    name=at.name,
                    description=at.description or at.name,
                    func=_make_impl(at),
                    args_schema=at.arguments,  # type: ignore[arg-type]
                ),
            )
        return tools

    def _get_cuga_agent_class(self) -> Any:
        try:
            from cuga import CugaAgent
        except Exception as exc:
            raise RuntimeError(
                "CUGA SDK is not installed. Install optional deps with `pip install -e '.[cuga]'` "
                "or run `exgentic setup --agent cuga_sdk`."
            ) from exc
        return CugaAgent

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        text = text.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
        return None

    @staticmethod
    def _observation_to_text(observation: Observation | None) -> str:
        if observation is None:
            return "No observation yet."
        chunks: list[str] = []
        for idx, obs in enumerate(observation.to_observation_list(), start=1):
            result = obs.result
            if isinstance(obs, MessageObservation) and isinstance(obs.result, MessagePayload):
                result = {"sender": obs.result.sender, "message": obs.result.message}
            chunks.append(
                json.dumps(
                    {
                        "index": idx,
                        "result": result,
                        "invoking_actions": [a.name for a in obs.invoking_actions],
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            )
        return "\n".join(chunks)

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        answer = getattr(response, "answer", None)
        if isinstance(answer, str):
            return answer
        if isinstance(response, dict):
            for key in ("answer", "response", "content", "text"):
                value = response.get(key)
                if isinstance(value, str):
                    return value
        return str(response)

    def _build_action_from_json_payload(self, payload: dict[str, Any]) -> Action | None:
        if isinstance(payload.get("actions"), list):
            collected: list[SingleAction] = []
            for item in payload["actions"]:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("action") or item.get("name") or "").strip()
                if not name:
                    continue
                arguments = item.get("arguments", {})
                action_type = self._actions_by_name.get(name)
                action = (
                    action_type.build_action(arguments)
                    if action_type is not None
                    else build_unknown_action(name, arguments)
                )
                collected.append(action)
            if not collected:
                return None
            if len(collected) == 1:
                return collected[0]
            return ParallelAction(actions=collected)

        name = str(payload.get("action") or payload.get("name") or "").strip()
        if not name:
            return None
        arguments = payload.get("arguments", {})
        action_type = self._actions_by_name.get(name)
        if action_type is None:
            return build_unknown_action(name, arguments)
        return action_type.build_action(arguments)

    def _fallback_message_action(self, response_text: str) -> Action | None:
        message_action_type = next((a for a in self._action_types if a.is_message), None)
        if message_action_type is None and "message" in self._actions_by_name:
            message_action_type = self._actions_by_name["message"]
        if message_action_type is None:
            return None
        return MessageAction(arguments=Message(content=response_text))

    def _compose_turn_message(self, observation_text: str, step_label: int) -> str:
        header = (
            "You are driving an Exgentic benchmark session. "
            "Use the provided tools to act in the environment. "
            "Prefer calling tools over plain prose. "
            "If the task requires finishing or submitting an answer, use the appropriate finish tool when ready.\n"
        )
        ctx = ""
        if self.context:
            ctx = f"CONTEXT_JSON:\n{json.dumps(self.context, ensure_ascii=False, default=str)}\n\n"
        transcript = ""
        if self._transcript_lines:
            transcript = "PRIOR_STEPS:\n" + "\n".join(self._transcript_lines) + "\n\n"
        return (
            f"{header}"
            f"TASK:\n{self.task}\n\n"
            f"{ctx}"
            f"{transcript}"
            f"STEP_{step_label}_OBSERVATION:\n{observation_text}\n"
        )

    async def _ainvoke_cuga(self, message: str, tools: list[Any]) -> Any:
        cuga_agent_cls = self._get_cuga_agent_class()
        agent = cuga_agent_cls(tools=tools)
        return await agent.invoke(message, thread_id=self.session_id, track_tool_calls=True)

    def _tool_calls_from_invoke_result(self, result: Any) -> list[ToolCall]:
        raw = getattr(result, "tool_calls", None) or []
        out: list[ToolCall] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            norm = _normalize_cuga_tool_entry(item)
            if norm is not None:
                out.append(norm)
        return out

    def react(self, observation: Observation | None) -> Action | None:
        self._step_count += 1
        if self._step_count > self.max_steps:
            self.logger.warning("Finished: max steps reached (%d)", self.max_steps)
            return None

        observation_text = self._observation_to_text(observation)
        self._tool_capture_buffer.clear()

        action_subset = self._shortlist_action_types(observation_text)
        tools = self._build_langchain_tools(action_subset)

        message = self._compose_turn_message(observation_text, self._step_count)
        err: str | None = None
        response_text = ""
        try:
            result = run_sync(self._ainvoke_cuga(message, tools), timeout=300.0)
            err = getattr(result, "error", None)
            response_text = self._extract_response_text(result)
            if err:
                self.logger.warning("CUGA invoke reported error: %s", err)
        except Exception as exc:
            self.logger.exception("CUGA invoke failed: %s", exc)
            if self.use_legacy_json_fallback:
                return self._fallback_message_action(f"CUGA error: {exc!s}")
            raise

        combined_calls = list(self._tool_capture_buffer)
        combined_calls.extend(self._tool_calls_from_invoke_result(result))
        combined_calls = _dedupe_tool_calls(combined_calls)

        if combined_calls:
            action = self._registry.tool_calls_to_action(combined_calls)
            if action is not None:
                self._transcript_lines.append(
                    f"step_{self._step_count}: acted with tool calls: {[c['name'] for c in combined_calls]}",
                )
                return action

        if self.use_legacy_json_fallback:
            payload = self._extract_json_object(response_text)
            if payload is not None:
                action = self._build_action_from_json_payload(payload)
                if action is not None:
                    self._transcript_lines.append(f"step_{self._step_count}: legacy JSON action")
                    return action
            fallback = self._fallback_message_action(response_text)
            if fallback is not None:
                self._transcript_lines.append(f"step_{self._step_count}: message fallback")
            return fallback

        return None

    def close(self) -> None:
        return None


class CUGASDKAgent(Agent):
    display_name: ClassVar[str] = "CUGA SDK"
    slug_name: ClassVar[str] = "cuga_sdk"

    max_steps: int = 150
    model_settings: ModelSettings | None = None
    enable_action_shortlisting: bool = False
    max_selected_actions: int = 50
    use_legacy_json_fallback: bool = True

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
            enable_action_shortlisting=self.enable_action_shortlisting,
            max_selected_actions=self.max_selected_actions,
            use_legacy_json_fallback=self.use_legacy_json_fallback,
        )
