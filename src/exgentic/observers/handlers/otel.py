# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from typing import Dict, Optional, cast
from pathlib import Path
from datetime import datetime

from opentelemetry import trace, context
from opentelemetry.util.types import AttributeValue
from opentelemetry.trace import Tracer
from opentelemetry.sdk.trace import Span
from opentelemetry.trace.status import Status, StatusCode

from ...utils.paths import get_run_id
from ...utils.otel import (
    init_tracing_from_env,
    flush_traces,
    get_session_logger,
    write_attribute,
    safe_repr,
)
from ...core.orchestrator.observer import Observer
from ...core.context import get_context, set_context, OtelContext
from ...interfaces.registry import get_agent_entries, get_benchmark_entries


tracer = init_tracing_from_env()


class SessionSpanManager:
    """Manages isolated span hierarchy for a single session."""

    def __init__(self, session_id: str, session_root: Path, tracer: Tracer = tracer):
        self.session_id = session_id
        self._tracer = tracer
        self._span_stack: list[Span] = []
        self._heritable_attributes: Dict[str, AttributeValue] = {}

        # Initialize session logger
        self._logger = get_session_logger(session_root, __name__)
        self._logger.info(
            f"SessionSpanManager initialized for session {self.session_id}"
        )

    def start_span(self, name: str, **kwargs) -> Span:
        """Start a new span as a child of the current span.

        Args:
            name: Name of the span
            **kwargs: Additional arguments passed to tracer.start_span
        """
        parent_span = self._span_stack[-1] if self._span_stack else None
        ctx = (
            trace.set_span_in_context(parent_span)
            if parent_span
            else context.get_current()
        )

        span = cast(Span, self._tracer.start_span(name, context=ctx, **kwargs))
        self._span_stack.append(span)

        # Apply heritable attributes to new span
        for k, v in self._heritable_attributes.items():
            write_attribute(span, k, v)

        # Log span start
        span_ctx = span.get_span_context()
        parent_span_id = (
            format(parent_span.get_span_context().span_id, "016x")
            if parent_span
            else None
        )
        start_time = datetime.fromtimestamp(span.start_time / 1_000_000_000)

        self._logger.log_span_start(
            span_name=name,
            span_id=format(span_ctx.span_id, "016x"),
            trace_id=format(span_ctx.trace_id, "032x"),
            parent_span_id=parent_span_id,
            is_root=(len(self._span_stack) == 1),
            depth=len(self._span_stack),
            start_time=start_time,
        )

        # Update context with new span
        self._update_otel_context()

        return span

    def end_current_span(self) -> None:
        """End the current span and set parent as current"""
        if not self._span_stack:
            self._logger.warning("Attempted to end span with empty stack")
            return

        span = self._span_stack.pop()
        span_ctx = span.get_span_context()
        span_name = getattr(span, "_name", "unknown")  # Try to get span name

        span.end()
        end_time = datetime.fromtimestamp(span.end_time / 1_000_000_000)

        self._logger.log_span_end(
            span_name=span_name,
            span_id=format(span_ctx.span_id, "016x"),
            is_root=(len(self._span_stack) == 0),
            depth=len(self._span_stack),
            end_time=end_time,
        )

        # Update context with parent span (or None if stack is empty)
        self._update_otel_context()

    @property
    def current_span(self) -> Optional[Span]:
        return self._span_stack[-1] if self._span_stack else None

    def get_otel_context(self) -> Optional[OtelContext]:
        """Export current span context for subprocess transport.

        Returns:
            OtelContext with trace_id and span_id as hex strings, or None if no current span.
        """
        if not self.current_span:
            return None

        span_ctx = self.current_span.get_span_context()
        return OtelContext(
            trace_id=format(span_ctx.trace_id, "032x"),
            span_id=format(span_ctx.span_id, "016x"),
        )

    def _update_otel_context(self) -> None:
        """Update the global Context with current OTEL span context."""
        otel_context = self.get_otel_context()
        ctx = get_context()
        new_ctx = ctx.with_otel_context(otel_context)
        set_context(new_ctx)
        self._logger.log_context_update(
            otel_context.trace_id if otel_context is not None else None,
            otel_context.span_id if otel_context is not None else None,
            operation="write",
        )

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        if self.current_span is None:
            raise (AttributeError("No current span"))
        write_attribute(self.current_span, key, value)
        span_ctx = self.current_span.get_span_context()
        self._logger.log_attribute_set(key, value, format(span_ctx.span_id, "016x"))

    def set_attributes(
        self, attributes: Optional[Dict[str, AttributeValue]] = None, **kwargs
    ) -> None:
        if attributes is None:
            attributes = {}
        attributes.update(kwargs)
        for k, v in attributes.items():
            self.set_attribute(k, v)

    def set_heritable_attribute(self, key: str, value: AttributeValue) -> None:
        self._heritable_attributes[key] = value
        if self.current_span:
            self.set_attribute(key, value)

    def set_heritable_attributes(
        self, attributes: Optional[Dict[str, AttributeValue]] = None, **kwargs
    ) -> None:
        if attributes:
            self._heritable_attributes.update(attributes)
        self._heritable_attributes.update(kwargs)
        if self.current_span:
            self.set_attributes(attributes, **kwargs)

    def record_exception(self, exc: Exception, set_error_status: bool = True) -> None:
        if self.current_span:
            self.current_span.record_exception(exc)
            if set_error_status:
                self.current_span.set_status(Status(StatusCode.ERROR))
        self._logger.log_exception(exc)

    def update_current_span_name(self, new_name: str) -> None:
        if self.current_span:
            old_name = getattr(self.current_span, "_name", "unknown")
            self.current_span.update_name(new_name)
            span_ctx = self.current_span.get_span_context()
            self._logger.log_span_rename(
                old_name, new_name, format(span_ctx.span_id, "016x")
            )


# OBSERVER
class OtelTracingObserver(Observer):
    """OpenTelemetry tracing observer for sessions.

    This observer creates a complete trace for each session, with spans for:
    - Session (root span)
    - Steps (child spans)
    - Actions and observations (nested child spans)

    Each session gets its own SessionSpanContext instance, ensuring complete
    isolation between concurrent sessions
    """

    def __init__(self):
        super().__init__()
        self._run_attributes: Dict[str, AttributeValue] = {}
        self._span_managers: Dict[str, SessionSpanManager] = {}
        self._session_step_counters: Dict[str, int] = {}

    def _get_span_manager(self, session_id: str) -> SessionSpanManager:
        return self._span_managers[session_id]

    def on_run_start(self, run_config) -> None:
        bench_entry = get_benchmark_entries().get(run_config.benchmark)
        agent_entry = get_agent_entries().get(run_config.agent)
        bench_name = (
            bench_entry.display_name
            if bench_entry is not None
            else run_config.benchmark
        )
        agent_name = (
            agent_entry.display_name if agent_entry is not None else run_config.agent
        )
        self._run_attributes = {
            "benchmark.name": bench_name,
            "benchmark.agent.name": agent_name,
            "run.id": get_run_id(),
        }

    def on_session_start(self, session, agent, observation) -> None:
        span_manager = SessionSpanManager(
            session.session_id, self.paths.session(session.session_id).root
        )
        self._span_managers[session.session_id] = span_manager
        self._session_step_counters[session.session_id] = 0

        # Start root session span
        bench_name = self._run_attributes.get("benchmark.name", "unknown-benchmark")
        span_manager.start_span(f"{bench_name} session")

        span_manager.set_heritable_attributes(self._run_attributes)

        # Set session-level attributes
        span_manager.set_heritable_attribute(
            "session.id",
            session.session_id,
        )
        span_manager.set_attribute("session.task_id", session.task_id)
        span_manager.set_attribute(
            "session.task",
            session.task,
        )
        span_manager.set_attribute("session.path", safe_repr(session.paths.root))
        for action in session.actions:
            span_manager.set_attribute(
                f"session.action.{action.name}.name", action.name
            )
            span_manager.set_attribute(
                f"session.action.{action.name}.description", action.description
            )
            span_manager.set_attribute(
                f"session.action.{action.name}.is_message", action.is_message
            )
            span_manager.set_attribute(
                f"session.action.{action.name}.is_finish", action.is_finish
            )
        for k, v in session.context.items():
            span_manager.set_attribute(f"context.{k}", v)
        span_manager.set_attribute("session.agent.id", agent.agent_id)
        span_manager.set_attribute(
            "session.agent.path", safe_repr(agent.paths.agent_dir)
        )

        # Start first step span for initial observation
        span_manager.start_span("step execute")
        span_manager.set_heritable_attribute(
            "step.index", self._session_step_counters[session.session_id]
        )

        # Record initial observation
        span_manager.start_span("observation get")
        self._record_observation(session.session_id, observation)
        span_manager.end_current_span()  # end observation span
        span_manager.end_current_span()  # end step span

        # Increment step counter and start next step span for first action
        self._session_step_counters[session.session_id] += 1
        self._start_next_step(session.session_id)

    def _start_next_step(self, session_id: str) -> None:
        """Start a new step span and action span"""
        span_manager = self._get_span_manager(session_id)
        step_index = self._session_step_counters[session_id]
        span_manager.start_span("step execute")  # step span
        span_manager.set_heritable_attribute("step.index", step_index)
        span_manager.start_span("agent react")  # nested agent span

    def _record_observation(self, session_id: str, observation) -> None:
        """Record observation details on the current span."""
        span_manager = self._get_span_manager(session_id)
        observation_list = (
            observation.to_observation_list() if observation is not None else []
        )
        observation_repr = safe_repr(observation_list[0]) if observation_list else ""

        span_manager.set_attribute("observation.repr", observation_repr)
        span_manager.set_attribute("observation", safe_repr(observation))
        span_manager.set_attributes(
            {
                f"observation.{i}": safe_repr(obs)
                for i, obs in enumerate(observation_list)
            }
        )

    def _record_action(self, session_id: str, action) -> None:
        """Record action details on the current span."""
        span_manager = self._get_span_manager(session_id)
        action_list = action.to_action_list() if action is not None else []
        action_repr = safe_repr(action_list[0]) if action_list else ""

        span_manager.set_attribute("action.repr", action_repr)
        span_manager.set_attribute("action", safe_repr(action))
        span_manager.set_attributes(
            {f"action.{i}": safe_repr(act) for i, act in enumerate(action_list)}
        )

    def on_react_success(self, session, action) -> None:
        span_manager = self._get_span_manager(session.session_id)
        self._record_action(session.session_id, action)
        span_manager.end_current_span()  # end agent span
        span_manager.start_span("observation get")

    def on_react_error(self, session, error) -> None:
        span_manager = self._get_span_manager(session.session_id)
        span_manager.record_exception(error)
        span_manager.end_current_span()  # end agent span
        span_manager.start_span("observation get")

    def on_step_success(self, session, observation) -> None:
        span_manager = self._get_span_manager(session.session_id)
        self._record_observation(session.session_id, observation)
        span_manager.end_current_span()  # end observation span
        span_manager.end_current_span()  # end step span

        self._session_step_counters[session.session_id] += 1
        self._start_next_step(session.session_id)

    def on_step_error(self, session, error) -> None:
        span_manager = self._get_span_manager(session.session_id)
        span_manager.record_exception(error)
        span_manager.end_current_span()  # end observation span
        span_manager.end_current_span()  # end step span

        self._session_step_counters[session.session_id] += 1
        self._start_next_step(session.session_id)

    def on_session_success(self, session, score, agent) -> None:
        span_manager = self._get_span_manager(session.session_id)

        # Close waiting action span
        span_manager.update_current_span_name("session close")
        span_manager.end_current_span()

        # Close final step span
        span_manager.update_current_span_name("session close")
        span_manager.end_current_span()

        # Add final session attributes
        span_manager.set_attribute("score.success", score.success)
        span_manager.set_attribute("score", score.score)
        span_manager.set_attribute("score.is_finished", score.is_finished)
        span_manager.set_attribute(
            "session.steps", self._session_step_counters[session.session_id]
        )
        span_manager.set_attribute("agent.agent_cost", agent.get_cost())
        span_manager.set_attribute("session.cost", session.get_cost())
        span_manager.set_attribute("session.task_id", session.task_id)

        # Close session span
        span_manager.end_current_span()

        # Flush traces to ensure they are exported
        flush_traces()

        # Clean up
        del self._span_managers[session.session_id]
        del self._session_step_counters[session.session_id]

    def on_session_error(self, session, error) -> None:
        span_manager = self._get_span_manager(session.session_id)

        span_manager.end_current_span()  # Close observation/action span
        span_manager.end_current_span()  # Close step span

        # Record error on session span
        span_manager.record_exception(error)
        span_manager.set_attribute("session.cost", session.get_cost())
        span_manager.set_attribute("session.task_id", session.task_id)

        span_manager.end_current_span()  # Close session span

        # Flush traces to ensure they are exported
        flush_traces()

        # Clean up
        del self._span_managers[session.session_id]
        del self._session_step_counters[session.session_id]
