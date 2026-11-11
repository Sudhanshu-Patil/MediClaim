"""Langfuse tracing layer (roadmap step 6, README §11) — graceful no-op.

Wraps the Langfuse v4 SDK (``langfuse>=4.7,<5``, OpenTelemetry-based) behind
functions that degrade to identity/no-ops whenever tracing is off, so agent
code never imports langfuse directly and never breaks without keys.

Enabled iff LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY are set (and the SDK is
importable). Target host comes from LANGFUSE_BASE_URL (LANGFUSE_HOST is still
honored); this project uses Langfuse Cloud (US: https://us.cloud.langfuse.com).

v4 model: trace-level attributes (session, tags, trace name) no longer get set
after the fact — they are *propagated* onto every observation created inside
``root_span``, so they must be established before the graph runs.

LangSmith (dev-time tracing, README §11) needs no code at all: LangGraph
traces itself when LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY are set.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger(__name__)

_ENABLED: Optional[bool] = None  # resolved lazily, once


def enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        has_keys = bool(
            os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
        )
        if not has_keys:
            _ENABLED = False
        else:
            try:
                from langfuse import get_client

                get_client()
                _ENABLED = True
                logger.info("Langfuse tracing enabled (%s)",
                            os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
                            or "SDK default host")
            except Exception:  # ImportError, or a broken OpenTelemetry install
                _ENABLED = False
                logger.warning("LANGFUSE keys set but SDK unusable; tracing disabled",
                               exc_info=True)
    return _ENABLED


def traced(name: str, as_type: Optional[str] = None) -> Callable:
    """Decorator factory: Langfuse span/generation, or identity when disabled."""

    def decorator(fn: Callable) -> Callable:
        if not enabled():
            return fn
        from langfuse import observe

        if as_type == "generation":
            return observe(name=name, as_type="generation")(fn)
        return observe(name=name)(fn)

    return decorator


@contextlib.contextmanager
def root_span(name: str, session_id: Optional[str] = None,
              tags: Optional[list[str]] = None, input: Any = None) -> Iterator[None]:
    """One root observation per request; session/tags/trace name propagate to
    every child (node spans, LLM generations) created inside the block."""
    if not enabled():
        yield
        return
    from langfuse import get_client, propagate_attributes
    from opentelemetry import context as otel_context

    # Detach from any ambient span (FastAPI's built-in request telemetry): v4
    # doesn't export those, so our root would point at a parent Langfuse never
    # receives. An empty context makes this a true trace root.
    token = otel_context.attach(otel_context.Context())
    try:
        with propagate_attributes(trace_name=name, tags=tags,
                                  session_id=str(session_id)[:200] if session_id else None):
            with get_client().start_as_current_observation(as_type="span", name=name,
                                                           input=input):
                yield
    finally:
        otel_context.detach(token)


def update_root(output: Any = None, metadata: Optional[dict] = None) -> None:
    """Set output/metadata on the current (root) span — call inside root_span
    after the graph returns, when route/status are known."""
    if not enabled():
        return
    try:
        from langfuse import get_client

        get_client().update_current_span(output=output, metadata=metadata)
    except Exception:  # tracing must never take the pipeline down
        logger.debug("update_root failed", exc_info=True)


def update_generation(model: str, usage: Optional[dict] = None,
                      metadata: Optional[dict] = None) -> None:
    """Attach model + token usage to the current generation observation."""
    if not enabled():
        return
    try:
        from langfuse import get_client

        kwargs: dict[str, Any] = {"model": model}
        usage_details = {k: int(v) for k, v in (usage or {}).items() if v}
        if usage_details:
            kwargs["usage_details"] = usage_details
        if metadata:
            kwargs["metadata"] = metadata
        get_client().update_current_generation(**kwargs)
    except Exception:
        logger.debug("update_generation failed", exc_info=True)


def score_trace(name: str, value: float, comment: Optional[str] = None) -> None:
    """Attach a numeric score (e.g. the judge's) to the current trace."""
    if not enabled():
        return
    try:
        from langfuse import get_client

        get_client().score_current_trace(name=name, value=value, comment=comment)
    except Exception:
        logger.debug("score_trace failed", exc_info=True)


def flush() -> None:
    """Drain the export queue — REQUIRED before short-lived CLIs exit."""
    if not enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        logger.debug("flush failed", exc_info=True)
