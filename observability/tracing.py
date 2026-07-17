"""Langfuse tracing layer (roadmap step 6, README §11) — graceful no-op.

Wraps the Langfuse v2 SDK's ``@observe`` decorators behind functions that
degrade to identity/no-ops whenever tracing is off, so agent code never
imports langfuse directly and never breaks without keys.

Enabled iff LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY are set (and the SDK is
importable). Host defaults to the self-hosted instance at localhost:3000.

Server compatibility: the self-hosted Langfuse in docker-compose is v2 →
requires the v2 SDK (``langfuse>=2.53,<3``). The v3 SDK speaks OTel to v3
servers only.

LangSmith (dev-time tracing, README §11) needs no code at all: LangGraph
traces itself when LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY are set.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

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
            os.environ.setdefault("LANGFUSE_HOST", "http://localhost:3000")
            try:
                from langfuse.decorators import observe  # noqa: F401

                _ENABLED = True
                logger.info("Langfuse tracing enabled (%s)", os.environ["LANGFUSE_HOST"])
            except ImportError:
                _ENABLED = False
                logger.warning("LANGFUSE keys set but SDK missing; tracing disabled")
    return _ENABLED


def traced(name: str, as_type: Optional[str] = None) -> Callable:
    """Decorator factory: Langfuse span/generation, or identity when disabled."""

    def decorator(fn: Callable) -> Callable:
        if not enabled():
            return fn
        from langfuse.decorators import observe

        if as_type == "generation":
            return observe(name=name, as_type="generation")(fn)
        return observe(name=name)(fn)

    return decorator


def update_generation(model: str, usage: Optional[dict] = None,
                      metadata: Optional[dict] = None) -> None:
    """Attach model + token usage to the current generation observation."""
    if not enabled():
        return
    try:
        from langfuse.decorators import langfuse_context

        kwargs: dict[str, Any] = {"model": model}
        if usage and any(v for v in usage.values()):
            kwargs["usage"] = usage
        if metadata:
            kwargs["metadata"] = metadata
        langfuse_context.update_current_observation(**kwargs)
    except Exception:  # tracing must never take the pipeline down
        logger.debug("update_generation failed", exc_info=True)


def update_trace(name: Optional[str] = None, metadata: Optional[dict] = None,
                 tags: Optional[list[str]] = None) -> None:
    if not enabled():
        return
    try:
        from langfuse.decorators import langfuse_context

        kwargs: dict[str, Any] = {}
        if name:
            kwargs["name"] = name
        if metadata:
            kwargs["metadata"] = metadata
        if tags:
            kwargs["tags"] = tags
        langfuse_context.update_current_trace(**kwargs)
    except Exception:
        logger.debug("update_trace failed", exc_info=True)


def score_trace(name: str, value: float, comment: Optional[str] = None) -> None:
    """Attach a numeric score (e.g. the judge's) to the current trace."""
    if not enabled():
        return
    try:
        from langfuse.decorators import langfuse_context

        langfuse_context.score_current_trace(name=name, value=value, comment=comment)
    except Exception:
        logger.debug("score_trace failed", exc_info=True)


def flush() -> None:
    """Drain the batch queue — REQUIRED before short-lived CLIs exit."""
    if not enabled():
        return
    try:
        from langfuse.decorators import langfuse_context

        langfuse_context.flush()
    except Exception:
        logger.debug("flush failed", exc_info=True)
