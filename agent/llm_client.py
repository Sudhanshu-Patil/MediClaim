"""LLM clients: local Ollama, and Groq's free-tier hosted API.

Two interchangeable backends behind the same interface (chat/chat_stream/
is_available), selected by LLM_PROVIDER:
  * ollama (default) — local Ollama, any Ollama-compatible endpoint via
    LLM_BASE_URL. Used for local dev with the fine-tuned medclaim-llm.
  * groq — Groq's hosted API (console.groq.com), genuinely free tier (not a
    trial), no card required, OpenAI-compatible schema. Used for the cloud
    deployment (roadmap step 10) — running Ollama+a local GGUF in the cloud
    turned out to need more RAM than any no-card host offers (measured: the
    project's own NLI+embedding+reranker stack alone is 719MB with torch
    loaded; adding a multi-GB Ollama process on top was never going to fit
    inside a free tier). Trading the fine-tuned 3B for a much larger stock
    model via Groq is arguably not even a downgrade for a public demo — see
    RUNME §15.

Both implement Ollama's `format: json` / OpenAI's `response_format:
json_object` — the generator relies on this for the {"answer", "citations"}
schema. A circuit breaker (README §12) fails fast after repeated errors
instead of stacking timeouts on a struggling endpoint.
"""

from __future__ import annotations

import json as _json
import logging
import os
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class CircuitOpenError(RuntimeError):
    """LLM endpoint is failing; calls are being rejected without trying."""


class _CircuitBreaker:
    """Shared fail-fast logic — both clients hit the same failure modes
    (a struggling free-tier endpoint should stop being hammered, same as a
    struggling local one)."""

    def __init__(self, failure_threshold: int, cooldown_seconds: float) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._consecutive_failures = 0
        self._open_until = 0.0

    def check(self, label: str) -> None:
        if time.monotonic() < self._open_until:
            raise CircuitOpenError(
                f"LLM circuit open until {self._open_until - time.monotonic():.0f}s "
                f"from now ({label})"
            )

    def record(self, ok: bool) -> None:
        if ok:
            self._consecutive_failures = 0
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._open_until = time.monotonic() + self.cooldown_seconds
            self._consecutive_failures = 0
            logger.error("LLM circuit opened for %.0fs", self.cooldown_seconds)


class OllamaClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "medclaim-llm")
        self.timeout = timeout
        self._breaker = _CircuitBreaker(failure_threshold, cooldown_seconds)

    def chat(
        self,
        messages: list[dict],
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """One chat completion; returns the assistant message content.

        Recorded as a Langfuse generation (model + real Ollama token counts)
        when tracing is enabled; plain call otherwise.
        """
        from observability import tracing

        @tracing.traced("ollama-chat", as_type="generation")
        def _call() -> str:
            self._breaker.check(f"{self.base_url}, model={self.model}")
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                # Keep the model resident between short-lived CLI runs —
                # otherwise every cold call pays ~12 s of load time.
                "keep_alive": os.getenv("LLM_KEEP_ALIVE", "30m"),
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    # Brake for the 3B's repetition loops at temperature 0
                    # (observed live: same 3 sentences looped to token cap).
                    "repeat_penalty": float(os.getenv("LLM_REPEAT_PENALTY", "1.15")),
                },
            }
            if json_mode:
                payload["format"] = "json"
            try:
                response = httpx.post(
                    f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                self._breaker.record(ok=True)
                tracing.update_generation(
                    model=self.model,
                    usage={
                        "input": data.get("prompt_eval_count"),
                        "output": data.get("eval_count"),
                    },
                    metadata={"json_mode": json_mode, "base_url": self.base_url},
                )
                return data["message"]["content"]
            except (httpx.HTTPError, KeyError) as exc:
                self._breaker.record(ok=False)
                raise RuntimeError(
                    f"LLM call failed against {self.base_url}: {exc}"
                ) from exc

        return _call()

    def chat_stream(
        self,
        messages: list[dict],
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        """Streaming chat completion — yields content deltas as they generate.

        Same circuit breaker as chat(). The caller accumulates the full text;
        Ollama sends usage counts in the final chunk (ignored here — the
        traced non-stream path records usage).
        """
        self._breaker.check(f"{self.base_url}, model={self.model}")
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": os.getenv("LLM_KEEP_ALIVE", "30m"),
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "repeat_penalty": float(os.getenv("LLM_REPEAT_PENALTY", "1.15")),
            },
        }
        if json_mode:
            payload["format"] = "json"
        try:
            with httpx.stream("POST", f"{self.base_url}/api/chat",
                              json=payload, timeout=self.timeout) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = _json.loads(line)
                    delta = data.get("message", {}).get("content", "")
                    if delta:
                        yield delta
                    if data.get("done"):
                        break
            self._breaker.record(ok=True)
        except httpx.HTTPError as exc:
            self._breaker.record(ok=False)
            raise RuntimeError(
                f"LLM stream failed against {self.base_url}: {exc}"
            ) from exc

    def is_available(self) -> bool:
        try:
            return httpx.get(f"{self.base_url}/api/tags", timeout=5.0).status_code == 200
        except httpx.HTTPError:
            return False


class GroqClient:
    """Groq's hosted API (OpenAI-compatible schema), free tier, no card.

    Free-tier limits (console.groq.com/docs/rate-limits, verified at
    integration time): ~30 requests/min, model-dependent token/day caps —
    plenty for a low-traffic portfolio demo, tight for a load test. Model
    catalog rotates; GROQ_MODEL defaults to a currently-available Llama
    instruct model but is fully overridable via env without a code change.
    """

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set — required when LLM_PROVIDER=groq"
            )
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.timeout = timeout
        self._breaker = _CircuitBreaker(failure_threshold, cooldown_seconds)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"}

    def chat(
        self,
        messages: list[dict],
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        from observability import tracing

        @tracing.traced("groq-chat", as_type="generation")
        def _call() -> str:
            self._breaker.check(f"groq, model={self.model}")
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            try:
                response = httpx.post(
                    f"{self.BASE_URL}/chat/completions", json=payload,
                    headers=self._headers(), timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                self._breaker.record(ok=True)
                usage = data.get("usage", {})
                tracing.update_generation(
                    model=self.model,
                    usage={"input": usage.get("prompt_tokens"),
                          "output": usage.get("completion_tokens")},
                    metadata={"json_mode": json_mode, "provider": "groq"},
                )
                return data["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                self._breaker.record(ok=False)
                raise RuntimeError(f"Groq call failed: {exc}") from exc

        return _call()

    def chat_stream(
        self,
        messages: list[dict],
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        self._breaker.check(f"groq, model={self.model}")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            with httpx.stream(
                "POST", f"{self.BASE_URL}/chat/completions", json=payload,
                headers=self._headers(), timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    raw = line[len("data: "):]
                    if raw.strip() == "[DONE]":
                        break
                    delta = _json.loads(raw)["choices"][0].get("delta", {}).get("content")
                    if delta:
                        yield delta
            self._breaker.record(ok=True)
        except httpx.HTTPError as exc:
            self._breaker.record(ok=False)
            raise RuntimeError(f"Groq stream failed: {exc}") from exc

    def is_available(self) -> bool:
        try:
            return httpx.get(f"{self.BASE_URL}/models", headers=self._headers(),
                             timeout=5.0).status_code == 200
        except httpx.HTTPError:
            return False


def make_llm_client(**kwargs):
    """Factory: LLM_PROVIDER=ollama (default) or groq."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "groq":
        return GroqClient(**kwargs)
    return OllamaClient(**kwargs)
