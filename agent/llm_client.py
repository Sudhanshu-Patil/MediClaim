"""Minimal Ollama chat client (no LangChain wrapper needed).

Points at any Ollama-compatible endpoint via LLM_BASE_URL:
  * local Ollama:      http://localhost:11434  (default)
  * Colab GPU tunnel:  https://<something>.trycloudflare.com

``json_mode=True`` uses Ollama's ``format: json`` constrained decoding — the
generator relies on it for the {"answer", "citations"} schema.

A circuit breaker (README §12) fails fast after repeated errors instead of
stacking timeouts on an overloaded local model.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class CircuitOpenError(RuntimeError):
    """LLM endpoint is failing; calls are being rejected without trying."""


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
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._consecutive_failures = 0
        self._open_until = 0.0

    def _check_circuit(self) -> None:
        if time.monotonic() < self._open_until:
            raise CircuitOpenError(
                f"LLM circuit open until {self._open_until - time.monotonic():.0f}s from now "
                f"({self.base_url}, model={self.model})"
            )

    def _record(self, ok: bool) -> None:
        if ok:
            self._consecutive_failures = 0
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._open_until = time.monotonic() + self.cooldown_seconds
            self._consecutive_failures = 0
            logger.error("LLM circuit opened for %.0fs", self.cooldown_seconds)

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
            self._check_circuit()
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                # Keep the model resident between short-lived CLI runs —
                # otherwise every cold call pays ~12 s of load time.
                "keep_alive": os.getenv("LLM_KEEP_ALIVE", "30m"),
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
            if json_mode:
                payload["format"] = "json"
            try:
                response = httpx.post(
                    f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                self._record(ok=True)
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
                self._record(ok=False)
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
        self._check_circuit()
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": os.getenv("LLM_KEEP_ALIVE", "30m"),
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        try:
            with httpx.stream("POST", f"{self.base_url}/api/chat",
                              json=payload, timeout=self.timeout) as response:
                response.raise_for_status()
                import json as _json

                for line in response.iter_lines():
                    if not line:
                        continue
                    data = _json.loads(line)
                    delta = data.get("message", {}).get("content", "")
                    if delta:
                        yield delta
                    if data.get("done"):
                        break
            self._record(ok=True)
        except httpx.HTTPError as exc:
            self._record(ok=False)
            raise RuntimeError(
                f"LLM stream failed against {self.base_url}: {exc}"
            ) from exc

    def is_available(self) -> bool:
        try:
            return httpx.get(f"{self.base_url}/api/tags", timeout=5.0).status_code == 200
        except httpx.HTTPError:
            return False
