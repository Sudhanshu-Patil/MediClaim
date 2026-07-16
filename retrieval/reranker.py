"""Cross-encoder reranker — BAAI/bge-reranker-base via fastembed (ONNX, CPU).

Re-scores the RRF-fused candidate list against the query with full
query-document attention, which bi-encoder retrieval can't do. Runs after
fusion so it only ever sees a few dozen candidates (README §4).

Memory discipline: the ONNX model is ~1 GB resident. It loads lazily, can be
disabled outright (RERANKER_ENABLED=0), and a load failure degrades to the
fused order with a loud error instead of taking retrieval down — matching the
fail-fast-but-degrade posture of README §12. Every result marks whether it
was actually reranked, so degraded quality is visible, never silent.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from config import get_settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: Optional[str] = None, enabled: Optional[bool] = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        self.enabled = settings.reranker_enabled if enabled is None else enabled
        self._model = None
        self._load_failed = False

    def _get_model(self):
        if self._model is None and not self._load_failed:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                logger.info("Loading reranker model %s", self.model_name)
                self._model = TextCrossEncoder(model_name=self.model_name)
            except Exception:
                self._load_failed = True
                logger.exception(
                    "Reranker %s failed to load — retrieval will return RRF order. "
                    "Low RAM? Set RERANKER_ENABLED=0 to silence this, or free memory.",
                    self.model_name,
                )
        return self._model

    def rerank(
        self, query: str, documents: Sequence[str]
    ) -> Optional[list[tuple[int, float]]]:
        """Score documents against the query.

        Returns [(original_index, score), ...] best-first — or None when
        reranking is disabled/unavailable (caller keeps the fused order).
        """
        if not self.enabled or not documents:
            return None
        model = self._get_model()
        if model is None:
            return None
        scores = list(model.rerank(query, list(documents)))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in order]
