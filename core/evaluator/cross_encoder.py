import time
import asyncio
from typing import List
from sentence_transformers import CrossEncoder
from .base import EvaluatorBase, EvalResult


# Model chosen for best latency/accuracy trade-off on MS MARCO.
# Alternatives ranked by speed:
#   cross-encoder/ms-marco-MiniLM-L-2-v2  → fastest,  lower accuracy
#   cross-encoder/ms-marco-MiniLM-L-6-v2  → balanced  ← DEFAULT
#   cross-encoder/ms-marco-MiniLM-L-12-v2 → slowest,  highest accuracy
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderEvaluator(EvaluatorBase):
    """
    Discriminative relevance grader using a Cross-Encoder model.

    Unlike bi-encoders (which embed query and document separately),
    a cross-encoder sees both strings simultaneously, making it
    significantly more accurate at judging query-document relevance.

    Raw model output is an unbounded logit score. We apply sigmoid
    normalization to map it into [0.0, 1.0] before threshold grading.

    Latency profile (Apple M-series CPU, MiniLM-L-6):
      Single chunk : ~15-25ms
      3 chunks     : ~20-30ms (batched, not sequential)

    This keeps total CRAG pipeline well under 50ms at <10k docs.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        correct_threshold: float = 0.8,
        ambiguous_threshold: float = 0.4,
    ):
        super().__init__(correct_threshold, ambiguous_threshold)
        self.model_name = model_name
        print(f"[-] Loading cross-encoder: {model_name}")
        self._model = CrossEncoder(model_name)
        print(f"[+] Cross-encoder loaded.")

    def _normalize_score(self, raw_score: float) -> float:
        """
        Sigmoid normalization: maps unbounded logit → (0, 1).
        Required because CrossEncoder outputs raw logits, not probabilities.
        """
        import math
        return 1.0 / (1.0 + math.exp(-raw_score))

    def evaluate_single(self, query: str, chunk: str) -> EvalResult:
        """Evaluate one query-chunk pair."""
        start = time.perf_counter()
        raw_score = self._model.predict([(query, chunk)])[0]
        latency_ms = (time.perf_counter() - start) * 1000

        confidence = self._normalize_score(raw_score)
        grade = self.grade(confidence)

        return EvalResult(
            grade=grade,
            confidence=round(confidence, 4),
            latency_ms=round(latency_ms, 3),
            backend="cross_encoder",
            chunk_text=chunk,
        )

    def evaluate_batch(self, query: str, chunks: List[str]) -> List[EvalResult]:
        """
        Evaluate all chunks in a single batched model call.
        This is faster than calling evaluate_single N times because
        the cross-encoder processes all pairs in one forward pass.
        """
        if not chunks:
            return []

        start = time.perf_counter()
        pairs = [(query, chunk) for chunk in chunks]
        raw_scores = self._model.predict(pairs)
        total_latency_ms = (time.perf_counter() - start) * 1000

        # Distribute latency evenly across results for logging
        per_chunk_latency = total_latency_ms / len(chunks)

        results = []
        for chunk, raw_score in zip(chunks, raw_scores):
            confidence = self._normalize_score(raw_score)
            results.append(EvalResult(
                grade=self.grade(confidence),
                confidence=round(confidence, 4),
                latency_ms=round(per_chunk_latency, 3),
                backend="cross_encoder",
                chunk_text=chunk,
            ))

        return results