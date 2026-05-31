from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class EvalResult:
    """
    Structured verdict returned by any evaluator backend.
    The CRAG pipeline only ever sees this object — never the backend directly.
    """
    grade: str           # "CORRECT" | "AMBIGUOUS" | "INCORRECT"
    confidence: float    # 0.0 – 1.0 (normalized by each backend)
    latency_ms: float    # Measured wall-clock time for this evaluation
    backend: str         # "cross_encoder" | "llm"
    chunk_text: str      # The chunk that was graded (for logging/debugging)


class EvaluatorBase(ABC):
    """
    Strategy interface for all evaluator backends.

    Thresholds map a confidence float → CRAG grade:
      confidence >= correct_threshold   → CORRECT
      confidence >= ambiguous_threshold → AMBIGUOUS
      confidence <  ambiguous_threshold → INCORRECT

    These are config-driven so threshold calibration sweeps
    don't require touching evaluator logic.
    """

    def __init__(self, correct_threshold: float = 0.8, ambiguous_threshold: float = 0.4):
        if not (0.0 < ambiguous_threshold < correct_threshold <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy: 0 < ambiguous ({ambiguous_threshold}) "
                f"< correct ({correct_threshold}) <= 1.0"
            )
        self.correct_threshold = correct_threshold
        self.ambiguous_threshold = ambiguous_threshold

    def grade(self, confidence: float) -> str:
        """
        Single threshold-mapping method shared across all backends.
        Changing thresholds here changes behavior for every evaluator.
        """
        if confidence >= self.correct_threshold:
            return "CORRECT"
        elif confidence >= self.ambiguous_threshold:
            return "AMBIGUOUS"
        return "INCORRECT"

    @abstractmethod
    def evaluate_batch(self, query: str, chunks: List[str]) -> List[EvalResult]:
        """
        Evaluate a query against a list of retrieved chunks concurrently.
        Must return one EvalResult per chunk, in the same order.
        """
        pass

    @abstractmethod
    def evaluate_single(self, query: str, chunk: str) -> EvalResult:
        """Evaluate a single query-chunk pair."""
        pass