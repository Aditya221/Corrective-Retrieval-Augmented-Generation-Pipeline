import time
import json
import re
from typing import List
from .base import EvaluatorBase, EvalResult

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

SYSTEM_PROMPT = """You are a document relevance grader for a RAG pipeline.
Your job is to decide whether a retrieved document chunk adequately answers a user query.
Rules:
- Respond ONLY with valid JSON. No preamble, no explanation.
- The JSON must have exactly two keys: "confidence" and "reasoning"
- "confidence" must be a float between 0.0 and 1.0
- "reasoning" must be one sentence explaining your score

Scoring guide:
  0.9 - 1.0 : Chunk directly and completely answers the query
  0.6 - 0.9 : Chunk is clearly relevant but partially answers the query
  0.4 - 0.6 : Chunk is tangentially related; ambiguous usefulness
  0.1 - 0.4 : Chunk is mostly irrelevant to the query
  0.0 - 0.1 : Chunk is completely unrelated

Example output:
{"confidence": 0.85, "reasoning": "The chunk directly discusses the query topic with specific supporting details."}"""

USER_TEMPLATE = """Query: {query}

Retrieved chunk:
\"\"\"
{chunk}
\"\"\"

Grade this chunk."""


class LLMEvaluator(EvaluatorBase):
    def __init__(self, model: str = "claude-haiku-4-5-20251001",
                 correct_threshold: float = 0.8, ambiguous_threshold: float = 0.4):
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError("Run: pip install anthropic")
        super().__init__(correct_threshold, ambiguous_threshold)
        self.model = model
        self._client = anthropic.Anthropic()

    def _call_llm(self, query: str, chunk: str) -> tuple:
        response = self._client.messages.create(
            model=self.model, max_tokens=100,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": USER_TEMPLATE.format(query=query, chunk=chunk)}]
        )
        raw_text = re.sub(r"```json|```", "", response.content[0].text.strip()).strip()
        parsed = json.loads(raw_text)
        return float(parsed["confidence"]), parsed.get("reasoning", "")

    def evaluate_single(self, query: str, chunk: str) -> EvalResult:
        start = time.perf_counter()
        confidence, _ = self._call_llm(query, chunk)
        latency_ms = (time.perf_counter() - start) * 1000
        return EvalResult(grade=self.grade(confidence), confidence=round(confidence, 4),
                          latency_ms=round(latency_ms, 3), backend="llm", chunk_text=chunk)

    def evaluate_batch(self, query: str, chunks: List[str]) -> List[EvalResult]:
        return [self.evaluate_single(query, chunk) for chunk in chunks]
