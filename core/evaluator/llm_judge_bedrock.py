"""
LLM Judge Evaluator — AWS Bedrock Backend

Same interface as llm_judge.py but uses boto3.client('bedrock-runtime')
instead of anthropic.Anthropic().

Model: us.anthropic.claude-haiku-4-5-20251001-v1:0 (inference profile)
Region: us-east-1
"""

import os
import time
import json
import re
from typing import List
from .base import EvaluatorBase, EvalResult
import boto3
from dotenv import load_dotenv

# Load environment variables from the project root .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
load_dotenv(dotenv_path=env_path)


_BEDROCK_CLIENT = None

def get_bedrock_client():
    global _BEDROCK_CLIENT
    if _BEDROCK_CLIENT is None:
        aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
        if aws_secret and aws_secret.startswith("ABSK"):
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = aws_secret
            os.environ.pop("AWS_ACCESS_KEY_ID", None)
            os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
        _BEDROCK_CLIENT = boto3.client('bedrock-runtime', region_name='us-east-1')
    return _BEDROCK_CLIENT


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
    """
    Generative LLM-as-a-Judge evaluator using AWS Bedrock + Claude Haiku.
    """

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        correct_threshold: float = 0.8,
        ambiguous_threshold: float = 0.4,
    ):
        super().__init__(correct_threshold, ambiguous_threshold)
        self.model_id = model_id
        self.client = get_bedrock_client()

    def _call_bedrock(self, query: str, chunk: str) -> tuple:
        """Single Bedrock call. Returns (confidence, reasoning)."""
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-06-01",
                "max_tokens": 100,
                "system": SYSTEM_PROMPT,
                "messages": [{
                    "role": "user",
                    "content": USER_TEMPLATE.format(query=query, chunk=chunk)
                }]
            })
        )

        result = json.loads(response['body'].read())
        raw_text = result['content'][0]['text'].strip()
        raw_text = re.sub(r"```json|```", "", raw_text).strip()
        parsed = json.loads(raw_text)
        confidence = float(parsed["confidence"])
        reasoning = parsed.get("reasoning", "")
        return confidence, reasoning

    def evaluate_single(self, query: str, chunk: str) -> EvalResult:
        start = time.perf_counter()
        confidence, _ = self._call_bedrock(query, chunk)
        latency_ms = (time.perf_counter() - start) * 1000

        return EvalResult(
            grade=self.grade(confidence),
            confidence=round(confidence, 4),
            latency_ms=round(latency_ms, 3),
            backend="llm",
            chunk_text=chunk,
        )

    def evaluate_batch(self, query: str, chunks: List[str]) -> List[EvalResult]:
        """Sequential evaluation (for async version, see app.py)."""
        return [self.evaluate_single(query, chunk) for chunk in chunks]