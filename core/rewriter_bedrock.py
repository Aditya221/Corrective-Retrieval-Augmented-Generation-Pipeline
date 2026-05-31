"""
Query Rewriter Module — AWS Bedrock Backend

Model: us.anthropic.claude-haiku-4-5-20251001-v1:0 (inference profile)
Region: us-east-1
"""

import os
import time
import json
import re
from typing import List
from dataclasses import dataclass
import boto3
from dotenv import load_dotenv

# Load environment variables from the project root .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
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


@dataclass
class RewriteResult:
    original_query: str
    rewritten_query: str
    strategy: str
    reasoning: str
    latency_ms: float
    trigger_grade: str


EXPAND_SYSTEM = """You are a search query optimizer for a RAG pipeline. The retrieved documents were AMBIGUOUS. Rewrite the query to be broader with alternative vocabulary."""

PIVOT_SYSTEM = """You are a search query optimizer for a RAG pipeline. The retrieved documents were INCORRECT. Rewrite the query using entirely different keywords and a different angle."""

USER_TEMPLATE = """Original query: {query}

Failed chunks:
{chunks}

Respond ONLY with JSON: {{"rewritten_query": "...", "reasoning": "..."}}"""


class QueryRewriter:
    def __init__(self, model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"):
        self.model_id = model_id
        self.client = get_bedrock_client()

    def _call_bedrock(self, query: str, failed_chunks: List[str], strategy: str) -> tuple:
        system = EXPAND_SYSTEM if strategy == "expand" else PIVOT_SYSTEM
        chunks_text = "\n".join(f"  - {c[:120]}" for c in failed_chunks)

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-06-01",
                "max_tokens": 150,
                "system": system,
                "messages": [{
                    "role": "user",
                    "content": USER_TEMPLATE.format(query=query, chunks=chunks_text)
                }]
            })
        )

        result = json.loads(response['body'].read())
        raw_text = result['content'][0]['text'].strip()
        raw_text = re.sub(r"```json|```", "", raw_text).strip()
        parsed = json.loads(raw_text)
        return parsed["rewritten_query"], parsed.get("reasoning", "")

    def rewrite(self, original_query: str, failed_chunks: List[str], trigger_grade: str) -> RewriteResult:
        strategy = "expand" if trigger_grade == "AMBIGUOUS" else "pivot"
        start = time.perf_counter()
        rewritten_query, reasoning = self._call_bedrock(original_query, failed_chunks, strategy)
        latency_ms = (time.perf_counter() - start) * 1000

        return RewriteResult(
            original_query=original_query,
            rewritten_query=rewritten_query,
            strategy=strategy,
            reasoning=reasoning,
            latency_ms=round(latency_ms, 3),
            trigger_grade=trigger_grade,
        )