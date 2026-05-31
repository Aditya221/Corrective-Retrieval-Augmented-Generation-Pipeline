import os
import time
import json
from typing import List
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
# Load environment variables from the project root .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path=env_path)



@dataclass
class RewriteResult:
    original_query: str
    rewritten_query: str
    reasoning: str
    strategy: str
    trigger_grade: str
    latency_ms: float
    backend: str

SYSTEM_PROMPT = """You are an expert Query Rewriter for a Retrieval-Augmented Generation (RAG) system.
The user submitted a query, but the search engine retrieved unhelpful documents.

Analyze the original query and the failed documents, then write a NEW, optimized query.

You MUST respond in valid JSON matching this schema:
{
    "rewritten_query": "The new, optimized search string",
    "strategy": "Pivot | Expand | Simplify", 
    "reasoning": "One short sentence explaining why you changed it"
}
"""

class QueryRewriter:
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("Missing GROQ_API_KEY environment variable. Please export it.")
            
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_api_key
        )
        self.model = model

    def _call_llm(self, original_query: str, failed_chunks: List[str]) -> tuple[str, str, str]:
        context = f"ORIGINAL QUERY: {original_query}\n\nFAILED CHUNKS:\n"
        for i, chunk in enumerate(failed_chunks):
            context += f"[{i+1}] {chunk}\n"
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context}
                ],
                response_format={"type": "json_object"},
                max_tokens=150,
                temperature=0.2 
            )
            
            raw_text = response.choices[0].message.content
            parsed = json.loads(raw_text)
            
            return (
                parsed.get("rewritten_query", original_query), 
                parsed.get("strategy", "Unknown"),
                parsed.get("reasoning", "Parsed successfully.")
            )
            
        except Exception as e:
            print(f"[!] Rewriter API Error: {e}")
            return original_query, "Error", f"Fallback triggered due to error: {e}"

    def rewrite(self, original_query: str, failed_chunks: List[str], trigger_grade: str) -> RewriteResult:
        start = time.perf_counter()
        rewritten_query, strategy, reasoning = self._call_llm(original_query, failed_chunks)
        latency_ms = (time.perf_counter() - start) * 1000
        
        return RewriteResult(
            original_query=original_query,
            rewritten_query=rewritten_query,
            reasoning=reasoning,
            strategy=strategy,
            trigger_grade=trigger_grade,
            latency_ms=round(latency_ms, 2),
            backend="groq_llama3"
        )