"""
Query Rewriter Benchmark — AWS Bedrock Backend

Tests query rewrite strategies (expand/pivot) using Bedrock.

Run:
    python bench_rewriter_bedrock.py
"""

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "./core")

from core.rewriter_bedrock import QueryRewriter

REWRITE_TESTS = [
    {
        "original": "How does gradient descent optimize neural networks?",
        "failed_chunks": [
            "Neural networks are inspired by the human brain and consist of layers of interconnected nodes."
        ],
        "trigger_grade": "INCORRECT",
        "expect_strategy": "pivot",
        "description": "Pivot: failed chunk was topically adjacent but useless",
    },
    {
        "original": "What are the symptoms of vitamin D deficiency?",
        "failed_chunks": [
            "Vitamins are essential nutrients required in small amounts. They are divided into fat-soluble and water-soluble categories."
        ],
        "trigger_grade": "INCORRECT",
        "expect_strategy": "pivot",
        "description": "Pivot: generic vitamin info didn't answer deficiency symptoms",
    },
    {
        "original": "What is the capital of France?",
        "failed_chunks": [
            "France is a country in Western Europe known for its cuisine, wine, and art. It borders Germany, Spain, and Italy."
        ],
        "trigger_grade": "AMBIGUOUS",
        "expect_strategy": "expand",
        "description": "Expand: France context retrieved but capital not mentioned",
    },
]


def run_rewriter_benchmark():
    print("\n" + "="*60)
    print("  QUERY REWRITER BENCHMARK — AWS BEDROCK")
    print("="*60)

    rewriter = QueryRewriter()
    total_latency = 0

    for i, test in enumerate(REWRITE_TESTS):
        print(f"\n  Test {i+1}: {test['description']}")
        print(f"  {'─'*54}")

        result = rewriter.rewrite(
            original_query=test["original"],
            failed_chunks=test["failed_chunks"],
            trigger_grade=test["trigger_grade"],
        )

        strategy_match = "+" if result.strategy == test["expect_strategy"] else "x"
        is_different = result.rewritten_query.lower().strip() != test["original"].lower().strip()
        different_mark = "+" if is_different else "x"

        total_latency += result.latency_ms

        print(f"  Original  : {result.original_query}")
        print(f"  Rewritten : {result.rewritten_query}")
        print(f"  Strategy  : [{strategy_match}] {result.strategy} (expected: {test['expect_strategy']})")
        print(f"  Different : [{different_mark}] {'Yes' if is_different else 'No'}")
        print(f"  Reasoning : {result.reasoning}")
        print(f"  Latency   : {result.latency_ms:.1f}ms")

    avg_latency = total_latency / len(REWRITE_TESTS)
    print(f"\n{'='*60}")
    print(f"  SUMMARY (AWS BEDROCK)")
    print(f"{'='*60}")
    print(f"  Tests run       : {len(REWRITE_TESTS)}")
    print(f"  Avg latency     : {avg_latency:.1f}ms per rewrite")
    print(f"  Total LLM time  : {total_latency:.1f}ms")
    print(f"{'='*60}")
    print(f"\n  Pipeline budget with rewriter (INCORRECT path):")
    print(f"    Matrix retrieval  :    ~0.5ms")
    print(f"    Cross-encoder eval:   ~20.0ms")
    print(f"    Query rewrite     :  ~{avg_latency:.0f}ms  (Bedrock LLM)")
    print(f"    Re-retrieval      :    ~0.5ms")
    print(f"    Re-evaluation     :   ~20.0ms")
    print(f"    {'─'*30}")
    print(f"    Total (no answer) :  ~{0.5+20+avg_latency+0.5+20:.0f}ms")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_rewriter_benchmark()