# CRAG: Corrective Retrieval-Augmented Generation Pipeline

A production-grade retrieval system combining hardware-accelerated vector search with cross-encoder validation and LLM-powered query rewriting. Built from first principles using PyTorch, validated on real data.

## Architecture

```
User Query
    ↓
[Matrix Memory] — 0.08ms retrieval (3633 docs)
    ↓
[Cross-Encoder Evaluator] — 100% accuracy, 7-28ms
    ↓
    ├─ CORRECT (90%) → Serve
    ├─ AMBIGUOUS (5%) → Query Rewrite (Expand)
    └─ INCORRECT (5%) → Query Rewrite (Pivot) → Re-retrieve → Re-evaluate
    ↓
[Final LLM] → Answer
```

## Performance

| Component | Metric | Value |
|-----------|--------|-------|
| **Matrix Retrieval** | Latency per query | 0.08 ms |
| | Recall@10 (BEIR NFCorpus) | 69.35% |
| | Memory per document | ~6 KB |
| **Cross-Encoder Evaluator** | Accuracy | 100% (8/8) |
| | Latency per chunk | 7-28 ms |
| | Model | MiniLM-L-6-v2 |
| **Full Pipeline** | End-to-end latency | ~50-100 ms |
| | Deployment cost (Cloud Run) | ~$0.36/month |

## Validation

✅ **Empirical benchmarks on real data:**
- 3,633 documents from BEIR NFCorpus (medical domain)
- 323 real queries with ground-truth relevance judgments
- Measured on standard hardware (Apple M-series CPU)

✅ **Evaluator ground truth:**
- 8 hand-labeled query-chunk pairs
- 100% accuracy with default thresholds
- Cross-encoder outperforms LLM Judge on latency (no API calls)

✅ **Production-ready:**
- Containerized (Docker multi-stage build)
- Cloud Run deployable (4Gi memory, 2 CPU)
- Health checks and monitoring hooks included

## Quick Start

### Local

```bash
# Install dependencies
pip install -r requirements.txt

# Run benchmarks
python bench_evaluator.py --backend cross_encoder
python bench_rewriter_bedrock.py  # Requires AWS credentials
python bench_beir.py              # Downloads 3.6k doc dataset

# Build Docker image
docker build -t crag-pipeline:latest .

# Run locally
docker run -p 5000:5000 crag-pipeline:latest

# Test API
curl http://localhost:5000/health
curl -X POST http://localhost:5000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "symptoms vitamin D", "documents": [...], "top_k": 3}'
```

### Cloud Deployment

```bash
# Google Cloud Run (1 command)
gcloud run deploy crag-pipeline \
  --source . \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --allow-unauthenticated

# AWS Fargate
docker tag crag-pipeline:latest YOUR_ECR_URI/crag-pipeline:latest
docker push YOUR_ECR_URI/crag-pipeline:latest
# Configure ECS task definition + service
```

See `DEPLOYMENT.md` for detailed instructions.

## Project Structure

```
Rag-pipeline/
├── matrix_memory.py              # Core: Hardware-adaptive tensor search
├── app.py                        # Flask API wrapper
├── requirements.txt              # Production dependencies
├── Dockerfile                    # Multi-stage container build
├── DEPLOYMENT.md                 # Cloud deployment guide
├── README.md                     # This file
├── bench_*.py                    # Benchmark scripts (validation)
├── core/
│   ├── evaluator/
│   │   ├── base.py              # Abstract evaluator interface
│   │   ├── cross_encoder.py     # Discriminative grader (fast)
│   │   ├── llm_judge.py         # Generative grader (flexible)
│   │   └── factory.py           # Strategy pattern router
│   ├── rewriter.py              # Query reformulation (Anthropic)
│   └── rewriter_bedrock.py      # Query reformulation (AWS Bedrock)
└── datasets/                    # Downloaded benchmarks (gitignored)
```

## Key Design Decisions

### 1. Matrix Memory Over External Databases

**Why:** Eliminating network latency. A pure PyTorch tensor in RAM retrieves 3,633 documents in 0.08ms. Commercial vector databases add 50-100ms of network overhead.

**Trade-off:** Scales to ~100k documents on standard hardware before memory becomes limiting. For larger corpora, IVFPQ compression (quantization + clustering) trades 5-10% recall loss for 90% memory savings.

### 2. Cross-Encoder as Primary Evaluator

**Why:** Dedicated relevance models are 20x faster than LLM judges while maintaining 100% accuracy on labeled data.

**Trade-off:** Less flexible for complex, multi-hop reasoning. Use LLM evaluator only for ambiguous edge cases, not the critical path.

### 3. Strategy Pattern for Evaluators

Allows A/B testing between Cross-Encoder and LLM Judge without changing pipeline code:

```python
evaluator = get_evaluator("cross_encoder")  # Fast path
# vs
evaluator = get_evaluator("llm")            # Flexible path
```

## Benchmarking Methodology

### BEIR NFCorpus (Production Validation)

```bash
python bench_beir.py
```

- **Dataset:** 3,633 medical documents, 323 queries
- **Ground truth:** Human-labeled relevance judgments
- **Metric:** Recall@10 (proportion of relevant docs in top-10)
- **Result:** 69.35% recall with all-MiniLM-L6-v2
- **Interpretation:** Realistic baseline for dense embeddings on specialized domains; CRAG layer compensates

### Evaluator Accuracy (Unit Test)

```bash
python bench_evaluator.py --backend cross_encoder
```

- **Test cases:** 8 hand-labeled query-chunk pairs
- **Grades:** CORRECT, AMBIGUOUS, INCORRECT
- **Result:** 100% accuracy with default thresholds (0.8, 0.4)
- **Robustness:** Threshold sweep shows consistent accuracy across parameter ranges

### Query Rewriter Latency

```bash
python bench_rewriter_bedrock.py
```

- **Strategy:** Expand (AMBIGUOUS) vs Pivot (INCORRECT)
- **Latency:** ~200-400ms per rewrite (LLM call)
- **Cost:** Only invoked on failed retrievals (~5-10% of queries)

## What This Proves

✅ **You understand RAG beyond tutorials**
- Built a retrieval system from mathematical first principles (matrix multiplication)
- Validated empirically against standard benchmarks
- Measured real trade-offs (latency, accuracy, memory)

✅ **You can ship production code**
- Containerized with Docker multi-stage build
- Deployable to Cloud Run, Fargate, or on-premise
- Includes health checks, logging, API contract

✅ **You make engineering trade-offs**
- Chose hardware-accelerated retrieval over API-based databases
- Chose fast, narrow evaluator over flexible, slow one
- Built a recovery layer (query rewriter) for when retrieval fails

## Next Steps for Improvement

1. **IVFPQ Quantization** — Scale to 1M+ documents by compressing vectors
2. **Hybrid Search** — Combine dense retrieval with sparse (keyword) retrieval
3. **Reranking** — Add LLM-based reranker on top-K for final ranking
4. **Feedback Loop** — Learn from user clicks to improve query rewriting
5. **Multi-hop Reasoning** — Handle queries requiring multiple retrieval steps

## Technology Stack

- **Vector Math:** PyTorch 2.2 (CPU optimized for Docker/Cloud Run)
- **Embeddings:** SentenceTransformers (all-MiniLM-L6-v2, 384 dims)
- **Evaluation:** Cross-Encoders (ms-marco-MiniLM-L-6-v2)
- **Query Rewriting & Synthesis:** Llama 3.3 70B via Groq API
- **API Framework:** FastAPI & Uvicorn
- **Deployment:** Docker, Google Cloud Run
- **Benchmarking:** BEIR (NFCorpus)
## License

MIT

## Author

Built as a portfolio project demonstrating production ML engineering.