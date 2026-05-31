import torch
import numpy as np
import time
from matrix_memory import MatrixVectorMemory

DIMENSION = 1536
BATCH_SIZE = 5
np.random.seed(42)

doc_counts = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000]

print(f"{'Docs':>10} | {'CPU (ms)':>10} | {'MPS (ms)':>10} | {'Winner':>8} | {'MPS/CPU':>8}")
print("-" * 58)

for n_docs in doc_counts:
    chunks = [f"chunk_{i}" for i in range(n_docs)]
    embeddings = np.random.randn(n_docs, DIMENSION).tolist()
    queries = np.random.randn(BATCH_SIZE, DIMENSION).tolist()

    timings = {}
    for device in ["cpu", "mps"]:
        mem = MatrixVectorMemory(embedding_dim=DIMENSION, device=device)
        mem.add_documents(chunks, embeddings)

        _ = mem.retrieve(queries, top_k=3)  # warm-up

        t = []
        for _ in range(5):
            s = time.perf_counter()
            mem.retrieve(queries, top_k=3)
            t.append((time.perf_counter() - s) * 1000)

        timings[device] = sum(t) / len(t)

        # Free memory explicitly
        del mem

    winner = "MPS ✓" if timings["mps"] < timings["cpu"] else "CPU ✓"
    ratio = timings["mps"] / timings["cpu"]
    print(f"{n_docs:>10,} | {timings['cpu']:>10.3f} | {timings['mps']:>10.3f} | {winner:>8} | {ratio:>8.2f}x")

print("\nConclusion: Use CPU below crossover point, MPS above it.")
print("Update MatrixVectorMemory to auto-select based on matrix size.")