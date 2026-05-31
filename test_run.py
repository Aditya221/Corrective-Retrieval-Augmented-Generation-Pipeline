import numpy as np
import time
import torch
from matrix_memory import MatrixVectorMemory

def run_local_benchmark():
    DIMENSION = 1536
    NUM_DOCUMENTS = 10000
    BATCH_SIZE = 5

    print(f"[-] Initializing Matrix Vector Memory...")
    memory = MatrixVectorMemory(embedding_dim=DIMENSION)
    print(f"[+] Engine loaded on target device: {memory.device.upper()}")
    print(f"[+] PyTorch version: {torch.__version__}")

    print(f"\n[-] Generating {NUM_DOCUMENTS} synthetic document chunks and vectors...")
    mock_chunks = [f"This is the raw text context for document chunk number {i}." for i in range(NUM_DOCUMENTS)]
    np.random.seed(42)
    mock_embeddings = np.random.randn(NUM_DOCUMENTS, DIMENSION).tolist()

    start_time = time.perf_counter()
    memory.add_documents(mock_chunks, mock_embeddings)
    write_latency = (time.perf_counter() - start_time) * 1000
    print(f"[+] Matrix built. Write time: {write_latency:.2f} ms")
    print(f"[+] Matrix shape: {memory.doc_matrix.shape}")
    matrix_size_mb = (memory.doc_matrix.element_size() * memory.doc_matrix.nelement()) / (1024 ** 2)
    print(f"[+] Matrix memory footprint: {matrix_size_mb:.2f} MB")

    print(f"\n[-] Running retrieval benchmark ({BATCH_SIZE} concurrent queries)...")
    mock_queries = np.random.randn(BATCH_SIZE, DIMENSION).tolist()

    # Warm-up run (important for accurate CPU timing)
    _ = memory.retrieve(mock_queries, top_k=3)

    # Timed run — average over 10 iterations for stability
    timings = []
    for _ in range(10):
        start_time = time.perf_counter()
        results = memory.retrieve(mock_queries, top_k=3)
        timings.append((time.perf_counter() - start_time) * 1000)

    avg_latency = sum(timings) / len(timings)
    min_latency = min(timings)
    max_latency = max(timings)

    print("\n" + "="*55)
    print(f"  PERFORMANCE METRICS ({memory.device.upper()}) — 10k Documents")
    print("="*55)
    print(f"  Documents in matrix    : {NUM_DOCUMENTS:,}")
    print(f"  Embedding dimensions   : {DIMENSION}")
    print(f"  Batch query size       : {BATCH_SIZE}")
    print(f"  Matrix footprint       : {matrix_size_mb:.2f} MB")
    print(f"  Write latency          : {write_latency:.2f} ms")
    print(f"  Avg retrieval latency  : {avg_latency:.4f} ms")
    print(f"  Min retrieval latency  : {min_latency:.4f} ms")
    print(f"  Max retrieval latency  : {max_latency:.4f} ms")
    print(f"  Avg per query          : {(avg_latency / BATCH_SIZE):.4f} ms")
    print("="*55)

    print("\n[+] Sample output for Query #0:")
    for rank, match in enumerate(results[0]):
        print(f"  Rank {rank+1}: Score={match['score']} | {match['text'][:60]}...")

    # Scale sweep
    print("\n[-] Running scale sweep...")
    print(f"\n{'Docs':>8} | {'Avg Latency (ms)':>18} | {'Per Query (ms)':>15} | {'Memory (MB)':>12}")
    print("-" * 62)

    for n_docs in [1_000, 5_000, 10_000, 50_000, 100_000]:
        mem = MatrixVectorMemory(embedding_dim=768)  # 768-dim for sweep (faster)
        chunks = [f"chunk_{i}" for i in range(n_docs)]
        embeddings = np.random.randn(n_docs, 768).tolist()
        mem.add_documents(chunks, embeddings)

        q = np.random.randn(BATCH_SIZE, 768).tolist()
        _ = mem.retrieve(q, top_k=3)  # warm-up

        t = []
        for _ in range(5):
            s = time.perf_counter()
            mem.retrieve(q, top_k=3)
            t.append((time.perf_counter() - s) * 1000)

        avg = sum(t) / len(t)
        mb = (mem.doc_matrix.element_size() * mem.doc_matrix.nelement()) / (1024**2)
        print(f"{n_docs:>8,} | {avg:>18.4f} | {avg/BATCH_SIZE:>15.4f} | {mb:>12.2f}")

if __name__ == "__main__":
    run_local_benchmark()