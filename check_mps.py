import torch
import numpy as np
import time
from matrix_memory import MatrixVectorMemory

print("=" * 50)
print("  DEVICE AVAILABILITY CHECK")
print("=" * 50)
print(f"  PyTorch version    : {torch.__version__}")
print(f"  CUDA available     : {torch.cuda.is_available()}")
print(f"  MPS available      : {torch.backends.mps.is_available()}")
print(f"  MPS built          : {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    best_device = "mps"
elif torch.cuda.is_available():
    best_device = "cuda"
else:
    best_device = "cpu"

print(f"\n  Best available device: {best_device.upper()}")
print("=" * 50)

# Run head-to-head comparison if MPS is available
if best_device == "mps":
    print("\n[-] Running CPU vs MPS head-to-head comparison...")
    
    DIMENSION = 1536
    NUM_DOCS = 10000
    BATCH_SIZE = 5
    
    np.random.seed(42)
    chunks = [f"chunk_{i}" for i in range(NUM_DOCS)]
    embeddings = np.random.randn(NUM_DOCS, DIMENSION).tolist()
    queries = np.random.randn(BATCH_SIZE, DIMENSION).tolist()

    results = {}
    for device in ["cpu", "mps"]:
        mem = MatrixVectorMemory(embedding_dim=DIMENSION, device=device)
        mem.add_documents(chunks, embeddings)
        
        # Warm-up
        _ = mem.retrieve(queries, top_k=3)
        
        # Timed runs
        timings = []
        for _ in range(10):
            s = time.perf_counter()
            mem.retrieve(queries, top_k=3)
            timings.append((time.perf_counter() - s) * 1000)
        
        avg = sum(timings) / len(timings)
        results[device] = avg
        print(f"  {device.upper():4s} — Avg: {avg:.4f} ms | Per query: {avg/BATCH_SIZE:.4f} ms")

    if "mps" in results and "cpu" in results:
        speedup = results["cpu"] / results["mps"]
        print(f"\n  MPS speedup over CPU: {speedup:.2f}x")
        print(f"\n  Recommendation: Set device='mps' in MatrixVectorMemory")
else:
    print("\n  MPS not available on this machine.")
    print("  Your CPU baseline of ~3.56ms/query is your working number.")
    print("  This is sufficient for CRAG pipeline development.")