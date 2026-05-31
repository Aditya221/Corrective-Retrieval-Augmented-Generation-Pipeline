import torch
import numpy as np
from typing import List, Dict, Any, Optional

# Empirically determined on Apple M-series (MacBook, 1536-dim, batch=5)
# Below this threshold, MPS kernel dispatch overhead exceeds compute savings.
# Above it, Metal GPU parallelism wins. Determined via crossover_bench.py.
MPS_CROSSOVER_THRESHOLD = 50_000


def _select_device(n_docs: int, user_override: Optional[str] = None) -> str:
    """
    Adaptive device selection based on empirical crossover benchmarks.
    
    Decision logic:
      - User override always wins (for testing/explicit control)
      - CUDA: always preferred when available (no dispatch overhead issue)
      - MPS:  only above MPS_CROSSOVER_THRESHOLD (50k docs on Apple Silicon)
      - CPU:  default for small-to-medium corpora on Apple Silicon
    """
    if user_override is not None:
        return user_override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and n_docs >= MPS_CROSSOVER_THRESHOLD:
        return "mps"
    return "cpu"


class MatrixVectorMemory:
    """
    Hardware-adaptive tensor-based vector memory for RAG retrieval.

    Core operation: Similarity = normalize(Q) x normalize(D)^T
    When vectors are L2-normalised, dot product == cosine similarity.
    This transforms semantic search into a single batched matmul call.

    Device selection is adaptive:
      <50k docs  → CPU  (MPS dispatch overhead dominates on Apple Silicon)
      ≥50k docs  → MPS  (Metal GPU parallelism pays off)
      CUDA avail → CUDA (always preferred, no crossover issue)

    Crossover threshold validated empirically via crossover_bench.py.
    """

    def __init__(self, embedding_dim: int = 1536, device: Optional[str] = None):
        # Device resolved lazily on first add_documents call when n_docs is known.
        # For now, default to CPU as a safe initial state.
        self._user_device_override = device
        self.device = device if device is not None else "cpu"
        self.dim = embedding_dim
        self.doc_matrix = torch.empty((0, self.dim), dtype=torch.float32, device=self.device)
        self.chunk_map: Dict[int, str] = {}

    def _maybe_migrate_device(self, incoming_count: int) -> None:
        """
        Re-evaluates device selection after each add_documents call.
        Migrates the existing matrix to the new device if needed.
        This ensures the crossover threshold is respected dynamically
        as the corpus grows over multiple add_documents calls.
        """
        total_docs = self.doc_matrix.shape[0] + incoming_count
        optimal_device = _select_device(total_docs, self._user_device_override)

        if optimal_device != self.device:
            self.doc_matrix = self.doc_matrix.to(optimal_device)
            self.device = optimal_device

    def add_documents(self, chunks: List[str], embeddings: List[List[float]]) -> None:
        """
        Normalizes and stacks incoming embeddings into the document matrix.
        Accepts Python lists or numpy arrays for embeddings.
        Triggers device migration if corpus crosses the MPS threshold.
        """
        if not chunks or not embeddings:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk/embedding count mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings."
            )

        # Migrate device before adding (so new vectors land on the right device)
        self._maybe_migrate_device(len(embeddings))

        # Accept both list-of-lists and numpy arrays
        if isinstance(embeddings, np.ndarray):
            new_vectors = torch.from_numpy(embeddings.astype(np.float32)).to(self.device)
        else:
            new_vectors = torch.tensor(embeddings, dtype=torch.float32, device=self.device)

        if new_vectors.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.dim}, got {new_vectors.shape[1]}."
            )

        # L2 normalize: enables cosine similarity via dot product
        new_vectors = torch.nn.functional.normalize(new_vectors, p=2, dim=1)

        start_idx = self.doc_matrix.shape[0]
        self.doc_matrix = torch.cat([self.doc_matrix, new_vectors], dim=0)

        for i, chunk in enumerate(chunks):
            self.chunk_map[start_idx + i] = chunk

    def retrieve(self, query_embeddings: List[List[float]], top_k: int = 3) -> List[List[Dict[str, Any]]]:
        """
        Executes batched cosine similarity search via matrix multiplication.

        Shape flow:
          Query  (M x D) x Doc^T (D x N) = Similarity (M x N)
          torch.topk extracts top-K indices per row.

        Returns a list of result lists, one per query in the batch.
        """
        if self.doc_matrix.shape[0] == 0:
            return [[] for _ in range(len(query_embeddings))]

        # Guard: topk crashes if k > available rows
        actual_k = min(top_k, self.doc_matrix.shape[0])

        if isinstance(query_embeddings, np.ndarray):
            query_matrix = torch.from_numpy(query_embeddings.astype(np.float32)).to(self.device)
        else:
            query_matrix = torch.tensor(query_embeddings, dtype=torch.float32, device=self.device)

        query_matrix = torch.nn.functional.normalize(query_matrix, p=2, dim=1)

        # Core operation: single batched matmul
        similarity_matrix = torch.matmul(query_matrix, self.doc_matrix.T)

        top_scores, top_indices = torch.topk(similarity_matrix, k=actual_k, dim=1)

        batch_results = []
        for q_idx in range(query_matrix.shape[0]):
            query_results = [
                {
                    "text": self.chunk_map[doc_idx.item()],
                    "score": round(score.item(), 4),
                    "doc_id": doc_idx.item(),
                }
                for score, doc_idx in zip(top_scores[q_idx], top_indices[q_idx])
            ]
            batch_results.append(query_results)

        return batch_results

    @property
    def stats(self) -> Dict[str, Any]:
        """Quick diagnostics — useful for logging and README benchmarks."""
        matrix_mb = (self.doc_matrix.element_size() * self.doc_matrix.nelement()) / (1024 ** 2)
        return {
            "n_documents": self.doc_matrix.shape[0],
            "embedding_dim": self.dim,
            "device": self.device,
            "matrix_mb": round(matrix_mb, 2),
            "mps_threshold": MPS_CROSSOVER_THRESHOLD,
        }