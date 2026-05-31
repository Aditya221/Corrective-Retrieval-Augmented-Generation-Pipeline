import os
import time
import torch
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
from matrix_memory import MatrixVectorMemory

def run_beir_benchmark():
    print("="*60)
    print("  BEIR NFCorpus BENCHMARK: MATRIX VECTOR MEMORY")
    print("="*60)

    # 1. Download and Load NFCorpus
    print("[-] Downloading NFCorpus dataset...")
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    
    print("[-] Loading test split...")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    
    # Extract raw texts and query strings
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[doc_id].get("text", "") for doc_id in corpus_ids]
    
    query_ids = list(queries.keys())
    query_texts = [queries[q_id] for q_id in query_ids]

    print(f"[+] Loaded {len(corpus_texts)} documents and {len(query_texts)} queries.")

    # 2. Initialize Embedder and Matrix
    print("\n[-] Initializing Embedder (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    EMBEDDING_DIM = 384
    
    memory = MatrixVectorMemory(embedding_dim=EMBEDDING_DIM)
    print(f"[+] Matrix Engine loaded on: {memory.device.upper()}")

    # 3. Embed and Inject Corpus
    print("\n[-] Embedding corpus (this may take a minute)...")
    start_time = time.perf_counter()
    
    # Batch encode to prevent memory spikes
    corpus_embeddings = embedder.encode(corpus_texts, batch_size=128, show_progress_bar=True).tolist()
    memory.add_documents(corpus_texts, corpus_embeddings)
    
    write_latency = (time.perf_counter() - start_time)
    print(f"[+] Matrix built in {write_latency:.2f} seconds.")

    # 4. Execute Queries and Measure Latency
    print("\n[-] Embedding test queries...")
    query_embeddings = embedder.encode(query_texts, batch_size=128).tolist()

    print(f"[-] Executing {len(query_embeddings)} matrix retrievals...")
    start_time = time.perf_counter()
    
    # Retrieve top 10 for recall calculations
    batch_results = memory.retrieve(query_embeddings, top_k=10)
    
    retrieval_latency_ms = (time.perf_counter() - start_time) * 1000
    avg_latency = retrieval_latency_ms / len(query_embeddings)

    # 5. Calculate Recall@10
    print("\n[-] Calculating Recall@10...")
    hits = 0
    total_relevant_docs = 0

    for idx, q_id in enumerate(query_ids):
        # The ground truth relevant document IDs for this query
        true_relevant_ids = set(qrels[q_id].keys())
        total_relevant_docs += len(true_relevant_ids)
        
        # We need to map our matrix index back to the BEIR corpus_id
        # Our retrieve() method returns the text. We find the index of that text in corpus_texts
        # (In a production DB, your matrix would return IDs, not text strings directly)
        retrieved_texts = [match["text"] for match in batch_results[idx]]
        
        # Check if any retrieved text matches the text of a true relevant ID
        for true_id in true_relevant_ids:
            if true_id in corpus and corpus[true_id].get("text", "") in retrieved_texts:
                hits += 1
                # Count one hit per query for standard Recall@10 logic
                break 

    recall_at_10 = (hits / len(query_ids)) * 100

    # 6. Final Report
    print("\n" + "="*60)
    print("  EMPIRICAL BENCHMARK RESULTS")
    print("="*60)
    print(f"  Dataset            : BEIR NFCorpus (Test Split)")
    print(f"  Matrix Size        : {len(corpus_texts)} documents")
    print(f"  Dimensions         : {EMBEDDING_DIM}")
    print(f"  Total Queries      : {len(query_texts)}")
    print(f"  Total Matrix Time  : {retrieval_latency_ms:.2f} ms")
    print(f"  Avg Latency/Query  : {avg_latency:.4f} ms")
    print(f"  Recall@10          : {recall_at_10:.2f}%")
    print("="*60)

if __name__ == "__main__":
    run_beir_benchmark()