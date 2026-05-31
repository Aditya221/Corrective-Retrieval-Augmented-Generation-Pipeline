import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Import your custom infrastructure
from matrix_memory import MatrixVectorMemory
from core.evaluator.cross_encoder import CrossEncoderEvaluator
from core.rewriter import QueryRewriter

# Load environment variables from the project root .env file
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=env_path)

# Add this temporary debug line:
print(f"DEBUG - Groq Key Loaded: {str(os.environ.get('GROQ_API_KEY'))[:10]}...")


# --- Initialization ---
print("[-] Initializing CRAG Pipeline Components...")
app = FastAPI(title="Ultra-Low Latency CRAG API")

# 1. Load the Embedder
print("[-] Loading fast local embedder...")
embedder = SentenceTransformer("all-MiniLM-L6-v2") 
EMBEDDING_DIM = 384 

# 2. Load the custom tensor matrix
memory = MatrixVectorMemory(embedding_dim=EMBEDDING_DIM)

mock_chunks = [
    "Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function.",
    "Paris is the capital and most populous city of France.",
    "Vitamin D deficiency can lead to bone density loss, fatigue, and muscle weakness."
]
print("[-] Embedding mock knowledge base...")
mock_embeddings = embedder.encode(mock_chunks).tolist()
memory.add_documents(mock_chunks, mock_embeddings)

# 3. Load the Evaluator 
evaluator = CrossEncoderEvaluator()

# 4. Load the Query Rewriter 
rewriter = QueryRewriter(model="llama-3.3-70b-versatile")

# 5. Final Synthesis Client 
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

# --- API Data Models ---
class QueryRequest(BaseModel):
    query: str

class PipelineResponse(BaseModel):
    answer: str
    sources: List[str]
    pipeline_latency_ms: float
    rewritten: bool

# --- Helper Functions ---
def generate_final_answer(query: str, valid_chunks: List[str]) -> str:
    if not valid_chunks:
        return "I could not find verified information in my knowledge base to answer this query."
        
    context = "\n---\n".join(valid_chunks)
    prompt = f"Answer the user's query using ONLY the provided verified context.\n\nContext:\n{context}\n\nQuery: {query}"
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def execute_retrieval_and_eval(search_query: str):
    query_vector = embedder.encode(search_query).tolist()
    
    # Raw tensor search 
    raw_results = memory.retrieve([query_vector], top_k=2)[0]
    
    if not raw_results:
        return []
        
    # Extract just the text chunks for the refactored evaluate_batch method
    chunks = [r["text"] for r in raw_results]
    
    return evaluator.evaluate_batch(search_query, chunks)

# --- Main CRAG Endpoint ---
@app.post("/ask", response_model=PipelineResponse)
async def ask_crag_pipeline(request: QueryRequest):
    start_time = time.perf_counter()
    original_query = request.query
    was_rewritten = False
    
    # 1. INITIAL PASS
    graded_results = execute_retrieval_and_eval(original_query)
    
    # Access the EvalResult object attributes directly
    correct_chunks = [c.chunk_text for c in graded_results if c.grade == "CORRECT"]
    ambiguous_chunks = [c.chunk_text for c in graded_results if c.grade == "AMBIGUOUS"]
    incorrect_chunks = [c.chunk_text for c in graded_results if c.grade == "INCORRECT"]

    valid_context = correct_chunks
    
    # 2. THE CORRECTIVE LOOP
    if not correct_chunks:
        print(f"[-] Retrieval failed. Triggering Rewriter...")
        was_rewritten = True
        
        rewrite_result = rewriter.rewrite(
            original_query=original_query, 
            failed_chunks=incorrect_chunks + ambiguous_chunks,
            trigger_grade="INCORRECT"
        )
        
        print(f"[+] Rewrote query to: {rewrite_result.rewritten_query}")
        
        # 3. SECONDARY PASS
        new_graded_results = execute_retrieval_and_eval(rewrite_result.rewritten_query)
        
        valid_context = [c.chunk_text for c in new_graded_results if c.grade == "CORRECT"]
        
        if not valid_context:
            valid_context = [c.chunk_text for c in new_graded_results if c.grade == "AMBIGUOUS"]

    # 4. FINAL SYNTHESIS
    final_answer = generate_final_answer(original_query, valid_context)
    
    total_latency = (time.perf_counter() - start_time) * 1000
    
    return PipelineResponse(
        answer=final_answer,
        sources=valid_context,
        pipeline_latency_ms=round(total_latency, 2),
        rewritten=was_rewritten
    )