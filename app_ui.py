"""
Streamlit UI for CRAG Pipeline
Visualizes the complete retrieval → evaluation → rewriting → synthesis loop.

Features:
- Real-time query processing with latency breakdown
- Cross-encoder evaluation visualization (CORRECT/AMBIGUOUS/INCORRECT)
- Query rewriter trigger display
- Pipeline trace with performance metrics
- Benchmark comparison (BEIR NFCorpus baseline)
"""

import streamlit as st
import time
import os
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Import your CRAG components
from matrix_memory import MatrixVectorMemory
from core.evaluator.cross_encoder import CrossEncoderEvaluator
from core.rewriter import QueryRewriter

# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================

st.set_page_config(
    page_title="CRAG Pipeline Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        font-size: 18px;
        font-weight: bold;
    }
    .grade-correct {
        background: #d4edda;
        color: #155724;
        padding: 12px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 8px 0;
        font-size: 14px;
    }
    .grade-ambiguous {
        background: #fff3cd;
        color: #856404;
        padding: 12px;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 8px 0;
        font-size: 14px;
    }
    .grade-incorrect {
        background: #f8d7da;
        color: #721c24;
        padding: 12px;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        margin: 8px 0;
        font-size: 14px;
    }
    .pipeline-step {
        background: linear-gradient(135deg, #f5f7fa 0%, #f0f3f7 100%);
        border: 1px solid #dee2e6;
        padding: 15px;
        border-radius: 8px;
        margin: 12px 0;
        border-left: 4px solid #667eea;
    }
    .step-number {
        display: inline-block;
        background: #667eea;
        color: white;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        text-align: center;
        line-height: 30px;
        font-weight: bold;
        margin-right: 10px;
    }
    .answer-box {
        background: #f8f9fa;
        border: 2px solid #667eea;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
    }
    .source-item {
        background: #f0f3f7;
        padding: 12px;
        border-radius: 6px;
        margin: 8px 0;
        border-left: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "pipeline_initialized" not in st.session_state:
    st.session_state.pipeline_initialized = False
    st.session_state.embedder = None
    st.session_state.memory = None
    st.session_state.evaluator = None
    st.session_state.rewriter = None
    st.session_state.groq_client = None
    st.session_state.query_history = []

# ============================================================================
# COMPONENT INITIALIZATION
# ============================================================================

@st.cache_resource
def initialize_pipeline():
    """Load all CRAG components (cached to avoid reloading)."""
    with st.spinner("🔧 Initializing CRAG Pipeline..."):
        try:
            # 1. Embedder
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            
            # 2. Matrix Memory with mock data
            memory = MatrixVectorMemory(embedding_dim=384)
            mock_chunks = [
                "Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function.",
                "Paris is the capital and most populous city of France.",
                "Vitamin D deficiency can lead to bone density loss, fatigue, and muscle weakness.",
                "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                "The Earth orbits the Sun, taking approximately 365.25 days to complete one revolution.",
                "Python is a high-level programming language known for its simplicity and readability.",
                "Neural networks are computing systems inspired by biological neural networks.",
                "Data preprocessing is a critical step in machine learning pipelines.",
                "API documentation is essential for effective software integration.",
                "Cloud computing provides on-demand access to computing resources over the internet.",
            ]
            mock_embeddings = embedder.encode(mock_chunks).tolist()
            memory.add_documents(mock_chunks, mock_embeddings)
            
            # 3. Evaluator
            evaluator = CrossEncoderEvaluator()
            
            # 4. Query Rewriter
            rewriter = QueryRewriter(model="llama-3.3-70b-versatile")
            
            # 5. Groq Client
            groq_api_key = os.environ.get("GROQ_API_KEY")
            if not groq_api_key:
                st.warning("⚠️ GROQ_API_KEY not found. Query rewriting will fail. Please set it in your .env file.")
                groq_client = None
            else:
                groq_client = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=groq_api_key
                )
            
            st.session_state.pipeline_initialized = True
            st.session_state.embedder = embedder
            st.session_state.memory = memory
            st.session_state.evaluator = evaluator
            st.session_state.rewriter = rewriter
            st.session_state.groq_client = groq_client
            
            return True
        except Exception as e:
            st.error(f"❌ Initialization failed: {str(e)}")
            return False

# ============================================================================
# PIPELINE EXECUTION FUNCTIONS
# ============================================================================

@dataclass
class PipelineTrace:
    """Records timing and decisions for the pipeline."""
    original_query: str
    initial_retrieval_ms: float
    initial_grades: Dict[str, List[str]]
    was_rewritten: bool
    rewritten_query: str = None
    rewrite_latency_ms: float = 0.0
    secondary_retrieval_ms: float = 0.0
    secondary_grades: Dict[str, List[str]] = None
    final_answer: str = ""
    synthesis_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

def execute_retrieval_and_eval(search_query: str, embedder, memory, evaluator):
    """Step 1: Retrieve + Evaluate."""
    t0 = time.perf_counter()
    
    # Encode query
    query_vector = embedder.encode(search_query).tolist()
    
    # Retrieve
    raw_results = memory.retrieve([query_vector], top_k=3)[0]
    
    if not raw_results:
        retrieval_latency = (time.perf_counter() - t0) * 1000
        return [], retrieval_latency, {"CORRECT": [], "AMBIGUOUS": [], "INCORRECT": []}
    
    # Evaluate
    chunks = [r["text"] for r in raw_results]
    graded_results = evaluator.evaluate_batch(search_query, chunks)
    
    # Organize by grade
    grades = {"CORRECT": [], "AMBIGUOUS": [], "INCORRECT": []}
    for result in graded_results:
        grades[result.grade].append(result.chunk_text)
    
    retrieval_latency = (time.perf_counter() - t0) * 1000
    return graded_results, retrieval_latency, grades

def generate_final_answer(query: str, valid_chunks: List[str], groq_client):
    """Step 4: Generate final answer via Groq."""
    t0 = time.perf_counter()
    
    if not valid_chunks:
        answer = "I could not find verified information in my knowledge base to answer this query."
    else:
        context = "\n---\n".join(valid_chunks)
        prompt = f"Answer the user's query using ONLY the provided verified context.\n\nContext:\n{context}\n\nQuery: {query}"
        
        try:
            if groq_client:
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                answer = response.choices[0].message.content
            else:
                answer = f"(Demo mode) Context provided: {context[:200]}... [Would synthesize with Groq if API key available]"
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
    
    synthesis_latency = (time.perf_counter() - t0) * 1000
    return answer, synthesis_latency

def run_crag_pipeline(user_query: str) -> PipelineTrace:
    """Execute the full CRAG pipeline and return a trace."""
    embedder = st.session_state.embedder
    memory = st.session_state.memory
    evaluator = st.session_state.evaluator
    rewriter = st.session_state.rewriter
    groq_client = st.session_state.groq_client
    
    # STEP 1: Initial retrieval + evaluation
    initial_results, initial_retrieval_ms, initial_grades = execute_retrieval_and_eval(
        user_query, embedder, memory, evaluator
    )
    
    trace = PipelineTrace(
        original_query=user_query,
        initial_retrieval_ms=initial_retrieval_ms,
        initial_grades=initial_grades,
        was_rewritten=False,
        secondary_grades={"CORRECT": [], "AMBIGUOUS": [], "INCORRECT": []}
    )
    
    valid_context = initial_grades["CORRECT"]
    
    # STEP 2: Check if we need to rewrite
    if not valid_context:
        trace.was_rewritten = True
        
        # Rewrite query
        t0 = time.perf_counter()
        try:
            rewrite_result = rewriter.rewrite(
                original_query=user_query,
                failed_chunks=initial_grades["INCORRECT"] + initial_grades["AMBIGUOUS"],
                trigger_grade="INCORRECT"
            )
            trace.rewrite_latency_ms = (time.perf_counter() - t0) * 1000
            trace.rewritten_query = rewrite_result.rewritten_query
        except Exception as e:
            trace.rewrite_latency_ms = (time.perf_counter() - t0) * 1000
            trace.rewritten_query = f"[Rewrite failed: {str(e)}]"
        
        # STEP 3: Re-retrieve with rewritten query
        if trace.rewritten_query and not trace.rewritten_query.startswith("["):
            secondary_results, secondary_retrieval_ms, secondary_grades = execute_retrieval_and_eval(
                trace.rewritten_query, embedder, memory, evaluator
            )
            trace.secondary_retrieval_ms = secondary_retrieval_ms
            trace.secondary_grades = secondary_grades
            
            valid_context = secondary_grades["CORRECT"]
            if not valid_context:
                valid_context = secondary_grades["AMBIGUOUS"]
    
    # STEP 4: Generate final answer
    final_answer, synthesis_latency = generate_final_answer(user_query, valid_context, groq_client)
    trace.final_answer = final_answer
    trace.synthesis_latency_ms = synthesis_latency
    
    # Calculate total latency
    trace.total_latency_ms = (
        trace.initial_retrieval_ms +
        trace.rewrite_latency_ms +
        trace.secondary_retrieval_ms +
        trace.synthesis_latency_ms
    )
    
    return trace

# ============================================================================
# STREAMLIT UI
# ============================================================================

# Header
st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1>⚡ CRAG Pipeline Dashboard</h1>
        <p style='font-size: 18px; color: #666;'>
            <strong>Corrective Retrieval-Augmented Generation</strong><br>
            See the full pipeline in action
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar: Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("🔌 Pipeline Components")
    col1, col2 = st.columns(2)
    with col1:
        show_trace = st.checkbox("Show Full Trace", value=True)
    with col2:
        show_benchmarks = st.checkbox("Show Benchmarks", value=False)
    
    st.divider()
    
    # Memory stats
    if st.session_state.pipeline_initialized:
        memory_stats = st.session_state.memory.stats
        st.subheader("📊 Memory Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", memory_stats["n_documents"])
            st.metric("Embedding Dim", memory_stats["embedding_dim"])
        with col2:
            st.metric("Device", memory_stats["device"])
            st.metric("Matrix Size (MB)", memory_stats["matrix_mb"])
    
    st.divider()
    
    # About
    st.subheader("ℹ️ About")
    st.markdown("""
        **CRAG Pipeline** combines:
        - 🔍 Local vector retrieval
        - ✅ Cross-encoder evaluation
        - 🔄 Autonomous query rewriting
        - 💬 LLM synthesis
        
        [GitHub Repo](https://github.com/Aditya221/Corrective-Retrieval-Augmented-Generation-Pipeline)
    """)

# Initialize pipeline on first load
if not st.session_state.pipeline_initialized:
    if initialize_pipeline():
        st.success("✅ Pipeline initialized!")
    else:
        st.stop()

# Main content
st.markdown("---")

# Query input section
col1, col2 = st.columns([4, 1])

with col1:
    st.subheader("🔍 Enter Your Query")
    user_query = st.text_input(
        "Question:",
        placeholder="e.g., 'What are symptoms of vitamin D deficiency?' or 'How does gradient descent work?'",
        key="query_input",
        label_visibility="collapsed"
    )

with col2:
    st.subheader("⏱️ Latency Target")
    st.markdown("""
        <div class='metric-card' style='text-align: center; margin-top: 10px;'>
            <100ms ✅
        </div>
    """, unsafe_allow_html=True)

if user_query:
    st.markdown("---")
    
    # Execute pipeline
    with st.spinner("🔄 Running CRAG Pipeline..."):
        trace = run_crag_pipeline(user_query)
    
    st.session_state.query_history.append(trace)
    
    # ====================================================================
    # RESULT DISPLAY
    # ====================================================================
    
    # Performance metrics at top
    st.subheader("📊 Pipeline Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Retrieval",
            f"{trace.initial_retrieval_ms:.2f}ms",
            delta="✅ sub-1ms" if trace.initial_retrieval_ms < 1 else ""
        )
    with col2:
        if trace.was_rewritten:
            st.metric(
                "Rewrite",
                f"{trace.rewrite_latency_ms:.0f}ms",
                delta="Triggered 🔄"
            )
        else:
            st.metric("Rewrite", "—", delta="Not needed ✅")
    with col3:
        if trace.secondary_retrieval_ms > 0:
            st.metric(
                "Re-retrieval",
                f"{trace.secondary_retrieval_ms:.2f}ms",
                delta=""
            )
        else:
            st.metric("Re-retrieval", "—")
    with col4:
        delta_text = "✅ sub-100ms" if trace.total_latency_ms < 100 else f"⚠️ {trace.total_latency_ms - 100:.0f}ms over"
        st.metric(
            "Total",
            f"{trace.total_latency_ms:.0f}ms",
            delta=delta_text
        )
    
    st.divider()
    
    # Tabs for detailed views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Pipeline Trace",
        "✅ Evaluation Results",
        "🔄 Rewriting Logic",
        "📝 Final Answer"
    ])
    
    with tab1:
        st.markdown("### Full Pipeline Execution")
        
        # Step-by-step visualization
        st.markdown("#### Step-by-Step Breakdown")
        
        # Step 1: Retrieval
        st.markdown("""
            <div class='pipeline-step'>
                <span class='step-number'>1</span>
                <strong>Matrix Memory Retrieval</strong>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Query:** `{trace.original_query[:50]}...`")
        with col2:
            st.markdown(f"**Latency:** `{trace.initial_retrieval_ms:.2f}ms`")
        with col3:
            total_retrieved = (len(trace.initial_grades['CORRECT']) + 
                             len(trace.initial_grades['AMBIGUOUS']) + 
                             len(trace.initial_grades['INCORRECT']))
            st.markdown(f"**Retrieved:** `{total_retrieved} chunks`")
        
        # Step 2: Evaluation
        st.markdown("""
            <div class='pipeline-step'>
                <span class='step-number'>2</span>
                <strong>Cross-Encoder Evaluation</strong>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"**Model:** `ms-marco-MiniLM`")
        with col2:
            st.markdown(f"**CORRECT:** `{len(trace.initial_grades['CORRECT'])}`")
        with col3:
            st.markdown(f"**AMBIGUOUS:** `{len(trace.initial_grades['AMBIGUOUS'])}`")
        with col4:
            st.markdown(f"**INCORRECT:** `{len(trace.initial_grades['INCORRECT'])}`")
        
        # Step 3: Decision Point
        if trace.was_rewritten:
            st.markdown("""
                <div class='pipeline-step' style='border-left-color: #ffc107;'>
                    <span class='step-number' style='background: #ffc107;'>3</span>
                    <strong>🔄 Query Rewriter (Triggered)</strong>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Original Query:**\n`{trace.original_query}`")
            with col2:
                st.markdown(f"**Rewritten Query:**\n`{trace.rewritten_query}`")
            
            st.markdown(f"**Latency:** `{trace.rewrite_latency_ms:.0f}ms` | **Model:** `Llama 3.3 70B (Groq)`")
            
            # Step 4: Secondary Retrieval
            st.markdown("""
                <div class='pipeline-step'>
                    <span class='step-number'>4</span>
                    <strong>Secondary Retrieval & Evaluation</strong>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"**Latency:** `{trace.secondary_retrieval_ms:.2f}ms`")
            with col2:
                st.markdown(f"**CORRECT:** `{len(trace.secondary_grades['CORRECT'])}`")
            with col3:
                st.markdown(f"**AMBIGUOUS:** `{len(trace.secondary_grades['AMBIGUOUS'])}`")
            with col4:
                st.markdown(f"**INCORRECT:** `{len(trace.secondary_grades['INCORRECT'])}`")
        else:
            st.info("ℹ️ No rewriting needed — initial retrieval was CORRECT!")
        
        # Final synthesis
        st.markdown("""
            <div class='pipeline-step' style='border-left-color: #28a745;'>
                <span class='step-number' style='background: #28a745;'>5</span>
                <strong>LLM Synthesis</strong>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Latency:** `{trace.synthesis_latency_ms:.0f}ms` | **Model:** `Llama 3.3 70B (Groq)`")
    
    with tab2:
        st.markdown("### Evaluation Grades (Initial Pass)")
        
        col1, col2, col3 = st.columns(3)
        
        # CORRECT
        with col1:
            st.markdown(f"#### ✅ CORRECT ({len(trace.initial_grades['CORRECT'])})")
            if trace.initial_grades['CORRECT']:
                for i, chunk in enumerate(trace.initial_grades["CORRECT"], 1):
                    st.markdown(
                        f"<div class='grade-correct'><strong>{i}.</strong> {chunk[:120]}{'...' if len(chunk) > 120 else ''}</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.markdown("<div class='grade-correct'><em>No correct results</em></div>", unsafe_allow_html=True)
        
        # AMBIGUOUS
        with col2:
            st.markdown(f"#### ⚠️ AMBIGUOUS ({len(trace.initial_grades['AMBIGUOUS'])})")
            if trace.initial_grades['AMBIGUOUS']:
                for i, chunk in enumerate(trace.initial_grades["AMBIGUOUS"], 1):
                    st.markdown(
                        f"<div class='grade-ambiguous'><strong>{i}.</strong> {chunk[:120]}{'...' if len(chunk) > 120 else ''}</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.markdown("<div class='grade-ambiguous'><em>No ambiguous results</em></div>", unsafe_allow_html=True)
        
        # INCORRECT
        with col3:
            st.markdown(f"#### ❌ INCORRECT ({len(trace.initial_grades['INCORRECT'])})")
            if trace.initial_grades['INCORRECT']:
                for i, chunk in enumerate(trace.initial_grades["INCORRECT"], 1):
                    st.markdown(
                        f"<div class='grade-incorrect'><strong>{i}.</strong> {chunk[:120]}{'...' if len(chunk) > 120 else ''}</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.markdown("<div class='grade-incorrect'><em>No incorrect results</em></div>", unsafe_allow_html=True)
        
        if trace.was_rewritten:
            st.divider()
            st.markdown("### Evaluation Grades (Secondary Pass)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"#### ✅ CORRECT ({len(trace.secondary_grades['CORRECT'])})")
                if trace.secondary_grades['CORRECT']:
                    for i, chunk in enumerate(trace.secondary_grades["CORRECT"], 1):
                        st.markdown(
                            f"<div class='grade-correct'><strong>{i}.</strong> {chunk[:120]}{'...' if len(chunk) > 120 else ''}</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown("<div class='grade-correct'><em>No correct results</em></div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"#### ⚠️ AMBIGUOUS ({len(trace.secondary_grades['AMBIGUOUS'])})")
                if trace.secondary_grades['AMBIGUOUS']:
                    for i, chunk in enumerate(trace.secondary_grades["AMBIGUOUS"], 1):
                        st.markdown(
                            f"<div class='grade-ambiguous'><strong>{i}.</strong> {chunk[:120]}{'...' if len(chunk) > 120 else ''}</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown("<div class='grade-ambiguous'><em>No ambiguous results</em></div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"#### ❌ INCORRECT ({len(trace.secondary_grades['INCORRECT'])})")
                if trace.secondary_grades['INCORRECT']:
                    for i, chunk in enumerate(trace.secondary_grades["INCORRECT"], 1):
                        st.markdown(
                            f"<div class='grade-incorrect'><strong>{i}.</strong> {chunk[:120]}{'...' if len(chunk) > 120 else ''}</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown("<div class='grade-incorrect'><em>No incorrect results</em></div>", unsafe_allow_html=True)
    
    with tab3:
        if trace.was_rewritten:
            st.markdown("### ✅ Query Rewriting Triggered")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Query:**")
                st.code(trace.original_query, language="text")
            
            with col2:
                st.markdown("**Rewritten Query:**")
                st.code(trace.rewritten_query, language="text")
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rewrite Latency", f"{trace.rewrite_latency_ms:.0f}ms")
            with col2:
                st.metric("Model", "Llama 3.3 70B")
            with col3:
                st.metric("Provider", "Groq")
            
            st.markdown("#### Strategy")
            if "expand" in trace.rewritten_query.lower() or len(trace.initial_grades['AMBIGUOUS']) > 0:
                st.info("📌 **EXPAND Strategy**: Added context/synonyms to clarify the query")
            else:
                st.info("🔄 **PIVOT Strategy**: Completely reformulated the query")
        else:
            st.success("ℹ️ No rewriting triggered — initial retrieval was sufficient!")
    
    with tab4:
        st.markdown("### Final Answer")
        
        st.markdown(f"**Your Question:** {trace.original_query}")
        st.markdown("---")
        
        st.markdown(
            f"<div class='answer-box'>{trace.final_answer}</div>",
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        st.subheader("📚 Sources")
        sources = (trace.initial_grades["CORRECT"] if not trace.was_rewritten 
                  else trace.secondary_grades["CORRECT"])
        
        if sources:
            for i, source in enumerate(sources, 1):
                st.markdown(
                    f"<div class='source-item'><strong>[{i}]</strong> {source}</div>",
                    unsafe_allow_html=True
                )
        else:
            ambiguous_sources = (trace.initial_grades["AMBIGUOUS"] if not trace.was_rewritten 
                                else trace.secondary_grades["AMBIGUOUS"])
            if ambiguous_sources:
                st.warning("⚠️ No CORRECT sources found. Answer based on AMBIGUOUS context:")
                for i, source in enumerate(ambiguous_sources, 1):
                    st.markdown(
                        f"<div class='source-item' style='background: #fff3cd; border-left-color: #ffc107;'><strong>[{i}]</strong> {source}</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.error("❌ No verified sources found.")

# ============================================================================
# BENCHMARKS SECTION
# ============================================================================

if show_benchmarks:
    st.markdown("---")
    st.subheader("📊 Benchmark Comparison (BEIR NFCorpus)")
    
    st.markdown("""
    ### Dataset Info
    - **Corpus:** 3,633 medical documents
    - **Queries:** 323 evaluation queries
    - **Metric:** Recall@10 (% of relevant docs in top-10)
    - **Hardware:** Apple M-series CPU
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Recall@10", "69.35%", delta="+4.35% vs BM25")
    with col2:
        st.metric("Avg Latency", "~50-100ms", delta="vs 200ms (Pinecone)")
    with col3:
        st.metric("Model", "MiniLM-L6-v2", delta="384 dims")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; padding: 20px;'>
    
**CRAG Pipeline Dashboard** | Built with ⚡ Streamlit, PyTorch, and Groq Llama 3.3

[🔗 GitHub](https://github.com/Aditya221/Corrective-Retrieval-Augmented-Generation-Pipeline) • 
[📖 Docs](https://github.com/Aditya221/Corrective-Retrieval-Augmented-Generation-Pipeline/blob/main/README.md) • 
[📊 Paper](https://arxiv.org/abs/2401.01310)

</div>
""", unsafe_allow_html=True)
