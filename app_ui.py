"""
Streamlit UI for CRAG Pipeline (Thin Client)
Visualizes the complete retrieval → evaluation → rewriting → synthesis loop.
Communicates with the FastAPI backend via HTTP requests.
"""

import streamlit as st
import requests
import time
import os

# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================

st.set_page_config(
    page_title="CRAG Pipeline Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
if "query_history" not in st.session_state:
    st.session_state.query_history = []

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
    
    st.subheader("🔌 Backend Connection")
    # Allow user to input their Ngrok URL or localhost
    api_url = st.text_input(
        "API Endpoint URL", 
        value="http://localhost:8080/ask",
        help="Use an Ngrok URL here if running Streamlit Cloud but hosting the backend locally."
    )
    
    show_benchmarks = st.checkbox("Show Benchmarks", value=True)
    
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
            <1s ✅
        </div>
    """, unsafe_allow_html=True)

if user_query:
    st.markdown("---")
    
    # Execute pipeline via HTTP Request
    with st.spinner("🔄 Sending query to CRAG Backend..."):
        try:
            start_time = time.perf_counter()
            response = requests.post(
                api_url,
                json={"query": user_query},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            network_latency = (time.perf_counter() - start_time) * 1000
            
            # Extract backend data
            answer = data.get("answer", "No answer generated.")
            sources = data.get("sources", [])
            pipeline_latency = data.get("pipeline_latency_ms", 0.0)
            rewritten = data.get("rewritten", False)
            
            st.session_state.query_history.append({"query": user_query, "data": data})
            
            # ====================================================================
            # RESULT DISPLAY
            # ====================================================================
            
            st.subheader("📊 Pipeline Performance")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Self-Healing Triggered",
                    "Yes 🔄" if rewritten else "No ✅",
                    delta="Fallback Used" if rewritten else "Direct Hit"
                )
            with col2:
                st.metric(
                    "Backend Processing Time",
                    f"{pipeline_latency:.2f}ms"
                )
            with col3:
                st.metric(
                    "Total Roundtrip (inc. Network)",
                    f"{network_latency:.2f}ms"
                )
            
            st.divider()
            
            # Tabs for detailed views
            tab1, tab2, tab3 = st.tabs([
                "📝 Final Answer",
                "📊 Pipeline Trace",
                "📚 Retrieved Sources"
            ])
            
            with tab1:
                st.markdown("### Synthesized Output")
                st.markdown(f"**Your Question:** {user_query}")
                st.markdown(
                    f"<div class='answer-box'>{answer}</div>",
                    unsafe_allow_html=True
                )
            
            with tab2:
                st.markdown("### Step-by-Step Execution Path")
                
                # Step 1: Retrieval
                st.markdown("""
                    <div class='pipeline-step'>
                        <span class='step-number'>1</span>
                        <strong>Matrix Memory Retrieval (PyTorch)</strong>
                    </div>
                """, unsafe_allow_html=True)
                
                # Step 2: Evaluation
                st.markdown("""
                    <div class='pipeline-step'>
                        <span class='step-number'>2</span>
                        <strong>Cross-Encoder Evaluation (ms-marco-MiniLM)</strong>
                    </div>
                """, unsafe_allow_html=True)
                
                # Step 3: Conditional Rewriting
                if rewritten:
                    st.markdown("""
                        <div class='pipeline-step' style='border-left-color: #ffc107;'>
                            <span class='step-number' style='background: #ffc107;'>3</span>
                            <strong>🔄 Query Rewriter (Triggered)</strong><br>
                            <small><em>Initial chunks were graded INCORRECT. Llama 3 pivoted the query.</em></small>
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                        <div class='pipeline-step'>
                            <span class='step-number'>4</span>
                            <strong>Secondary Retrieval (PyTorch)</strong>
                        </div>
                    """, unsafe_allow_html=True)
                    step_num = 5
                else:
                    st.info("ℹ️ No rewriting needed — initial retrieval was graded CORRECT!")
                    step_num = 3
                
                # Final synthesis
                st.markdown(f"""
                    <div class='pipeline-step' style='border-left-color: #28a745;'>
                        <span class='step-number' style='background: #28a745;'>{step_num}</span>
                        <strong>LLM Synthesis (Groq Llama 3.3 70B)</strong>
                    </div>
                """, unsafe_allow_html=True)
                
            with tab3:
                st.markdown("### Verified Context Used")
                if sources:
                    for i, source in enumerate(sources, 1):
                        st.markdown(
                            f"<div class='source-item'><strong>[{i}]</strong> {source}</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.error("❌ No verified sources found.")

        except requests.exceptions.ConnectionError:
            st.error(f"❌ Connection failed. Ensure your backend is running at {api_url}.")
            st.info("💡 Tip: If your UI is on Streamlit Cloud, you must use a public URL (like Ngrok) in the sidebar instead of localhost.")
        except Exception as e:
            st.error(f"❌ Pipeline Error: {str(e)}")

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
        st.metric("Avg Retrieval Latency", "0.08ms", delta="vs 50ms API DBs")
    with col3:
        st.metric("Model", "MiniLM-L6-v2", delta="384 dims")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; padding: 20px;'>
**CRAG Pipeline Dashboard** | Built with ⚡ Streamlit, PyTorch, and Groq Llama 3.3
</div>
""", unsafe_allow_html=True)
