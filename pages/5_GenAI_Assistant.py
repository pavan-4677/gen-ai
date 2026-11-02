"""
Page 5: Generative AI Dataset Assistant
Context-grounded AI reasoning module providing comprehensive dataset diagnosis,
root cause analysis, ML readiness risk evaluation, and interactive dataset Q&A chat.
"""

import streamlit as st
from src.utils import init_session_state, load_css, render_hero, render_footer, render_sidebar
from src.ai_service import AIService
from src.quality_analyzer import QualityAnalyzer

st.set_page_config(
    page_title="Gen AI Assistant | Dataset Quality Analysis",
    page_icon="🤖",
    layout="wide"
)

init_session_state()
load_css()
render_sidebar(current_step=4)

render_hero(
    title="Generative AI Dataset Assistant",
    subtitle="Context-grounded LLM synthesis for anomaly explanations, root cause discovery, and ML readiness counseling.",
    badge="Step 4 • Generative AI Reasoning"
)

if st.session_state.get("df_raw") is None:
    st.markdown(
        """
        <div class="warning-box">
            <strong>No Dataset in Memory:</strong> Please upload a dataset to enable the Generative AI Assistant.
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("📂 Go to Dataset Upload", type="primary"):
        st.switch_page("pages/2_Upload_Dataset.py")
    st.stop()

df = st.session_state["df_raw"]
diagnosis = st.session_state.get("diagnosis_results")

if diagnosis is None:
    with st.spinner("Analyzing dataset metrics..."):
        analyzer = QualityAnalyzer()
        diagnosis = analyzer.analyze(df, dataset_name=st.session_state.get("active_dataset_name", "Dataset"))
        st.session_state["diagnosis_results"] = diagnosis
        st.session_state["quality_scores"] = diagnosis["quality_scores"]

# Initialize AI Service
settings = st.session_state.get("settings", {})
ai_service = AIService(
    api_key=settings.get("openai_api_key"),
    model=settings.get("openai_model", "gpt-4o-mini"),
    base_url=settings.get("openai_base_url", "https://api.openai.com/v1")
)

# Engine connection banner
if ai_service.is_api_configured():
    st.markdown(
        f"""
        <div class="success-box">
            <strong>🟢 Live LLM Active:</strong> Connected to <code>{ai_service.model}</code> via <code>{ai_service.base_url}</code>. 
            Structured JSON summaries are being dispatched for dynamic natural language completions.
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div class="info-box">
            <strong>🔵 Heuristic Diagnostic Engine Active (Offline Mode):</strong> 
            No OpenAI API key detected. The assistant is operating using its internal expert diagnostic engine based 
            strictly on the actual diagnosed metrics. You can connect a live OpenAI API key in <strong>Settings</strong> at any time.
        </div>
        """,
        unsafe_allow_html=True
    )

tab_report, tab_chat = st.tabs(["📋 Comprehensive AI Diagnostic Report", "💬 Interactive Dataset Q&A Chat"])

# 1. Comprehensive AI Report Tab
with tab_report:
    st.subheader("Generative Data Quality Health Assessment")
    st.caption("Synthesized automatically from dataset profiling metrics, IQR outliers, and consistency checks.")

    col_btn, _ = st.columns([1.2, 3])
    with col_btn:
        regen_report = st.button("🔄 Generate / Refresh AI Analysis", type="primary", use_container_width=True)

    if regen_report or st.session_state.get("ai_summary_cache") is None:
        with st.spinner("Synthesizing data quality analysis into natural language..."):
            report_text = ai_service.generate_comprehensive_explanation(diagnosis)
            st.session_state["ai_summary_cache"] = report_text

    st.markdown("<hr style='border-color:rgba(56,189,248,0.12);'>", unsafe_allow_html=True)
    st.markdown(st.session_state.get("ai_summary_cache", ""))

# 2. Interactive Dataset Q&A Chat Tab
with tab_chat:
    st.subheader("Ask Contextual Questions About Your Dataset")
    st.caption("Query detected anomalies, missing data, ML feasibility, or recommended preprocessing steps.")

    # Suggested quick-question prompts
    st.markdown("**Suggested Prompts (Click to Query):**")
    prompt_cols = st.columns(3)
    preset_query = None

    with prompt_cols[0]:
        if st.button("What is wrong with my dataset?", use_container_width=True):
            preset_query = "What is wrong with my dataset?"
        if st.button("Why is my data quality score low?", use_container_width=True):
            preset_query = "Why is my data quality score low?"

    with prompt_cols[1]:
        if st.button("Which columns have the most missing values?", use_container_width=True):
            preset_query = "Which columns have the most missing values?"
        if st.button("Which columns contain outliers?", use_container_width=True):
            preset_query = "Which columns contain outliers?"

    with prompt_cols[2]:
        if st.button("Is this dataset ready for machine learning?", use_container_width=True):
            preset_query = "Is this dataset ready for machine learning?"
        if st.button("Explain this dataset in simple English", use_container_width=True):
            preset_query = "Explain this dataset in simple English"

    st.markdown("<hr style='border-color:rgba(56,189,248,0.12);'>", unsafe_allow_html=True)

    # Render Chat History
    chat_container = st.container()
    with chat_container:
        if not st.session_state["ai_chat_history"]:
            st.chat_message("assistant").write(
                f"Hello! I have reviewed the diagnostics for **'{st.session_state.get('active_dataset_name')}'**. "
                f"Overall Data Quality Score is **{diagnosis['quality_scores']['overall_score']}/100** ({diagnosis['quality_scores']['grade']}). "
                "How can I assist you with diagnosing or preparing this data?"
            )
        for msg in st.session_state["ai_chat_history"]:
            st.chat_message(msg["role"]).write(msg["content"])

    # Chat Input handler
    user_input = st.chat_input("Ask a question about your dataset (e.g. 'How should I handle missing values?')...")
    active_query = preset_query or user_input

    if active_query:
        # Display user query
        st.session_state["ai_chat_history"].append({"role": "user", "content": active_query})
        st.chat_message("user").write(active_query)

        # Generate response
        with st.spinner("Analyzing dataset metrics..."):
            ai_reply = ai_service.answer_user_query(
                user_query=active_query,
                chat_history=st.session_state["ai_chat_history"],
                diagnosis=diagnosis
            )
            st.session_state["ai_chat_history"].append({"role": "assistant", "content": ai_reply})
            st.chat_message("assistant").write(ai_reply)

    if st.session_state["ai_chat_history"]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Conversation History"):
            st.session_state["ai_chat_history"] = []
            st.rerun()

render_footer()
