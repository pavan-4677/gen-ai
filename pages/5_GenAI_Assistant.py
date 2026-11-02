"""
Page 5: Generative AI Dataset Assistant
Context-grounded AI reasoning module providing executive dataset summaries,
anomaly explanations, cleaning recommendations, ML readiness assessments,
and interactive dataset Q&A chat.
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

# 5 Dedicated Feature Tabs
tab_report, tab_anomalies, tab_recs, tab_ml, tab_chat = st.tabs([
    "📑 Executive AI Quality Report",
    "🔍 Anomalies Deep Dive",
    "🧹 AI Cleaning Recommendations",
    "🎯 ML Readiness Assessment",
    "💬 Interactive AI Chatbot"
])

# =============================================================================
# Tab 1: Executive AI Quality Report & Summary
# =============================================================================
with tab_report:
    st.subheader("Executive Dataset Summary & Quality Report")
    
    # 1. AI-generated Dataset Summary Card
    st.markdown("#### 1. Dataset Overview & Characteristics")
    with st.spinner("Synthesizing structural overview..."):
        ai_summary = ai_service.generate_dataset_summary(diagnosis)
        st.markdown(
            f"""
            <div style="background:rgba(15,23,42,0.8);border:1px solid rgba(56,189,248,0.25);border-radius:10px;padding:1.25rem 1.4rem;margin-bottom:1.5rem;line-height:1.65;">
                {ai_summary}
            </div>
            """,
            unsafe_allow_html=True
        )

    # 2. Comprehensive AI Report
    st.markdown("#### 2. Comprehensive Diagnostic Synthesis")
    col_btn, _ = st.columns([1.2, 3])
    with col_btn:
        regen_report = st.button("🔄 Refresh Comprehensive Report", type="primary", use_container_width=True)

    if regen_report or st.session_state.get("ai_summary_cache") is None:
        with st.spinner("Synthesizing data quality analysis into natural language..."):
            report_text = ai_service.generate_full_quality_report(diagnosis)
            st.session_state["ai_summary_cache"] = report_text

    st.markdown("<hr style='border-color:rgba(56,189,248,0.12);'>", unsafe_allow_html=True)
    st.markdown(st.session_state.get("ai_summary_cache", ""))

# =============================================================================
# Tab 2: Anomalies Deep Dive (Missing, Duplicates, Outliers)
# =============================================================================
with tab_anomalies:
    st.subheader("AI Technical Analysis of Detected Anomalies")
    st.caption("Context-grounded evaluations explaining why specific anomalies exist and how they distort downstream processing.")

    with st.spinner("Generating anomaly technical diagnostics..."):
        anomaly_exps = ai_service.explain_anomalies(diagnosis)

    # 1. Missing Values Card
    st.markdown("### ❓ Missing Values & Sparsity Analysis")
    st.markdown(anomaly_exps["missing_values_explanation"])
    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Duplicate Records Card
    st.markdown("### 👥 Duplicate Record Evaluation")
    st.markdown(anomaly_exps["duplicates_explanation"])
    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Statistical Outliers Card (IQR Method)
    st.markdown("### 📈 Statistical Outliers (IQR Method) Analysis")
    st.markdown(anomaly_exps["outliers_explanation"])

# =============================================================================
# Tab 3: AI-Based Data Cleaning Recommendations
# =============================================================================
with tab_recs:
    st.subheader("AI-Engineered Step-by-Step Data Cleaning Roadmap")
    st.caption("Actionable remediation strategies tailored specifically to the detected issues, directly actionable in the Cleaning Studio.")

    with st.spinner("Synthesizing remediation strategy..."):
        recs_text = ai_service.generate_cleaning_recommendations(diagnosis)

    st.markdown(
        f"""
        <div style="background:rgba(15,23,42,0.85);border:1px solid rgba(99,102,241,0.25);border-radius:12px;padding:1.5rem 1.8rem;line-height:1.7;">
            {recs_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🧹 Launch Data Cleaning Studio with these Recommendations", type="primary"):
        st.switch_page("pages/6_Data_Cleaning_Studio.py")

# =============================================================================
# Tab 4: Machine Learning Readiness Assessment
# =============================================================================
with tab_ml:
    st.subheader("Dataset Machine Learning Readiness Assessment")
    st.caption("Evaluates estimator stability, risk of overfitting/data leakage, and architecture suitability.")

    ml_eval = ai_service.assess_ml_readiness(diagnosis)

    # Readiness Scorecard Row
    col_r1, col_r2 = st.columns([1, 2])
    with col_r1:
        st.metric(
            "Overall Health Score",
            f"{ml_eval['overall_score']}/100",
            delta=ml_eval["status"],
            delta_color="normal" if ml_eval["status"] == "HIGH" else ("off" if ml_eval["status"] == "MODERATE" else "inverse")
        )
        st.markdown(f"**Readiness Status:** {ml_eval['badge']}", unsafe_allow_html=True)

    with col_r2:
        st.markdown(
            f"""
            <div class="info-box">
                <strong>Executive ML Evaluation:</strong> {ml_eval['summary_evaluation']}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<hr style='border-color:rgba(56,189,248,0.12);'>", unsafe_allow_html=True)

    # Model Family Compatibility Matrix
    st.markdown("#### Model Family Compatibility Matrix")
    for key, model_info in ml_eval["model_suitability"].items():
        comp = model_info["compatibility"]
        badge_style = "color:#10B981;" if comp == "High" else ("color:#F59E0B;" if comp == "Moderate" else "color:#F43F5E;")
        with st.expander(f"⚙️ {model_info['name']} — Compatibility: {comp}", expanded=True):
            st.markdown(f"**Compatibility Level:** <span style='{badge_style}font-weight:700;'>{comp}</span>", unsafe_allow_html=True)
            st.markdown(f"**Technical Consideration:** {model_info['notes']}")

    # Mandatory Preprocessing Checklist
    st.markdown("<br>#### Pre-Training Checklist (`model.fit()` Prerequisites)", unsafe_allow_html=True)
    check_cols = st.columns(len(ml_eval["checklist"]))
    for idx, item in enumerate(ml_eval["checklist"]):
        with check_cols[idx]:
            status_icon = "✅" if item["status"] == "Done" else ("🚨" if item["critical"] else "🟡")
            st.markdown(
                f"""
                <div style="background:#0F172A;border:1px solid rgba(56,189,248,0.15);border-radius:8px;padding:0.9rem;text-align:center;">
                    <div style="font-size:1.3rem;">{status_icon}</div>
                    <div style="font-weight:700;font-size:0.85rem;margin-top:0.3rem;">{item['item']}</div>
                    <div style="font-size:0.75rem;color:#94A3B8;">{item['status']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

# =============================================================================
# Tab 5: Interactive AI Chatbot
# =============================================================================
with tab_chat:
    st.subheader("Interactive Dataset Assistant Chat")
    st.caption("Ask questions regarding data quality, specific column statistics, ML feasibility, or recommended preprocessing steps.")

    # Suggested Prompts Chips Row
    st.markdown("**Suggested Questions (Click to Ask):**")
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
                f"Hello! I am your Intelligent Dataset Assistant. I have analyzed **'{st.session_state.get('active_dataset_name')}'**. "
                f"Its current Quality Score is **{diagnosis['quality_scores']['overall_score']}/100** ({diagnosis['quality_scores']['grade']}). "
                "Ask me any question about missing values, statistical outliers, ML readiness, or recommended cleaning strategies!"
            )
        for msg in st.session_state["ai_chat_history"]:
            st.chat_message(msg["role"]).write(msg["content"])

    # Chat Input handler
    user_input = st.chat_input("Ask a question about your dataset (e.g. 'How should I handle missing values?')...")
    active_query = preset_query or user_input

    if active_query:
        st.session_state["ai_chat_history"].append({"role": "user", "content": active_query})
        st.chat_message("user").write(active_query)

        with st.spinner("Analyzing dataset metrics and formulating verified answer..."):
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
