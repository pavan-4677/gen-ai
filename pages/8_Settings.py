"""
Page 8: Settings & Configuration
API key management, model selection, scoring formula weights, outlier thresholds,
dataset privacy disclosures, and session management.
"""

import os
import streamlit as st

from src.utils import init_session_state, load_css, render_hero, render_footer

st.set_page_config(
    page_title="Settings | Dataset Quality Analysis",
    page_icon="⚙️",
    layout="wide"
)

init_session_state()
load_css()

render_hero(
    title="System Settings & LLM Configuration",
    subtitle="Configure Generative AI model parameters, scoring formula weights, privacy rules, and session state.",
    badge="Configuration"
)

settings = st.session_state["settings"]

tab_llm, tab_scoring, tab_privacy, tab_session = st.tabs([
    "🤖 Generative AI & API Keys",
    "⚖️ Quality Scoring Weights & IQR",
    "🔒 Privacy & Security Disclosures",
    "⚠️ Session State & Reset"
])

# 1. LLM & API Keys Tab
with tab_llm:
    st.subheader("Generative AI Engine Configuration")
    st.markdown("""
    The system supports **OpenAI API** as well as any OpenAI-compatible LLM endpoint (e.g. Azure OpenAI, Groq, Ollama, vLLM).
    **Security Notice:** API keys entered here remain encrypted in browser session memory and are never written to disk, logged, or exposed in exported reports.
    """)

    col_llm1, col_llm2 = st.columns(2)
    with col_llm1:
        current_key = settings.get("openai_api_key", "")
        # Masked key preview
        display_key = current_key if not current_key or current_key.startswith("your_") else f"{current_key[:4]}••••••••{current_key[-4:]}"
        st.caption(f"Current Status: **{'🟢 Configured' if current_key and not current_key.startswith('your_') else '⚪ Offline Mode (No Key Set)'}**")

        new_key = st.text_input(
            "OpenAI API Key:",
            type="password",
            value="" if current_key.startswith("your_") else current_key,
            placeholder="sk-proj-...",
            help="Enter your API key or configure it in the .env file as OPENAI_API_KEY."
        )

        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "custom"]
        current_model = settings.get("openai_model", "gpt-4o-mini")
        selected_model_idx = model_options.index(current_model) if current_model in model_options else 3
        model_choice = st.selectbox("LLM Model:", model_options, index=selected_model_idx)

        if model_choice == "custom":
            final_model = st.text_input("Enter custom model identifier:", value=current_model)
        else:
            final_model = model_choice

        base_url = st.text_input(
            "API Base URL:",
            value=settings.get("openai_base_url", "https://api.openai.com/v1"),
            help="Default: https://api.openai.com/v1. Point to custom proxies or local models if desired."
        )

        if st.button("Save AI Configuration", type="primary"):
            settings["openai_api_key"] = new_key.strip()
            settings["openai_model"] = final_model.strip()
            settings["openai_base_url"] = base_url.strip()
            st.session_state["settings"] = settings
            st.session_state["ai_summary_cache"] = None
            st.success("AI configuration saved successfully!")
            st.rerun()

    with col_llm2:
        st.markdown("#### How to Set Environment Variables (.env)")
        st.markdown("""
        To persist your API key without re-entering it in the browser UI, create a `.env` file in the root directory:
        ```bash
        # .env file
        OPENAI_API_KEY=sk-proj-your-actual-api-key-here
        OPENAI_MODEL=gpt-4o-mini
        OPENAI_BASE_URL=https://api.openai.com/v1
        ```
        The application automatically loads `.env` on startup via `python-dotenv`.
        
        **Offline Engine Note:** Non-AI statistical profiling, IQR detection, and data cleaning remain 100% operational without an API key.
        """)

# 2. Quality Scoring & IQR Tab
with tab_scoring:
    st.subheader("Data Quality Scoring Formula Customization")
    st.markdown("""
    The overall quality score is calculated as a transparent weighted sum of 4 core dimensions:
    $$\\text{Overall Score} = w_c \\times \\text{Completeness} + w_u \\times \\text{Uniqueness} + w_v \\times \\text{Validity} + w_s \\times \\text{Consistency}$$
    """)

    weights = settings.get("scoring_weights", {"completeness": 0.25, "uniqueness": 0.25, "validity": 0.25, "consistency": 0.25})
    
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        w_comp = st.slider("Completeness Weight (Missing Data Penalty):", 0.0, 1.0, float(weights.get("completeness", 0.25)), 0.05)
        w_uniq = st.slider("Uniqueness Weight (Duplicate Penalty):", 0.0, 1.0, float(weights.get("uniqueness", 0.25)), 0.05)
        w_val = st.slider("Validity Weight (Placeholder / Format Penalty):", 0.0, 1.0, float(weights.get("validity", 0.25)), 0.05)
        w_cons = st.slider("Consistency Weight (Outliers / Casing Penalty):", 0.0, 1.0, float(weights.get("consistency", 0.25)), 0.05)

        total_w = w_comp + w_uniq + w_val + w_cons
        st.caption(f"Raw weight sum: **{total_w:.2f}** (automatically normalized to 1.0 during scoring)")

    with col_w2:
        st.markdown("#### Outlier Detection Threshold")
        iqr_thresh = st.slider(
            "IQR Multiplier Threshold:",
            min_value=1.0,
            max_value=3.0,
            value=float(settings.get("iqr_threshold", 1.5)),
            step=0.1,
            help="1.5 corresponds to standard Tukey outliers. 3.0 corresponds to extreme outliers."
        )

        if st.button("Apply Scoring Parameters", type="primary"):
            settings["scoring_weights"] = {
                "completeness": w_comp,
                "uniqueness": w_uniq,
                "validity": w_val,
                "consistency": w_cons
            }
            settings["iqr_threshold"] = iqr_thresh
            st.session_state["settings"] = settings
            # Trigger re-diagnosis if dataset loaded
            if st.session_state.get("df_raw") is not None:
                from src.quality_analyzer import QualityAnalyzer
                analyzer = QualityAnalyzer(iqr_multiplier=iqr_thresh, scoring_weights=settings["scoring_weights"])
                st.session_state["diagnosis_results"] = analyzer.analyze(
                    st.session_state["df_raw"], dataset_name=st.session_state.get("active_dataset_name", "Dataset")
                )
                st.session_state["quality_scores"] = st.session_state["diagnosis_results"]["quality_scores"]
            st.success("Scoring weights and outlier thresholds updated.")
            st.rerun()

# 3. Privacy & Security Tab
with tab_privacy:
    st.subheader("Data Privacy & Architecture Security Standards")
    st.markdown("""
    ### 🔒 Security Principles
    1. **Zero Raw Data Transmission**: The Generative AI module is strictly metadata-driven. Raw user records are never transmitted to external LLM servers. Only aggregate statistical metrics (null counts, quartiles, and sanitized column names) are forwarded.
    2. **Prompt Injection Mitigation**: All user inputs and dataset headers are sanitized through regex scrubbing prior to inclusion in LLM prompts to prevent instruction overriding.
    3. **Ephemeral In-Memory Storage**: Uploaded files and cleaned datasets are processed purely in RAM using temporary session buffers. No files are permanently retained on disk or cloud storage.
    4. **Credential Isolation**: API keys are accessed solely at runtime through environment variables or Streamlit session states, never committed to git repositories or exported in audit reports.
    """)

# 4. Session State & Reset Tab
with tab_session:
    st.subheader("Danger Zone & Session Management")
    st.warning("Managing active session data and stored configurations.")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("#### Clear Current Dataset")
        st.caption("Clears uploaded data, cleaning history, and diagnosis caches while keeping your API key settings.")
        if st.button("Clear Active Dataset Session", type="secondary", use_container_width=True):
            st.session_state["df_raw"] = None
            st.session_state["df_clean"] = None
            st.session_state["active_dataset_name"] = None
            st.session_state["dataset_metadata"] = {}
            st.session_state["diagnosis_results"] = None
            st.session_state["quality_scores"] = None
            st.session_state["cleaning_history"] = []
            st.session_state["ai_chat_history"] = []
            st.session_state["ai_summary_cache"] = None
            st.success("Dataset session cleared.")
            st.rerun()

    with col_d2:
        st.markdown("#### Factory Reset Application")
        st.caption("Resets all settings, API keys, and session variables back to initial defaults.")
        if st.button("Factory Reset Entire Application", type="primary", use_container_width=True):
            st.session_state.clear()
            init_session_state()
            st.warning("Application reset to factory defaults.")
            st.rerun()

render_footer()
