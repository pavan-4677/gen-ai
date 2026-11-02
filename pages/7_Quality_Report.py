"""
Page 7: AI Data Quality Report
Comprehensive audit report synthesis with Before-and-After score comparison,
and instant downloads for PDF, JSON, Cleaned CSV, and Cleaned Excel.
"""

from datetime import datetime
import streamlit as st
import pandas as pd

from src.utils import init_session_state, load_css, render_hero, render_footer, get_score_badge
from src.report_generator import ReportGenerator
from src.quality_analyzer import QualityAnalyzer
from src.ai_service import AIService

st.set_page_config(
    page_title="Quality Report | Dataset Quality Analysis",
    page_icon="📑",
    layout="wide"
)

init_session_state()
load_css()

render_hero(
    title="Comprehensive Data Quality & Audit Report",
    subtitle="Executive data health assessment, before-and-after cleaning comparisons, and multi-format compliance exports.",
    badge="Step 6 • Audit & Reporting"
)

if st.session_state.get("df_raw") is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset to generate a quality report.")
    if st.button("📂 Go to Dataset Upload"):
        st.switch_page("pages/2_Upload_Dataset.py")
    st.stop()

df_raw = st.session_state["df_raw"]
df_clean = st.session_state.get("df_clean", df_raw)
diagnosis_raw = st.session_state.get("diagnosis_results")

# Ensure raw diagnosis exists
if diagnosis_raw is None:
    analyzer = QualityAnalyzer()
    diagnosis_raw = analyzer.analyze(df_raw, dataset_name=st.session_state.get("active_dataset_name", "Dataset"))
    st.session_state["diagnosis_results"] = diagnosis_raw
    st.session_state["quality_scores"] = diagnosis_raw["quality_scores"]

# Calculate clean diagnosis to provide Before vs After comparison
with st.spinner("Compiling Before-and-After quality comparison..."):
    analyzer_clean = QualityAnalyzer()
    diagnosis_clean = analyzer_clean.analyze(df_clean, dataset_name=f"Cleaned_{st.session_state.get('active_dataset_name', 'Dataset')}")

# AI explanation
ai_text = st.session_state.get("ai_summary_cache")
if not ai_text:
    settings = st.session_state.get("settings", {})
    ai_service = AIService(
        api_key=settings.get("openai_api_key"),
        model=settings.get("openai_model", "gpt-4o-mini"),
        base_url=settings.get("openai_base_url", "https://api.openai.com/v1")
    )
    ai_text = ai_service.generate_comprehensive_explanation(diagnosis_raw)
    st.session_state["ai_summary_cache"] = ai_text

report_gen = ReportGenerator()
dataset_name = st.session_state.get("active_dataset_name", "dataset")
base_name = dataset_name.rsplit(".", 1)[0]

# Download Action Buttons Card
st.subheader("📥 Export & Download Audit Artifacts")
down_col1, down_col2, down_col3, down_col4 = st.columns(4)

with down_col1:
    pdf_bytes = report_gen.generate_pdf_report(
        diagnosis=diagnosis_clean,
        cleaning_history=st.session_state.get("cleaning_history", []),
        ai_explanation=ai_text
    )
    st.download_button(
        label="📄 Download PDF Audit Report",
        data=pdf_bytes,
        file_name=f"{base_name}_Quality_Report.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True
    )

with down_col2:
    json_str = report_gen.generate_json_report(
        diagnosis=diagnosis_clean,
        cleaning_history=st.session_state.get("cleaning_history", []),
        ai_explanation=ai_text
    )
    st.download_button(
        label="📋 Download JSON Audit Report",
        data=json_str,
        file_name=f"{base_name}_Quality_Audit.json",
        mime="application/json",
        use_container_width=True
    )

with down_col3:
    csv_bytes = report_gen.export_csv(df_clean)
    st.download_button(
        label="📊 Download Cleaned CSV",
        data=csv_bytes,
        file_name=f"{base_name}_cleaned.csv",
        mime="text/csv",
        use_container_width=True
    )

with down_col4:
    xlsx_bytes = report_gen.export_excel(df_clean)
    st.download_button(
        label="📑 Download Cleaned Excel (.xlsx)",
        data=xlsx_bytes,
        file_name=f"{base_name}_cleaned.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

st.markdown("---")

# Before vs After Comparison Section
st.subheader("⚖️ Before vs After Data Quality Impact")

raw_score = diagnosis_raw["quality_scores"]["overall_score"]
clean_score = diagnosis_clean["quality_scores"]["overall_score"]
score_diff = round(clean_score - raw_score, 2)

comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
with comp_col1:
    st.metric("Raw Dataset Quality", f"{raw_score}/100", diagnosis_raw["quality_scores"]["grade"])
with comp_col2:
    st.metric(
        "Cleaned Dataset Quality",
        f"{clean_score}/100",
        f"{'+' if score_diff >= 0 else ''}{score_diff} pts",
        delta_color="normal" if score_diff >= 0 else "inverse"
    )
with comp_col3:
    st.metric(
        "Total Missing Cells",
        f"{diagnosis_clean['profile']['overview']['total_missing_cells']:,}",
        f"{diagnosis_clean['profile']['overview']['total_missing_cells'] - diagnosis_raw['profile']['overview']['total_missing_cells']} cells",
        delta_color="inverse"
    )
with comp_col4:
    st.metric(
        "Duplicate Rows",
        diagnosis_clean["duplicate_analysis"]["duplicate_rows_count"],
        f"{diagnosis_clean['duplicate_analysis']['duplicate_rows_count'] - diagnosis_raw['duplicate_analysis']['duplicate_rows_count']} rows",
        delta_color="inverse"
    )

# Dimension Comparison Table
st.markdown("#### Dimensional Comparison")
raw_dims = diagnosis_raw["quality_scores"]["dimensions"]
clean_dims = diagnosis_clean["quality_scores"]["dimensions"]

comparison_df = pd.DataFrame([
    {
        "Dimension": "Completeness",
        "Raw Score": f"{raw_dims.get('completeness', {}).get('score', 0)}%",
        "Cleaned Score": f"{clean_dims.get('completeness', {}).get('score', 0)}%",
        "Delta": f"{clean_dims.get('completeness', {}).get('score', 0) - raw_dims.get('completeness', {}).get('score', 0):+.2f}%"
    },
    {
        "Dimension": "Uniqueness",
        "Raw Score": f"{raw_dims.get('uniqueness', {}).get('score', 0)}%",
        "Cleaned Score": f"{clean_dims.get('uniqueness', {}).get('score', 0)}%",
        "Delta": f"{clean_dims.get('uniqueness', {}).get('score', 0) - raw_dims.get('uniqueness', {}).get('score', 0):+.2f}%"
    },
    {
        "Dimension": "Validity",
        "Raw Score": f"{raw_dims.get('validity', {}).get('score', 0)}%",
        "Cleaned Score": f"{clean_dims.get('validity', {}).get('score', 0)}%",
        "Delta": f"{clean_dims.get('validity', {}).get('score', 0) - raw_dims.get('validity', {}).get('score', 0):+.2f}%"
    },
    {
        "Dimension": "Consistency",
        "Raw Score": f"{raw_dims.get('consistency', {}).get('score', 0)}%",
        "Cleaned Score": f"{clean_dims.get('consistency', {}).get('score', 0)}%",
        "Delta": f"{clean_dims.get('consistency', {}).get('score', 0) - raw_dims.get('consistency', {}).get('score', 0):+.2f}%"
    }
])
st.dataframe(comparison_df, use_container_width=True)

st.markdown("---")

# On-screen Report Summary
st.subheader("📑 Executive Report Summary")
tab_diag, tab_ai, tab_audit = st.tabs(["🔬 Diagnostic Audit Details", "🤖 Gen AI Strategic Insights", "🛠️ Applied Remediation Steps"])

with tab_diag:
    st.markdown("#### Structural Summary")
    prof = diagnosis_clean["profile"]["overview"]
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown(f"""
        - **Dataset Name**: `{dataset_name}`
        - **Total Records**: {prof['total_rows']:,}
        - **Total Features**: {prof['total_columns']}
        - **Numerical Features**: {prof['numerical_columns_count']}
        """)
    with col_d2:
        st.markdown(f"""
        - **Categorical Features**: {prof['categorical_columns_count']}
        - **Total Missing Cells**: {prof['total_missing_cells']} ({prof['missing_percentage']}%)
        - **Duplicate Rows**: {diagnosis_clean['duplicate_analysis']['duplicate_rows_count']}
        - **Outlier Features**: {len(diagnosis_clean['outlier_analysis']['columns_with_outliers'])}
        """)

with tab_ai:
    st.markdown(ai_text)

with tab_audit:
    history = st.session_state.get("cleaning_history", [])
    if history:
        for i, h in enumerate(history, 1):
            st.markdown(f"**Step {i} ({h['timestamp']}):** `{h['action']}` — {h['details']}")
    else:
        st.info("No cleaning operations were performed prior to generating this report.")

render_footer()
