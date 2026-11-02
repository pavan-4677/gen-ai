"""
Page 1: Home / Overview
Landing page explaining system architecture, features, workflow, and quick-start actions.
"""

import os
import streamlit as st
import pandas as pd

from src.utils import init_session_state, load_css, render_hero, render_footer
from src.data_loader import DataLoader
from src.quality_analyzer import QualityAnalyzer

st.set_page_config(
    page_title="Overview | Dataset Quality Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_session_state()
load_css()

render_hero(
    title="Intelligent Dataset Diagnosis & Quality Analysis",
    subtitle="End-to-End Automated Data Profiling, IQR Anomaly Detection, Explainable Quality Scoring, and Generative AI Cleaning Strategy.",
    badge="Final Year Gen AI Project • AI & ML"
)

# Active dataset alert banner
if st.session_state.get("df_raw") is not None:
    st.success(
        f"**Active Session:** Currently analyzing dataset **`{st.session_state.get('active_dataset_name')}`** "
        f"({len(st.session_state.get('df_raw'))} rows, {len(st.session_state.get('df_raw').columns)} columns). "
        "Navigate using the sidebar to view Diagnosis, Dashboard, AI Assistant, or Cleaning Studio."
    )
else:
    col_cta1, col_cta2 = st.columns([1, 1])
    with col_cta1:
        st.info("💡 **Quick Start:** You can upload your own CSV or Excel dataset, or load a pre-built dirty test dataset below.")
    with col_cta2:
        if st.button("🚀 Load Customer Demographics Sample Dataset (Instant Demo)", use_container_width=True, type="primary"):
            sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "sample_dataset.csv")
            if os.path.exists(sample_path):
                loader = DataLoader()
                df, meta = loader.load_dataset(sample_path)
                st.session_state["df_raw"] = df
                st.session_state["df_clean"] = df.copy()
                st.session_state["active_dataset_name"] = "sample_dataset.csv"
                st.session_state["dataset_metadata"] = meta
                
                # Precompute analysis
                analyzer = QualityAnalyzer(
                    iqr_multiplier=st.session_state["settings"].get("iqr_threshold", 1.5),
                    scoring_weights=st.session_state["settings"].get("scoring_weights")
                )
                st.session_state["diagnosis_results"] = analyzer.analyze(df, dataset_name="sample_dataset.csv")
                st.session_state["quality_scores"] = st.session_state["diagnosis_results"]["quality_scores"]
                st.session_state["cleaning_history"] = []
                st.rerun()

st.markdown("---")

# Feature Highlights Grid
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🔍 Automated Diagnostics
    - **Missing Value Profiling**: Column-level null counts, percentage impact, and sparsity indexing.
    - **Duplicate Records**: Exact row duplication and subset identification.
    - **Statistical Outliers**: Interquartile Range (IQR 1.5x) detection with lower/upper thresholds.
    - **Data Type Inference**: Discovers numeric or datetime fields disguised as strings.
    """)

with col2:
    st.markdown("""
    ### 📊 Explainable Scoring
    - **Completeness**: Evaluates non-null data ratio across all fields.
    - **Uniqueness**: Quantifies degree of distinct, non-redundant rows.
    - **Validity**: Detects invalid negative values and string placeholder tokens (`?`, `NULL`).
    - **Consistency**: Evaluates formatting uniformity and casing discrepancies.
    """)

with col3:
    st.markdown("""
    ### 🤖 Gen AI Assistant & Studio
    - **Contextual Reasoning**: Grounded in actual statistical facts without hallucinations.
    - **Interactive Chat**: Ask questions regarding ML readiness, risk analysis, and cleaning steps.
    - **Non-Destructive Cleaning**: Imputation, Winsorization, and deduplication with undo audit trail.
    - **Audit Reports**: Instant export to PDF, JSON, and cleaned CSV/Excel.
    """)

st.markdown("---")

# Architecture Workflow Section
st.subheader("System Architecture & Processing Pipeline")
st.markdown("""
```
┌─────────────────────┐      ┌─────────────────────────┐      ┌──────────────────────────┐
│  Dataset Ingestion  │ ───► │ 10-Point Statistical    │ ───► │  Explainable 4D Quality  │
│  (CSV, TSV, XLSX)   │      │ Profiling & IQR Outliers│      │  Scoring Engine (0-100)  │
└─────────────────────┘      └─────────────────────────┘      └──────────────────────────┘
                                                                           │
                                                                           ▼
┌─────────────────────┐      ┌─────────────────────────┐      ┌──────────────────────────┐
│ Multi-Format Audit  │ ◄─── │ Non-Destructive Cleaning│ ◄─── │ Generative AI Diagnostic │
│ (PDF, JSON, CSV/XLS)│      │ Studio (Audit & Undo)   │      │ Assistant & Q&A Chatbot  │
└─────────────────────┘      └─────────────────────────┘      └──────────────────────────┘
```
""")

col_w1, col_w2, col_w3 = st.columns(3)
with col_w1:
    st.markdown("""
    **Phase 1: Ingestion & Validation**
    Robust decoding with multi-encoding support (UTF-8, Latin-1, CP1252), delimiter sniffing, file size bounds, and structure validation.
    """)
with col_w2:
    st.markdown("""
    **Phase 2: Comprehensive Diagnostics**
    Computes distribution skewness, quartile ranges, IQR bounds, high-cardinality candidate keys, constant columns, and invalid tokens.
    """)
with col_w3:
    st.markdown("""
    **Phase 3: Remediation & Reporting**
    Empowers users with interactive Gen AI Q&A, audited data transformations, and full audit report generation for compliance and ML workflows.
    """)

render_footer()
