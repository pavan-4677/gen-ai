"""
Page 1: Home / Overview
Premium SaaS landing page showcasing platform capabilities, processing architecture,
feature highlights, and instant demonstration benchmarks.
"""

import os
import streamlit as st
import pandas as pd

from src.utils import init_session_state, load_css, render_hero, render_footer, render_sidebar
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
render_sidebar(current_step=1)

render_hero(
    title="Intelligent Dataset Diagnosis & Quality Analysis",
    subtitle="Enterprise AI platform for automated statistical profiling, IQR anomaly detection, explainable data quality scoring, and context-grounded Generative AI remediation.",
    badge="Enterprise AI SaaS • B.Tech Capstone Project"
)

# Active Session Notification or Quick Launch CTAs
if st.session_state.get("df_raw") is not None:
    st.markdown(
        f"""
        <div class="success-box">
            <strong>Active Session Loaded:</strong> Currently inspecting dataset 
            <code>{st.session_state.get('active_dataset_name')}</code> 
            ({len(st.session_state.get('df_raw')):,} records, {len(st.session_state.get('df_raw').columns)} attributes). 
            Navigate using the sidebar to explore <strong>Diagnosis</strong>, <strong>Dashboard</strong>, <strong>Gen AI Assistant</strong>, or the <strong>Cleaning Studio</strong>.
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div class="info-box">
            <strong>Quick Start:</strong> Upload your proprietary CSV or Excel spreadsheet, or load one of our pre-configured benchmarks containing real-world dirty data scenarios (missing values, extreme IQR outliers, duplicate entries, and inconsistent casing).
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col_demo1, col_demo2 = st.columns(2)
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")

    with col_demo1:
        if st.button("🚀 Load Customer Demographics Benchmark (Instant Demo)", use_container_width=True, type="primary"):
            p = os.path.join(sample_dir, "sample_dataset.csv")
            loader = DataLoader()
            df, meta = loader.load_dataset(p)
            st.session_state["df_raw"] = df
            st.session_state["df_clean"] = df.copy()
            st.session_state["active_dataset_name"] = "sample_dataset.csv"
            st.session_state["dataset_metadata"] = meta
            analyzer = QualityAnalyzer()
            st.session_state["diagnosis_results"] = analyzer.analyze(df, dataset_name="sample_dataset.csv")
            st.session_state["quality_scores"] = st.session_state["diagnosis_results"]["quality_scores"]
            st.session_state["cleaning_history"] = []
            st.session_state["ai_summary_cache"] = None
            st.rerun()

    with col_demo2:
        if st.button("📊 Load Telecom Churn Benchmark (Instant Demo)", use_container_width=True):
            p = os.path.join(sample_dir, "customer_churn_dirty.csv")
            loader = DataLoader()
            df, meta = loader.load_dataset(p)
            st.session_state["df_raw"] = df
            st.session_state["df_clean"] = df.copy()
            st.session_state["active_dataset_name"] = "customer_churn_dirty.csv"
            st.session_state["dataset_metadata"] = meta
            analyzer = QualityAnalyzer()
            st.session_state["diagnosis_results"] = analyzer.analyze(df, dataset_name="customer_churn_dirty.csv")
            st.session_state["quality_scores"] = st.session_state["diagnosis_results"]["quality_scores"]
            st.session_state["cleaning_history"] = []
            st.session_state["ai_summary_cache"] = None
            st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# 3-Column SaaS Feature Showcase Grid
f_col1, f_col2, f_col3 = st.columns(3)

with f_col1:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon-wrapper">🔍</div>
            <h4>Automated 10-Point Diagnosis</h4>
            <p>
                Robust detection of missing value sparsity, exact and subset duplicate records, 
                Tukey IQR (1.5x) statistical anomalies, sentinel tokens (<code>?</code>, <code>null</code>, <code>unknown</code>), 
                and schema mismatches like dirty currencies or dates stored as strings.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with f_col2:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon-wrapper">⚖️</div>
            <h4>Explainable 4D Quality Scoring</h4>
            <p>
                Transparent, deterministic scoring across four essential dimensions: 
                <strong>Completeness</strong>, <strong>Uniqueness</strong>, <strong>Validity</strong>, and <strong>Consistency</strong>. 
                Includes configurable formula weights, grade classifications (A to D), and penalty deductions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with f_col3:
    st.markdown(
        """
        <div class="feature-card">
            <div class="feature-icon-wrapper">🤖</div>
            <h4>Gen AI Remediation & Q&A</h4>
            <p>
                Context-grounded LLM reasoning that explains root causes, evaluates ML model readiness, 
                and guides non-destructive data remediation in the interactive studio. Includes full offline heuristic 
                fallback with prompt-injection defense.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br><hr style='border-color:rgba(56,189,248,0.12);'><br>", unsafe_allow_html=True)

# End-to-End Processing Architecture
st.subheader("System Architecture & Processing Pipeline")
st.markdown(
    """
    <div style="background:linear-gradient(145deg, rgba(15,23,42,0.95), rgba(30,41,59,0.4));padding:1.5rem 1.8rem;border:1px solid rgba(56,189,248,0.15);border-radius:12px;margin-bottom:1.8rem;">
        <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));gap:1rem;text-align:center;">
            <div style="padding:1rem;background:rgba(11,17,32,0.7);border-radius:8px;border:1px solid rgba(56,189,248,0.15);">
                <div style="font-size:1.4rem;">📥</div>
                <div style="font-size:0.85rem;font-weight:700;color:#F8FAFC;margin-top:0.3rem;">1. Ingest & Validate</div>
                <div style="font-size:0.75rem;color:#94A3B8;margin-top:0.2rem;">CSV, TSV, XLSX format & encoding sniffing</div>
            </div>
            <div style="padding:1rem;background:rgba(11,17,32,0.7);border-radius:8px;border:1px solid rgba(56,189,248,0.15);">
                <div style="font-size:1.4rem;">🔬</div>
                <div style="font-size:0.85rem;font-weight:700;color:#F8FAFC;margin-top:0.3rem;">2. 10-Point Diagnosis</div>
                <div style="font-size:0.75rem;color:#94A3B8;margin-top:0.2rem;">IQR outliers, nulls, duplicates & types</div>
            </div>
            <div style="padding:1rem;background:rgba(11,17,32,0.7);border-radius:8px;border:1px solid rgba(56,189,248,0.15);">
                <div style="font-size:1.4rem;">📊</div>
                <div style="font-size:0.85rem;font-weight:700;color:#F8FAFC;margin-top:0.3rem;">3. Visual Analytics</div>
                <div style="font-size:0.75rem;color:#94A3B8;margin-top:0.2rem;">Plotly gauges, distributions & heatmaps</div>
            </div>
            <div style="padding:1rem;background:rgba(11,17,32,0.7);border-radius:8px;border:1px solid rgba(56,189,248,0.15);">
                <div style="font-size:1.4rem;">🤖</div>
                <div style="font-size:0.85rem;font-weight:700;color:#F8FAFC;margin-top:0.3rem;">4. Generative AI</div>
                <div style="font-size:0.75rem;color:#94A3B8;margin-top:0.2rem;">Contextual diagnosis & chat Q&A</div>
            </div>
            <div style="padding:1rem;background:rgba(11,17,32,0.7);border-radius:8px;border:1px solid rgba(56,189,248,0.15);">
                <div style="font-size:1.4rem;">🧹</div>
                <div style="font-size:0.85rem;font-weight:700;color:#F8FAFC;margin-top:0.3rem;">5. Cleaning Studio</div>
                <div style="font-size:0.75rem;color:#94A3B8;margin-top:0.2rem;">Audited transformations with undo</div>
            </div>
            <div style="padding:1rem;background:rgba(11,17,32,0.7);border-radius:8px;border:1px solid rgba(56,189,248,0.15);">
                <div style="font-size:1.4rem;">📑</div>
                <div style="font-size:0.85rem;font-weight:700;color:#F8FAFC;margin-top:0.3rem;">6. Audit Reports</div>
                <div style="font-size:0.75rem;color:#94A3B8;margin-top:0.2rem;">PDF, JSON & cleaned CSV/XLS exports</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

render_footer()
