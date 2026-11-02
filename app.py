"""
Intelligent Dataset Diagnosis and Quality Analysis System
Main application entrypoint.
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

from src.utils import init_session_state, load_css, render_hero, render_footer
from src.data_loader import DataLoader
from src.quality_analyzer import QualityAnalyzer

st.set_page_config(
    page_title="Dataset Quality & Gen AI Diagnosis System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_session_state()
load_css()

# Redirect immediately to Home page for consistent multi-page structure
try:
    st.switch_page("pages/1_Home.py")
except Exception:
    pass

render_hero(
    title="Intelligent Dataset Diagnosis & Quality Analysis",
    subtitle="Automated Data Hygiene, IQR Anomaly Detection, Explainable Quality Scoring, and Generative AI Remediation.",
    badge="Final Year B.Tech AI & ML Project"
)

st.markdown("""
Welcome to the **Intelligent Dataset Diagnosis and Quality Analysis System**.

Use the sidebar navigation to access all modules:
1. **Home / Overview**: System architecture, processing pipeline, and quick demo benchmarks.
2. **Upload Dataset**: Ingestion for CSV, TSV, and Excel files with encoding auto-detection.
3. **Dataset Diagnosis**: 10-point structural, statistical, and semantic integrity checks.
4. **Quality Dashboard**: Interactive Plotly analytics, score gauges, and distribution charts.
5. **Gen AI Assistant**: Context-grounded LLM insights, root cause analysis, and interactive Q&A.
6. **Data Cleaning Studio**: Non-destructive transformations with audit trail and undo support.
7. **Quality Report**: Multi-format audit exports (PDF, JSON, Cleaned CSV, Cleaned Excel).
8. **Settings**: LLM configuration, custom scoring weights, and session management.
""")

render_footer()
