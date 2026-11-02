"""
Utility functions for Intelligent Dataset Diagnosis and Quality Analysis System.
Provides UI helpers, session state management, styling, and general utilities.
"""

import os
from typing import Any, Dict, Optional
import streamlit as st


def load_css(file_path: Optional[str] = None) -> None:
    """Inject custom CSS styles into the Streamlit app."""
    if file_path is None:
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "styles.css")
    
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def init_session_state() -> None:
    """Initialize session state variables required across pages."""
    defaults: Dict[str, Any] = {
        "df_raw": None,
        "df_clean": None,
        "active_dataset_name": None,
        "dataset_metadata": {},
        "diagnosis_results": None,
        "quality_scores": None,
        "cleaning_history": [],
        "ai_chat_history": [],
        "ai_summary_cache": None,
        "settings": {
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "openai_base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "scoring_weights": {
                "completeness": 0.25,
                "uniqueness": 0.25,
                "validity": 0.25,
                "consistency": 0.25
            },
            "iqr_threshold": 1.5,
            "max_upload_size_mb": 50,
            "theme": "Dark Charcoal"
        }
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def format_bytes(size_bytes: int) -> str:
    """Format bytes to human-readable string (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def render_hero(title: str, subtitle: str, badge: Optional[str] = None) -> None:
    """Render a premium hero banner."""
    badge_html = f'<div class="hero-badge">{badge}</div>' if badge else ""
    html = f"""
    <div class="hero-container">
        {badge_html}
        <h1 class="hero-title">{title}</h1>
        <p class="hero-subtitle">{subtitle}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(title: str, value: Any, footer: str = "", badge_class: str = "") -> str:
    """Generate HTML for a metric card."""
    badge_html = f'<span class="{badge_class}">{footer}</span>' if badge_class else f'<span class="metric-footer">{footer}</span>'
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {badge_html}
    </div>
    """


def get_score_badge(score: float) -> str:
    """Determine badge styling class based on score."""
    if score >= 90:
        return "badge-score-excellent"
    elif score >= 75:
        return "badge-score-good"
    elif score >= 50:
        return "badge-score-fair"
    else:
        return "badge-score-poor"


def get_score_label(score: float) -> str:
    """Determine descriptive score label."""
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 50:
        return "Needs Attention"
    else:
        return "Critical Issues"


def render_footer() -> None:
    """Render footer signature."""
    st.markdown(
        """
        <div class="app-footer">
            <strong>Intelligent Dataset Diagnosis and Quality Analysis System</strong> • Final Year B.Tech AI & ML Project<br>
            Powered by Generative AI, Pandas, Plotly & Advanced Statistical Profiling
        </div>
        """,
        unsafe_allow_html=True
    )
