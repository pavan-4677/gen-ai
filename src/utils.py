"""
Utility functions for Intelligent Dataset Diagnosis and Quality Analysis System.
Provides UI helpers, session state management, sidebar rendering, styling, and general utilities.
"""

import base64
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
    """Render a premium SaaS hero banner."""
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
        return "Excellent (A)"
    elif score >= 75:
        return "Good (B)"
    elif score >= 50:
        return "Needs Attention (C)"
    else:
        return "Critical Issues (D)"


def apply_plotly_theme(fig, height: int = 340, title: str = ""):
    """Apply consistent ultra-premium dark theme to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0F172A",
        height=height,
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#F8FAFC", size=11),
        title=dict(text=title, font=dict(size=14, color="#F8FAFC", family="Plus Jakarta Sans")),
        margin=dict(l=25, r=25, t=45 if title else 25, b=25),
        xaxis=dict(
            gridcolor="rgba(56, 189, 248, 0.08)",
            linecolor="rgba(56, 189, 248, 0.2)",
            tickfont=dict(color="#94A3B8")
        ),
        yaxis=dict(
            gridcolor="rgba(56, 189, 248, 0.08)",
            linecolor="rgba(56, 189, 248, 0.2)",
            tickfont=dict(color="#94A3B8")
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.7)",
            bordercolor="rgba(56, 189, 248, 0.15)",
            font=dict(color="#E2E8F0")
        )
    )
    return fig


def render_sidebar(current_step: int = 1) -> None:
    """
    Render a unified, premium SaaS sidebar across all pages.
    Displays branding, active dataset session stats, workflow stepper, and AI connectivity.
    """
    with st.sidebar:
        # 1. Branding Header
        logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "logo.png")
        logo_img_tag = ""
        if os.path.exists(logo_path):
            try:
                with open(logo_path, "rb") as img_f:
                    b64_logo = base64.b64encode(img_f.read()).decode("utf-8")
                logo_img_tag = f'<img src="data:image/png;base64,{b64_logo}" class="sidebar-logo" alt="Logo" />'
            except Exception:
                logo_img_tag = "🛡️"

        st.markdown(
            f"""
            <div class="sidebar-brand-box">
                {logo_img_tag}
                <div class="sidebar-brand-text">
                    <h2>DiagAI Suite</h2>
                    <p>Enterprise SaaS v1.0</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 2. Active Session Card
        df_raw = st.session_state.get("df_raw")
        df_clean = st.session_state.get("df_clean")
        scores = st.session_state.get("quality_scores")
        dataset_name = st.session_state.get("active_dataset_name")

        if df_raw is not None and scores is not None:
            overall = scores.get("overall_score", 0.0)
            badge_class = get_score_badge(overall)
            grade = scores.get("grade", "N/A")
            is_cleaned = len(st.session_state.get("cleaning_history", [])) > 0
            status_tag = '<span style="color:#10B981;font-weight:700;">● Cleaned</span>' if is_cleaned else '<span style="color:#38BDF8;font-weight:700;">● Raw</span>'

            st.markdown(
                f"""
                <div class="sidebar-session-card">
                    <div class="sidebar-session-title">
                        <span>ACTIVE DATASET</span>
                        {status_tag}
                    </div>
                    <div class="sidebar-session-filename" title="{dataset_name}">📁 {dataset_name}</div>
                    <div class="sidebar-meta-row">
                        <span>Records: <b>{len(df_clean if is_cleaned else df_raw):,}</b></span>
                        <span>Columns: <b>{len(df_clean.columns if is_cleaned else df_raw.columns)}</b></span>
                    </div>
                    <div style="margin-top:0.65rem;display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-size:0.75rem;color:#94A3B8;">Health Score:</span>
                        <span class="{badge_class}">{overall}/100</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="sidebar-session-card">
                    <div class="sidebar-session-title">
                        <span>DATASET STATUS</span>
                        <span style="color:#64748B;">● Empty</span>
                    </div>
                    <div style="font-size:0.8rem;color:#94A3B8;margin-top:0.3rem;">
                        No dataset loaded in memory. Upload a file or load benchmark data to begin.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # 3. Processing Pipeline Stepper
        steps = [
            (1, "Ingest & Validate"),
            (2, "10-Point Diagnosis"),
            (3, "Visual Analytics"),
            (4, "Generative AI"),
            (5, "Cleaning Studio"),
            (6, "Audit Report")
        ]
        
        stepper_html = '<div class="workflow-stepper"><div class="stepper-header">Analysis Pipeline</div>'
        for step_num, step_name in steps:
            is_active = (step_num == current_step)
            active_class = "active" if is_active else ""
            stepper_html += f"""
            <div class="stepper-step {active_class}">
                <div class="stepper-dot"></div>
                <span>{step_num}. {step_name}</span>
            </div>
            """
        stepper_html += '</div>'
        st.markdown(stepper_html, unsafe_allow_html=True)

        # 4. Engine Connection Status
        api_key = st.session_state.get("settings", {}).get("openai_api_key", "")
        model_name = st.session_state.get("settings", {}).get("openai_model", "gpt-4o-mini")
        if api_key and not api_key.startswith("your_"):
            st.markdown(f'<div style="font-size:0.75rem;color:#10B981;padding:0.4rem 0.6rem;background:rgba(16,185,129,0.1);border-radius:6px;border:1px solid rgba(16,185,129,0.25);">🟢 <b>Live LLM:</b> {model_name}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:0.75rem;color:#38BDF8;padding:0.4rem 0.6rem;background:rgba(56,189,248,0.1);border-radius:6px;border:1px solid rgba(56,189,248,0.25);">🔵 <b>Engine:</b> Deterministic Heuristic</div>', unsafe_allow_html=True)


def render_footer() -> None:
    """Render footer signature."""
    st.markdown(
        """
        <div class="app-footer">
            <strong>Intelligent Dataset Diagnosis and Quality Analysis System</strong> • Enterprise AI SaaS Edition<br>
            Powered by Streamlit, Scikit-learn, Plotly, Pandas & Generative AI Reasoning
        </div>
        """,
        unsafe_allow_html=True
    )
