"""
Page 4: Data Quality Dashboard
Interactive visual analytics using Plotly: Quality Score Gauge, Missing Values Bar Chart,
IQR Outlier Frequency, Type Distribution, Correlation Heatmap, and Distribution Histograms.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.utils import init_session_state, load_css, render_hero, render_footer, render_sidebar, apply_plotly_theme
from src.quality_analyzer import QualityAnalyzer

st.set_page_config(
    page_title="Quality Dashboard | Dataset Quality Analysis",
    page_icon="📊",
    layout="wide"
)

init_session_state()
load_css()
render_sidebar(current_step=3)

render_hero(
    title="Data Quality Analytics Dashboard",
    subtitle="Interactive visual diagnostics, statistical distribution charts, and transparent 4D scoring.",
    badge="Step 3 • Visual Analytics & KPIs"
)

if st.session_state.get("df_raw") is None:
    st.markdown(
        """
        <div class="warning-box">
            <strong>No Dataset in Memory:</strong> Please upload a dataset to view the interactive quality dashboard.
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
    analyzer = QualityAnalyzer()
    diagnosis = analyzer.analyze(df, dataset_name=st.session_state.get("active_dataset_name", "Dataset"))
    st.session_state["diagnosis_results"] = diagnosis
    st.session_state["quality_scores"] = diagnosis["quality_scores"]

scores = diagnosis["quality_scores"]
overall_score = scores.get("overall_score", 0.0)
grade = scores.get("grade", "N/A")
status = scores.get("status", "N/A")
dims = scores.get("dimensions", {})

# Top Section: Overall Score Gauge & 4 Dimensions
col_gauge, col_dims = st.columns([1.1, 1.9])

with col_gauge:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=overall_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Overall Quality Score<br><span style='font-size:0.85em;color:#38BDF8;'>Grade: {grade}</span>", 'font': {'size': 17, 'color': '#F8FAFC', 'family': 'Plus Jakarta Sans'}},
        number={'suffix': "/100", 'font': {'color': '#38BDF8', 'size': 34, 'family': 'Plus Jakarta Sans'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': "#38BDF8", 'thickness': 0.8},
            'bgcolor': "#0F172A",
            'borderwidth': 1,
            'bordercolor': "rgba(56, 189, 248, 0.2)",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(244, 63, 94, 0.2)'},
                {'range': [50, 75], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [75, 90], 'color': 'rgba(56, 189, 248, 0.2)'},
                {'range': [90, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#10B981", 'width': 3},
                'thickness': 0.8,
                'value': 90
            }
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_dims:
    st.markdown("#### Four Quality Dimensions Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        comp_score = dims.get('completeness', {}).get('score', 0)
        uniq_score = dims.get('uniqueness', {}).get('score', 0)
        st.metric("Completeness Score", f"{comp_score}%", f"{dims.get('completeness', {}).get('missing_cells', 0)} null cells", delta_color="normal" if comp_score >= 90 else "inverse")
        st.metric("Uniqueness Score", f"{uniq_score}%", f"{dims.get('uniqueness', {}).get('duplicate_rows', 0)} duplicates", delta_color="normal" if uniq_score == 100 else "inverse")
    with c2:
        val_score = dims.get('validity', {}).get('score', 0)
        cons_score = dims.get('consistency', {}).get('score', 0)
        st.metric("Validity Score", f"{val_score}%", f"{dims.get('validity', {}).get('invalid_values_count', 0)} invalid cells", delta_color="normal" if val_score >= 90 else "inverse")
        st.metric("Consistency Score", f"{cons_score}%", f"{dims.get('consistency', {}).get('inconsistency_points', 0)} penalty points", delta_color="normal" if cons_score >= 90 else "inverse")

st.markdown("<br><hr style='border-color:rgba(56,189,248,0.12);'><br>", unsafe_allow_html=True)

# Chart Row 1: Missing Values & Outlier Frequencies
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    missing_data = diagnosis["missing_analysis"].get("column_breakdown", [])
    if missing_data:
        m_df = pd.DataFrame(missing_data)
        fig_miss = px.bar(
            m_df,
            x="column",
            y="missing_percentage",
            color="severity",
            color_discrete_map={"High": "#F43F5E", "Medium": "#F59E0B", "Low": "#38BDF8"},
            labels={"missing_percentage": "Null Ratio (%)", "column": "Column"},
            title="Missing Values Sparsity by Column"
        )
        apply_plotly_theme(fig_miss, height=330, title="Missing Value Sparsity by Column")
        st.plotly_chart(fig_miss, use_container_width=True)
    else:
        st.markdown(
            """
            <div class="success-box">
                <strong>100% Completeness:</strong> Zero missing values detected across attributes.
            </div>
            """,
            unsafe_allow_html=True
        )

with row1_col2:
    outlier_reports = diagnosis["outlier_analysis"].get("column_reports", {})
    out_data = [
        {"column": col, "outlier_count": rep["outlier_count"], "outlier_pct": rep["outlier_pct"]}
        for col, rep in outlier_reports.items() if rep["outlier_count"] > 0
    ]
    if out_data:
        o_df = pd.DataFrame(out_data)
        fig_out = px.bar(
            o_df,
            x="column",
            y="outlier_count",
            hover_data=["outlier_pct"],
            color_discrete_sequence=["#8B5CF6"],
            labels={"outlier_count": "Outliers Detected", "column": "Column"},
            title="Statistical Outliers (Tukey IQR 1.5x Method)"
        )
        apply_plotly_theme(fig_out, height=330, title="Statistical Outliers Count per Feature (IQR 1.5x)")
        st.plotly_chart(fig_out, use_container_width=True)
    else:
        st.markdown(
            """
            <div class="success-box">
                <strong>No Extreme Outliers:</strong> No numerical values fall outside standard Tukey IQR thresholds.
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)

# Chart Row 2: Data Type Composition & Correlation Heatmap
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    prof_overview = diagnosis["profile"]["overview"]
    type_counts = {
        "Numerical": prof_overview.get("numerical_columns_count", 0),
        "Categorical/Text": prof_overview.get("categorical_columns_count", 0),
        "Datetime": prof_overview.get("datetime_columns_count", 0),
        "Boolean": prof_overview.get("boolean_columns_count", 0)
    }
    type_df = pd.DataFrame([{"Type": k, "Count": v} for k, v in type_counts.items() if v > 0])
    if not type_df.empty:
        fig_pie = px.pie(
            type_df,
            names="Type",
            values="Count",
            hole=0.45,
            color_discrete_sequence=["#38BDF8", "#6366F1", "#10B981", "#F59E0B"],
            title="Dataset Schema Composition"
        )
        apply_plotly_theme(fig_pie, height=330, title="Dataset Schema Composition")
        st.plotly_chart(fig_pie, use_container_width=True)

with row2_col2:
    corr_matrix = diagnosis["profile"].get("correlation_matrix", {})
    if corr_matrix and len(corr_matrix) >= 2:
        corr_df = pd.DataFrame(corr_matrix)
        fig_corr = px.imshow(
            corr_df,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Pearson Correlation Heatmap"
        )
        apply_plotly_theme(fig_corr, height=330, title="Pearson Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.markdown(
            """
            <div class="info-box">
                <strong>Correlation Notice:</strong> Minimum of 2 numerical attributes required to construct a Pearson correlation matrix.
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("<br><hr style='border-color:rgba(56,189,248,0.12);'><br>", unsafe_allow_html=True)

# Chart Row 3: Feature-Level Distribution Explorers
st.subheader("🔍 Attribute Level Exploration")
tab_num_exp, tab_cat_exp = st.tabs(["Numerical Distribution Explorer", "Categorical Frequency Explorer"])

with tab_num_exp:
    num_cols = prof_overview.get("numerical_column_names", [])
    if num_cols:
        col_chosen = st.selectbox("Select Numerical Attribute to Inspect Distribution:", num_cols)
        fig_dist = px.histogram(
            df,
            x=col_chosen,
            marginal="box",
            nbins=30,
            color_discrete_sequence=["#38BDF8"],
            title=f"Distribution & Marginal Box Plot for '{col_chosen}'"
        )
        apply_plotly_theme(fig_dist, height=360, title=f"Distribution & Marginal Box Plot for '{col_chosen}'")
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("No numerical attributes present in dataset.")

with tab_cat_exp:
    cat_cols = prof_overview.get("categorical_column_names", [])
    if cat_cols:
        cat_chosen = st.selectbox("Select Categorical Attribute to Inspect Frequencies:", cat_cols)
        top_counts = df[cat_chosen].dropna().astype(str).value_counts().head(10).reset_index()
        top_counts.columns = [cat_chosen, "Count"]

        fig_cat = px.bar(
            top_counts,
            x=cat_chosen,
            y="Count",
            color_discrete_sequence=["#6366F1"],
            title=f"Top 10 Most Frequent Categories for '{cat_chosen}'"
        )
        apply_plotly_theme(fig_cat, height=360, title=f"Top 10 Categories in '{cat_chosen}'")
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.info("No categorical attributes present in dataset.")

render_footer()
