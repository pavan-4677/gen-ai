"""
Page 3: Dataset Diagnosis
Performs and visualizes 10 comprehensive diagnostic checks with record-level inspection.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from src.utils import init_session_state, load_css, render_hero, render_footer, render_sidebar
from src.outlier_detector import OutlierDetector
from src.quality_analyzer import QualityAnalyzer
from src.data_loader import DataLoader

st.set_page_config(
    page_title="Dataset Diagnosis | Dataset Quality Analysis",
    page_icon="🔬",
    layout="wide"
)

init_session_state()
load_css()
render_sidebar(current_step=2)

render_hero(
    title="Comprehensive Dataset Diagnosis",
    subtitle="10-point structural, statistical, and semantic integrity analysis with record-level audit inspection.",
    badge="Step 2 • Diagnostic Deep Dive"
)

# Guard: Ensure dataset is loaded
if st.session_state.get("df_raw") is None:
    st.markdown(
        """
        <div class="warning-box">
            <strong>No Dataset in Memory:</strong> Please upload a dataset or load a benchmark dataset to run diagnostics.
        </div>
        """,
        unsafe_allow_html=True
    )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📂 Go to Dataset Upload Page", use_container_width=True):
            st.switch_page("pages/2_Upload_Dataset.py")
    with col_b:
        if st.button("🧪 Load Customer Demographics Sample (Instant)", use_container_width=True, type="primary"):
            sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "sample_dataset.csv")
            loader = DataLoader()
            df, meta = loader.load_dataset(sample_path)
            st.session_state["df_raw"] = df
            st.session_state["df_clean"] = df.copy()
            st.session_state["active_dataset_name"] = "sample_dataset.csv"
            st.session_state["dataset_metadata"] = meta
            analyzer = QualityAnalyzer()
            st.session_state["diagnosis_results"] = analyzer.analyze(df, dataset_name="sample_dataset.csv")
            st.session_state["quality_scores"] = st.session_state["diagnosis_results"]["quality_scores"]
            st.rerun()
    st.stop()

df = st.session_state["df_raw"]
diagnosis = st.session_state.get("diagnosis_results")

if diagnosis is None:
    with st.spinner("Analyzing dataset metrics..."):
        analyzer = QualityAnalyzer(
            iqr_multiplier=st.session_state["settings"].get("iqr_threshold", 1.5),
            scoring_weights=st.session_state["settings"].get("scoring_weights")
        )
        diagnosis = analyzer.analyze(df, dataset_name=st.session_state.get("active_dataset_name", "Dataset"))
        st.session_state["diagnosis_results"] = diagnosis
        st.session_state["quality_scores"] = diagnosis["quality_scores"]

profile = diagnosis["profile"]["overview"]
missing_info = diagnosis["missing_analysis"]
dup_info = diagnosis["duplicate_analysis"]
outlier_info = diagnosis["outlier_analysis"]

# Top Overview KPI Cards Grid
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric("Total Records", f"{profile['total_rows']:,}")
with col2:
    st.metric("Total Columns", profile['total_columns'])
with col3:
    st.metric("Missing Cells", f"{profile['total_missing_cells']:,}", f"{profile['missing_percentage']}%", delta_color="inverse")
with col4:
    st.metric("Duplicate Rows", f"{dup_info['duplicate_rows_count']:,}", f"{dup_info['duplicate_percentage']}%", delta_color="inverse")
with col5:
    st.metric("Outlier Rows", f"{outlier_info['outlier_rows_count']:,}", f"{outlier_info['outlier_rows_percentage']}%", delta_color="inverse")
with col6:
    st.metric("Total Issues", diagnosis['total_issues_count'])

st.markdown("<br>", unsafe_allow_html=True)

# Detailed Tabs for the 10 Checks
tab_issues, tab_missing, tab_dup, tab_outlier, tab_types, tab_cardinality, tab_distributions = st.tabs([
    f"⚠️ Detected Issues ({diagnosis['total_issues_count']})",
    f"❓ Missing Values ({missing_info['columns_with_missing_count']} cols)",
    f"👥 Duplicates ({dup_info['duplicate_rows_count']} rows)",
    f"📈 IQR Outliers ({len(outlier_info['columns_with_outliers'])} cols)",
    "🏷️ Schema & Type Inferences",
    "🔠 Cardinality & Formatting",
    "📊 Attribute Distributions"
])

# 1. Detected Issues Tab
with tab_issues:
    st.subheader("Prioritized Data Quality Remediation Queue")
    issues = diagnosis.get("detected_issues", [])
    if not issues:
        st.markdown(
            """
            <div class="success-box">
                <strong>Zero Issues Detected:</strong> Dataset passed all 10 diagnostic integrity checks with flying colors.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        for idx, issue in enumerate(issues, 1):
            severity = issue.get("severity", "Medium")
            icon = "🔴" if severity == "High" else ("🟡" if severity == "Medium" else "🔵")
            with st.expander(f"{icon} #{idx} [{issue['category']}] {issue['column']}: {issue['description']}", expanded=(idx <= 3)):
                st.markdown(f"**Category:** `{issue['category']}` | **Severity:** `{severity}` | **Target Column:** `{issue['column']}`")
                st.markdown(f"**Technical Diagnosis:** {issue['description']}")
                st.markdown(f"**Remediation Recommendation:** {issue['recommendation']}")

# 2. Missing Values Tab
with tab_missing:
    st.subheader("Missing Value & Sparsity Diagnosis")
    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        st.metric("Total Missing Cells", f"{missing_info['total_missing_cells']:,}")
        st.metric("Rows with >= 1 Null", f"{missing_info['rows_with_missing']:,} ({missing_info['rows_with_missing_pct']}%)")
    with col_m2:
        breakdown = missing_info.get("column_breakdown", [])
        if breakdown:
            st.dataframe(pd.DataFrame(breakdown), use_container_width=True)
        else:
            st.success("All columns are 100% complete (no missing values).")

    # Record-level inspection
    if missing_info['rows_with_missing'] > 0:
        with st.expander("🔍 Inspect Specific Records Containing Missing Values"):
            null_rows = df[df.isna().any(axis=1)]
            st.dataframe(null_rows, use_container_width=True)

# 3. Duplicate Records Tab
with tab_dup:
    st.subheader("Duplicate Record Detection")
    st.write(
        f"Detected **{dup_info['duplicate_rows_count']} duplicate rows** "
        f"({dup_info['duplicate_percentage']}% of total dataset). Duplicate records inflate metrics and create data leakage in train/test splits."
    )
    if dup_info['duplicate_rows_count'] > 0:
        dup_indices = dup_info.get("duplicate_indices", [])
        with st.expander("🔍 Inspect Full Duplicate Records", expanded=True):
            st.dataframe(df.loc[dup_indices], use_container_width=True)
    else:
        st.success("Zero duplicate records identified. All rows represent distinct observations.")

# 4. Outlier Analysis Tab (IQR Method)
with tab_outlier:
    st.subheader("Statistical Outlier Detection (Interquartile Range - IQR 1.5x)")
    st.markdown("""
    **Mathematical Formulation:**
    - $\\text{IQR} = Q_3 - Q_1$
    - $\\text{Lower Bound} = Q_1 - 1.5 \\times \\text{IQR}$
    - $\\text{Upper Bound} = Q_3 + 1.5 \\times \\text{IQR}$
    - Observations outside $[\\text{Lower Bound}, \\text{Upper Bound}]$ are flagged as statistical anomalies.
    """)

    outlier_cols = outlier_info.get("columns_with_outliers", [])
    if outlier_cols:
        col_select = st.selectbox("Select numerical column to inspect outliers:", outlier_cols)
        rep = outlier_info["column_reports"][col_select]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Outlier Count", rep["outlier_count"])
        c2.metric("Outlier Ratio", f"{rep['outlier_pct']}%")
        c3.metric("IQR Spread", f"{rep['iqr']:.2f}")
        c4.metric("Valid IQR Range", f"[{rep['lower_bound']:.2f}, {rep['upper_bound']:.2f}]")

        with st.expander(f"🔍 Inspect Outlier Rows in '{col_select}'", expanded=True):
            outlier_rows = OutlierDetector.get_outlier_rows(df, rep["outlier_indices"])
            st.dataframe(outlier_rows, use_container_width=True)
    else:
        st.success("No statistical outliers detected across numerical columns.")

# 5. Data Types Tab
with tab_types:
    st.subheader("Data Type Analysis & Schema Inferences")
    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        st.markdown("#### Detected Schema Types")
        type_df = pd.DataFrame([
            {"Column": col, "Data Type": dtype}
            for col, dtype in diagnosis["dtype_analysis"]["column_types"].items()
        ])
        st.dataframe(type_df, use_container_width=True)

    with col_t2:
        st.markdown("#### Type Mismatch & Coercion Warnings")
        warnings = diagnosis["dtype_analysis"].get("type_mismatch_warnings", [])
        if warnings:
            for w in warnings:
                st.warning(f"**`{w['column']}`**: Currently `{w['current_dtype']}`. Suggested: `{w['suggested_dtype']}`.\n*{w['reason']}*")
        else:
            st.success("No type mismatch anomalies found.")

# 6. Cardinality & Consistency Tab
with tab_cardinality:
    st.subheader("Cardinality & Text Consistency")
    c_card1, c_card2 = st.columns(2)
    with c_card1:
        st.markdown("#### Constant Columns (Zero Variance)")
        constants = diagnosis["cardinality_analysis"].get("constant_columns", [])
        if constants:
            for c in constants:
                st.error(f"Column **`{c['column']}`** has zero variance (constant value: `{c['constant_value']}`).")
        else:
            st.success("No constant columns detected.")

        st.markdown("#### High-Cardinality Candidate Keys")
        high_card = diagnosis["cardinality_analysis"].get("high_cardinality_columns", [])
        if high_card:
            for h in high_card:
                st.info(f"Column **`{h['column']}`** has {h['unique_values']} unique values ({h['cardinality_ratio']}%). {h['inference']}")
        else:
            st.caption("No excessive cardinality columns detected.")

    with c_card2:
        st.markdown("#### Text Consistency & Capitalization")
        text_issues = diagnosis["consistency_analysis"].get("text_issues", [])
        if text_issues:
            for t in text_issues:
                st.warning(
                    f"Column **`{t['column']}`**: {t['whitespace_count']} cells with untrimmed whitespace. "
                    f"Casing discrepancy: {t['casing_inconsistency']} ({t['unique_variants']} variants vs {t['canonical_variants']} canonical)."
                )
        else:
            st.success("All text columns follow uniform whitespace and capitalization.")

# 7. Attribute Distributions Tab
with tab_distributions:
    st.subheader("Attribute Distributions & Descriptive Statistics")
    num_cols = profile.get("numerical_column_names", [])
    if num_cols:
        st.markdown("#### Numerical Attribute Summary Table")
        num_summary = []
        for col in num_cols:
            col_prof = diagnosis["profile"]["columns"].get(col, {})
            num_summary.append({
                "Column": col,
                "Mean": col_prof.get("mean"),
                "Std Dev": col_prof.get("std"),
                "Min": col_prof.get("min"),
                "Q25": col_prof.get("q25"),
                "Median": col_prof.get("median"),
                "Q75": col_prof.get("q75"),
                "Max": col_prof.get("max"),
                "Skewness": col_prof.get("skewness"),
                "Zeros": col_prof.get("zeros_count"),
                "Negatives": col_prof.get("negatives_count")
            })
        st.dataframe(pd.DataFrame(num_summary), use_container_width=True)

render_footer()
