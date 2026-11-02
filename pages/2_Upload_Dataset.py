"""
Page 2: Dataset Upload
Provides file ingestion (CSV, TSV, XLSX, XLS), size & type validation, metadata preview,
and triggers full dataset diagnosis.
"""

import os
import streamlit as st
import pandas as pd

from src.utils import init_session_state, load_css, render_hero, render_footer, render_sidebar
from src.data_loader import DataLoader, DataLoaderError
from src.quality_analyzer import QualityAnalyzer

st.set_page_config(
    page_title="Upload Dataset | Dataset Quality Analysis",
    page_icon="📂",
    layout="wide"
)

init_session_state()
load_css()
render_sidebar(current_step=1)

render_hero(
    title="Dataset Ingestion & Validation",
    subtitle="Upload CSV or Excel spreadsheets. Automatic encoding sniffing, delimiter checking, and structural schema validation.",
    badge="Step 1 • Ingestion & Validation"
)

max_mb = st.session_state["settings"].get("max_upload_size_mb", 50)
loader = DataLoader(max_file_size_mb=max_mb)

# Upload tabs: Upload File OR Use Sample Data
tab_upload, tab_sample = st.tabs(["📤 Upload Proprietary Dataset", "🧪 Load Built-in Quality Benchmarks"])

with tab_upload:
    st.markdown(f"**Supported formats:** `.csv`, `.tsv`, `.xlsx`, `.xls` (Max size: **{max_mb} MB**)")
    uploaded_file = st.file_uploader(
        "Drag and drop your dataset here",
        type=["csv", "tsv", "xlsx", "xls"],
        help="Upload a tabular dataset for automated quality analysis and Generative AI cleaning."
    )

    if uploaded_file is not None:
        try:
            df, meta = loader.load_dataset(uploaded_file)
            st.session_state["temp_df"] = df
            st.session_state["temp_meta"] = meta
            st.session_state["temp_filename"] = uploaded_file.name
            st.markdown(
                f"""
                <div class="success-box">
                    <strong>Validation Passed:</strong> Successfully verified <code>{uploaded_file.name}</code> ({meta['formatted_size']}). 
                    Structured into <strong>{meta['total_rows']:,} rows</strong> and <strong>{meta['total_columns']} columns</strong>.
                </div>
                """,
                unsafe_allow_html=True
            )
        except DataLoaderError as e:
            st.markdown(f'<div class="warning-box"><strong>Validation Error:</strong> {str(e)}</div>', unsafe_allow_html=True)
            st.session_state["temp_df"] = None
        except Exception as e:
            st.error(f"Unexpected error while reading dataset: {str(e)}")
            st.session_state["temp_df"] = None

with tab_sample:
    st.markdown("Select a benchmark dataset with known real-world data quality issues:")
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon-wrapper">👤</div>
                <h4>Customer Demographics Benchmark</h4>
                <p>Features 30 records containing missing values in Age/Income, duplicate customer rows, mixed text casing ('Male', 'female', 'MALE'), extreme income outliers ($980,000), and sentinel '?' tokens.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Load 'sample_dataset.csv' into Buffer", use_container_width=True, type="primary"):
            p = os.path.join(sample_dir, "sample_dataset.csv")
            df, meta = loader.load_dataset(p)
            st.session_state["temp_df"] = df
            st.session_state["temp_meta"] = meta
            st.session_state["temp_filename"] = "sample_dataset.csv"
            st.success("Loaded 'sample_dataset.csv' into preview buffer.")

    with col_s2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon-wrapper">📉</div>
                <h4>Telecom Churn Benchmark</h4>
                <p>Features 21 records with missing tenure/charges, excessive outliers, '?' tokens, unformatted string currencies, and duplicate subscriber entries.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Load 'customer_churn_dirty.csv' into Buffer", use_container_width=True):
            p = os.path.join(sample_dir, "customer_churn_dirty.csv")
            df, meta = loader.load_dataset(p)
            st.session_state["temp_df"] = df
            st.session_state["temp_meta"] = meta
            st.session_state["temp_filename"] = "customer_churn_dirty.csv"
            st.success("Loaded 'customer_churn_dirty.csv' into preview buffer.")

# Preview Section if a file is ready
target_df = st.session_state.get("temp_df")
target_meta = st.session_state.get("temp_meta")
target_name = st.session_state.get("temp_filename")

if target_df is not None and target_meta is not None:
    st.markdown("<br><hr style='border-color:rgba(56,189,248,0.12);'><br>", unsafe_allow_html=True)
    st.subheader(f"📋 Dataset Ingestion Preview: {target_name}")

    # Metadata Metrics Card Row
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.metric("Total Records", f"{target_meta['total_rows']:,}")
    with col_m2:
        st.metric("Total Columns", target_meta['total_columns'])
    with col_m3:
        st.metric("File Size", target_meta['formatted_size'])
    with col_m4:
        st.metric("Memory Footprint", f"{target_meta['memory_usage_mb']} MB")
    with col_m5:
        st.metric("Numerical Features", len(target_meta['numerical_columns']))

    # Tabular preview
    st.markdown("#### Sample Records (First 10 Rows)")
    st.dataframe(target_df.head(10), use_container_width=True)

    # Column inspection table
    with st.expander("🔍 View Column Schema & Sparsity Index", expanded=False):
        col_info = []
        for c in target_df.columns:
            null_count = int(target_df[c].isna().sum())
            null_pct = (null_count / len(target_df)) * 100 if len(target_df) > 0 else 0.0
            col_info.append({
                "Column Name": str(c),
                "Detected Type": str(target_df[c].dtype),
                "Non-Null Count": int(target_df[c].notna().sum()),
                "Null Count": null_count,
                "Null %": f"{null_pct:.1f}%",
                "Unique Values": int(target_df[c].nunique(dropna=True))
            })
        st.dataframe(pd.DataFrame(col_info), use_container_width=True)

    # Run Analysis Button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Execute Full Dataset Diagnosis & Quality Scoring", type="primary", use_container_width=True):
        with st.spinner("Executing 10-point statistical profiling, IQR anomaly detection, and scoring..."):
            st.session_state["df_raw"] = target_df.copy()
            st.session_state["df_clean"] = target_df.copy()
            st.session_state["active_dataset_name"] = target_name
            st.session_state["dataset_metadata"] = target_meta

            analyzer = QualityAnalyzer(
                iqr_multiplier=st.session_state["settings"].get("iqr_threshold", 1.5),
                scoring_weights=st.session_state["settings"].get("scoring_weights")
            )
            diagnosis = analyzer.analyze(target_df, dataset_name=target_name)
            st.session_state["diagnosis_results"] = diagnosis
            st.session_state["quality_scores"] = diagnosis["quality_scores"]
            st.session_state["cleaning_history"] = []
            st.session_state["ai_summary_cache"] = None

        st.success("Diagnosis completed successfully! Redirecting to Dataset Diagnosis.")
        st.switch_page("pages/3_Dataset_Diagnosis.py")

elif st.session_state.get("df_raw") is not None:
    st.info(f"Currently active in session: **`{st.session_state.get('active_dataset_name')}`** ({len(st.session_state.get('df_raw')):,} records). Upload another dataset above to analyze a new file.")

render_footer()
