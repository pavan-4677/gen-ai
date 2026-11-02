"""
Page 2: Dataset Upload
Provides file ingestion (CSV, TSV, XLSX, XLS), size & type validation, metadata preview,
and triggers full dataset diagnosis.
"""

import os
import streamlit as st
import pandas as pd

from src.utils import init_session_state, load_css, render_hero, render_footer
from src.data_loader import DataLoader, DataLoaderError
from src.quality_analyzer import QualityAnalyzer

st.set_page_config(
    page_title="Upload Dataset | Dataset Quality Analysis",
    page_icon="📂",
    layout="wide"
)

init_session_state()
load_css()

render_hero(
    title="Dataset Ingestion & Validation",
    subtitle="Upload CSV or Excel spreadsheets. Automatic encoding sniffing, format checking, and structural validation.",
    badge="Step 1 • Ingestion"
)

max_mb = st.session_state["settings"].get("max_upload_size_mb", 50)
loader = DataLoader(max_file_size_mb=max_mb)

# Upload tabs: Upload File OR Use Sample Data
tab_upload, tab_sample = st.tabs(["📤 Upload New Dataset", "🧪 Use Pre-Loaded Sample Datasets"])

with tab_upload:
    st.markdown(f"**Accepted formats:** `.csv`, `.tsv`, `.xlsx`, `.xls` (Maximum file size: **{max_mb} MB**)")
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
            st.success(f"Successfully validated `{uploaded_file.name}` ({meta['formatted_size']}). Ready for diagnosis.")
        except DataLoaderError as e:
            st.error(f"Validation Error: {str(e)}")
            st.session_state["temp_df"] = None
        except Exception as e:
            st.error(f"Unexpected error while reading dataset: {str(e)}")
            st.session_state["temp_df"] = None

with tab_sample:
    st.markdown("Select a benchmark dataset with known real-world data quality issues:")
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("#### 👤 Customer Demographics Benchmark")
        st.caption("Contains missing values, duplicate records, mixed string casing, extreme outliers in income/age, and placeholder tokens.")
        if st.button("Load 'sample_dataset.csv'", use_container_width=True):
            p = os.path.join(sample_dir, "sample_dataset.csv")
            df, meta = loader.load_dataset(p)
            st.session_state["temp_df"] = df
            st.session_state["temp_meta"] = meta
            st.session_state["temp_filename"] = "sample_dataset.csv"
            st.success("Loaded 'sample_dataset.csv' into preview buffer.")

    with col_s2:
        st.markdown("#### 📉 Telecom Churn Quality Benchmark")
        st.caption("Contains missing tenure/charges, outliers, '?' tokens, unformatted string currencies, and duplicate subscriber entries.")
        if st.button("Load 'customer_churn_dirty.csv'", use_container_width=True):
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
    st.markdown("---")
    st.subheader(f"📋 Dataset Overview: {target_name}")

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
    st.markdown("#### First 10 Records Preview")
    st.dataframe(target_df.head(10), use_container_width=True)

    # Column inspection table
    with st.expander("🔍 View Column Names & Data Types", expanded=False):
        col_info = []
        for c in target_df.columns:
            col_info.append({
                "Column Name": str(c),
                "Detected Type": str(target_df[c].dtype),
                "Non-Null Count": int(target_df[c].notna().sum()),
                "Null Count": int(target_df[c].isna().sum()),
                "Null %": f"{(target_df[c].isna().sum() / len(target_df)) * 100:.1f}%",
                "Unique Values": int(target_df[c].nunique(dropna=True))
            })
        st.dataframe(pd.DataFrame(col_info), use_container_width=True)

    # Run Analysis Button
    st.markdown("### 🚀 Execute Dataset Diagnosis")
    if st.button("Start Full Dataset Diagnosis & Quality Analysis", type="primary", use_container_width=True):
        with st.spinner("Executing statistical profiling, IQR anomaly detection, and scoring..."):
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

        st.success("Diagnosis completed successfully! Proceeding to Dataset Diagnosis.")
        st.switch_page("pages/3_Dataset_Diagnosis.py")

elif st.session_state.get("df_raw") is not None:
    st.info(f"Currently active dataset: **`{st.session_state.get('active_dataset_name')}`** ({len(st.session_state.get('df_raw'))} records). Upload another file above to replace it.")

render_footer()
