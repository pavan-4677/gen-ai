"""
Page 6: Data Cleaning Studio
Interactive, non-destructive data remediation studio with granular transformations,
before/after diff comparisons, audit logging, and full undo/reset capability.
"""

from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

from src.utils import init_session_state, load_css, render_hero, render_footer
from src.data_cleaner import DataCleaner
from src.outlier_detector import OutlierDetector
from src.quality_analyzer import QualityAnalyzer

st.set_page_config(
    page_title="Cleaning Studio | Dataset Quality Analysis",
    page_icon="🧹",
    layout="wide"
)

init_session_state()
load_css()

render_hero(
    title="Data Cleaning & Remediation Studio",
    subtitle="Interactive non-destructive transformations with strict confirmation, before/after diffs, and audit trail.",
    badge="Step 5 • Data Remediation"
)

if st.session_state.get("df_raw") is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset to begin cleaning.")
    if st.button("📂 Go to Dataset Upload"):
        st.switch_page("pages/2_Upload_Dataset.py")
    st.stop()

# Ensure df_clean exists
if st.session_state.get("df_clean") is None:
    st.session_state["df_clean"] = st.session_state["df_raw"].copy()

df_raw = st.session_state["df_raw"]
df_clean = st.session_state["df_clean"]
cleaner = DataCleaner()

# Top Stats Row: Comparison between Raw and Clean
col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
raw_rows = len(df_raw)
cur_rows = len(df_clean)
raw_cols = len(df_raw.columns)
cur_cols = len(df_clean.columns)
raw_nulls = int(df_raw.isna().sum().sum())
cur_nulls = int(df_clean.isna().sum().sum())

with col_s1:
    st.metric("Cleaned Rows", f"{cur_rows:,}", delta=f"{cur_rows - raw_rows} rows" if cur_rows != raw_rows else "No change")
with col_s2:
    st.metric("Cleaned Columns", cur_cols, delta=f"{cur_cols - raw_cols} cols" if cur_cols != raw_cols else "No change")
with col_s3:
    st.metric("Remaining Nulls", f"{cur_nulls:,}", delta=f"{cur_nulls - raw_nulls} nulls" if cur_nulls != raw_nulls else "No change", delta_color="inverse")
with col_s4:
    st.metric("Steps Applied", len(st.session_state["cleaning_history"]))
with col_s5:
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        if st.button("↩️ Undo", help="Revert the most recent cleaning operation", disabled=len(st.session_state["cleaning_history"]) == 0):
            history = st.session_state["cleaning_history"]
            if history:
                history.pop()  # Remove last operation record
                # Replay history from df_raw to reconstruct clean state accurately
                reconstructed = df_raw.copy()
                # For safety, if history is now empty, reset
                if not history:
                    st.session_state["df_clean"] = reconstructed
                else:
                    # Restore previous snapshot
                    st.session_state["df_clean"] = history[-1].get("snapshot", reconstructed).copy()
                st.session_state["ai_summary_cache"] = None
                st.success("Reverted last cleaning operation.")
                st.rerun()

    with col_u2:
        if st.button("🔄 Reset", help="Reset all changes back to raw dataset"):
            st.session_state["df_clean"] = df_raw.copy()
            st.session_state["cleaning_history"] = []
            st.session_state["ai_summary_cache"] = None
            st.warning("Dataset reset to original raw state.")
            st.rerun()

st.markdown("---")

# Cleaning Operations Tabs
tab_missing, tab_dup, tab_outlier, tab_types, tab_text, tab_drop = st.tabs([
    "❓ Missing Values",
    "👥 Duplicate Records",
    "📈 Outliers Treatment",
    "🏷️ Type Casting",
    "🔠 Text & Formatting",
    "🗑️ Drop Columns"
])

def apply_transformation(new_df: pd.DataFrame, record: dict):
    """Helper to commit a cleaning step with snapshot and audit log."""
    record["timestamp"] = datetime.now().strftime("%H:%M:%S")
    record["snapshot"] = new_df.copy()
    st.session_state["cleaning_history"].append(record)
    st.session_state["df_clean"] = new_df
    # Invalidate AI summary cache so it reflects the updated clean state
    st.session_state["ai_summary_cache"] = None
    st.success(f"Executed: {record['action']} — {record['details']}")
    st.rerun()

# 1. Missing Values Tab
with tab_missing:
    st.subheader("Missing Value Remediation")
    op_type = st.radio("Choose operation:", ["Impute Missing Values", "Drop Rows with Missing Values", "Drop Columns Exceeding Threshold"], horizontal=True)

    if op_type == "Impute Missing Values":
        cols_with_null = df_clean.columns[df_clean.isna().any()].tolist()
        if not cols_with_null:
            st.success("No columns currently contain missing values.")
        else:
            col_target = st.selectbox("Select column to impute:", cols_with_null)
            is_num = pd.api.types.is_numeric_dtype(df_clean[col_target])
            strategies = ["median", "mean", "mode", "constant"] if is_num else ["mode", "constant"]
            strat = st.selectbox("Imputation Strategy:", strategies)
            custom_val = None
            if strat == "constant":
                custom_val = st.text_input("Enter custom fill value:", value="Missing" if not is_num else "0")
                if is_num:
                    try:
                        custom_val = float(custom_val)
                    except ValueError:
                        st.error("Please enter a valid numeric value for a numerical column.")

            null_count = int(df_clean[col_target].isna().sum())
            st.caption(f"Target column '{col_target}' has **{null_count} null cells**.")

            if st.button(f"Apply {strat.capitalize()} Imputation to '{col_target}'", type="primary"):
                try:
                    new_df, rec = cleaner.impute_missing(df_clean, col_target, strategy=strat, custom_value=custom_val)
                    apply_transformation(new_df, rec)
                except Exception as e:
                    st.error(f"Error during imputation: {str(e)}")

    elif op_type == "Drop Rows with Missing Values":
        mode = st.selectbox("Drop condition:", ["any", "all"], help="'any' drops rows where ANY selected column is null; 'all' drops rows where ALL selected columns are null.")
        subset_cols = st.multiselect("Apply to specific columns (leave blank for all columns):", df_clean.columns.tolist())
        
        subset = subset_cols if subset_cols else None
        preview_drop_count = len(df_clean) - len(df_clean.dropna(subset=subset, how=mode))
        st.warning(f"This will drop **{preview_drop_count} rows** from the dataset.")
        
        if st.button("Confirm: Drop Missing Rows", type="primary"):
            new_df, rec = cleaner.drop_missing_rows(df_clean, columns=subset_cols, how=mode)
            apply_transformation(new_df, rec)

    elif op_type == "Drop Columns Exceeding Threshold":
        thresh = st.slider("Missing percentage threshold to drop column (%):", min_value=10, max_value=90, value=50, step=5)
        exceeding = [c for c in df_clean.columns if (df_clean[c].isna().sum() / len(df_clean)) * 100 >= thresh]
        if exceeding:
            st.warning(f"Columns matching criteria (>= {thresh}% null): `{exceeding}`")
            if st.button(f"Confirm: Drop {len(exceeding)} Column(s)", type="primary"):
                new_df, rec = cleaner.drop_missing_columns(df_clean, threshold_pct=float(thresh))
                apply_transformation(new_df, rec)
        else:
            st.info(f"No columns currently exceed the {thresh}% missing threshold.")

# 2. Duplicates Tab
with tab_dup:
    st.subheader("Duplicate Records Removal")
    exact_dups = int(df_clean.duplicated().sum())
    st.write(f"The cleaned dataset currently contains **{exact_dups} duplicate rows**.")
    
    keep_opt = st.selectbox("Keep occurrence:", ["first", "last"])
    subset_dup = st.multiselect("Deduplicate based on subset key columns (leave blank for exact full-row match):", df_clean.columns.tolist())
    
    if exact_dups > 0 or subset_dup:
        if st.button("Confirm: Remove Duplicate Records", type="primary"):
            new_df, rec = cleaner.remove_duplicates(df_clean, subset=subset_dup if subset_dup else None, keep=keep_opt)
            apply_transformation(new_df, rec)
    else:
        st.success("No duplicate records exist in the current cleaned dataset.")

# 3. Outliers Tab
with tab_outlier:
    st.subheader("Statistical Outliers Treatment (IQR Method)")
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.info("No numerical columns available for outlier detection.")
    else:
        outlier_col = st.selectbox("Select numerical column to analyze for outliers:", num_cols)
        k_val = st.slider("IQR multiplier threshold:", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

        # Inspect outliers
        detector = OutlierDetector(iqr_multiplier=k_val)
        res = detector.detect_outliers_iqr(df_clean, columns=[outlier_col], multiplier=k_val)
        rep = res["column_reports"].get(outlier_col, {})
        out_cnt = rep.get("outlier_count", 0)

        st.write(
            f"Column **`{outlier_col}`** has **{out_cnt} outliers** outside IQR bounds "
            f"**[{rep.get('lower_bound', 0):.2f}, {rep.get('upper_bound', 0):.2f}]**."
        )

        if out_cnt > 0:
            with st.expander("🔍 Preview Outlier Records Prior to Remediation"):
                st.dataframe(OutlierDetector.get_outlier_rows(df_clean, rep.get("outlier_indices", [])), use_container_width=True)

            c_act1, c_act2 = st.columns(2)
            with c_act1:
                st.markdown("##### Option A: Winsorize (Cap Outliers)")
                st.caption("Replaces values exceeding boundaries with the exact boundary value. Preserves row count.")
                if st.button(f"Cap Outliers in '{outlier_col}' (Winsorize)", type="primary"):
                    new_df, rec = cleaner.cap_outliers_iqr(df_clean, outlier_col, multiplier=k_val)
                    apply_transformation(new_df, rec)

            with c_act2:
                st.markdown("##### Option B: Remove Outlier Rows")
                st.caption("Filters out rows containing anomalies from the dataset.")
                if st.button(f"Drop {out_cnt} Outlier Rows in '{outlier_col}'"):
                    new_df, rec = cleaner.remove_outliers_iqr(df_clean, outlier_col, multiplier=k_val)
                    apply_transformation(new_df, rec)
        else:
            st.success(f"No statistical outliers detected in '{outlier_col}' with multiplier {k_val}.")

# 4. Data Types Tab
with tab_types:
    st.subheader("Data Type Conversion & Schema Casting")
    type_col = st.selectbox("Select column to cast:", df_clean.columns.tolist())
    st.caption(f"Current data type: **`{df_clean[type_col].dtype}`**")

    target_t = st.selectbox("Target Data Type:", ["integer", "float", "string", "datetime"])
    dt_format = None
    if target_t == "datetime":
        dt_format = st.text_input("Custom date format (optional, e.g. %Y-%m-%d):", value="")

    if st.button(f"Convert '{type_col}' to {target_t.capitalize()}", type="primary"):
        try:
            new_df, rec = cleaner.convert_data_type(
                df_clean, type_col, target_type=target_t, datetime_format=dt_format if dt_format else None
            )
            apply_transformation(new_df, rec)
        except Exception as e:
            st.error(f"Conversion failed: {str(e)}")

# 5. Text & Consistency Tab
with tab_text:
    st.subheader("Text Formatting & Sentinel Value Replacement")
    col_str1, col_str2 = st.columns(2)

    with col_str1:
        st.markdown("#### Text Standardization")
        text_cols = df_clean.select_dtypes(include=["object", "string"]).columns.tolist()
        if text_cols:
            t_col = st.selectbox("Select text column to standardize:", text_cols)
            strip_ws = st.checkbox("Strip leading & trailing whitespace", value=True)
            casing_choice = st.selectbox("Apply casing transformation:", ["title", "lower", "upper", "none"])

            if st.button(f"Standardize '{t_col}'", type="primary"):
                new_df, rec = cleaner.standardize_text(df_clean, t_col, strip_whitespace=strip_ws, casing=casing_choice)
                apply_transformation(new_df, rec)
        else:
            st.info("No text attributes available in dataset.")

    with col_str2:
        st.markdown("#### Replace Sentinel Placeholders with NaN")
        st.caption("Converts text tokens like `?`, `null`, `unknown`, `NA`, `-999` to standard pandas NaN.")
        custom_placeholders = st.text_input("Placeholder tokens (comma-separated):", value="?, na, n/a, null, none, unknown, -999, 9999")
        if st.button("Convert Placeholders to NaN", type="primary"):
            token_list = [t.strip() for t in custom_placeholders.split(",") if t.strip()]
            new_df, rec = cleaner.replace_placeholders(df_clean, placeholders=token_list)
            apply_transformation(new_df, rec)

# 6. Drop Columns Tab
with tab_drop:
    st.subheader("Drop Unnecessary Columns")
    cols_to_remove = st.multiselect("Select columns to permanently drop:", df_clean.columns.tolist())
    if cols_to_remove:
        st.warning(f"Will delete columns: `{cols_to_remove}`")
        if st.button(f"Confirm: Drop {len(cols_to_remove)} Column(s)", type="primary"):
            new_df, rec = cleaner.drop_columns(df_clean, cols_to_remove)
            apply_transformation(new_df, rec)

st.markdown("---")

# Audit Trail & Preview
st.subheader("📋 Cleaning Audit Trail & Transformation History")
history = st.session_state.get("cleaning_history", [])
if history:
    history_display = [
        {
            "Step": i + 1,
            "Time": h["timestamp"],
            "Action": h["action"],
            "Details": h["details"],
            "Rows Delta": h["rows_delta"],
            "Cols Delta": h["cols_delta"]
        }
        for i, h in enumerate(history)
    ]
    st.dataframe(pd.DataFrame(history_display), use_container_width=True)
else:
    st.info("No cleaning steps applied yet. The dataset is currently in its original state.")

# Side-by-side or Tabbed Cleaned Preview
with st.expander("🔍 Preview First 10 Records of Cleaned Dataset", expanded=False):
    st.dataframe(df_clean.head(10), use_container_width=True)

render_footer()
