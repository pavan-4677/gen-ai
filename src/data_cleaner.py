"""
Data Cleaner Module for Intelligent Dataset Diagnosis and Quality Analysis System.
Provides non-destructive, audited data cleaning transformations with change-tracking.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class DataCleaner:
    """Non-destructive data cleaning engine with comprehensive before/after metrics."""

    @staticmethod
    def _create_change_record(
        action: str,
        details: str,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        cells_modified: int = 0
    ) -> Dict[str, Any]:
        """Generate audit record comparing before and after DataFrames."""
        rows_before = len(df_before)
        rows_after = len(df_after)
        cols_before = len(df_before.columns)
        cols_after = len(df_after.columns)

        return {
            "action": action,
            "details": details,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_delta": rows_after - rows_before,
            "cols_before": cols_before,
            "cols_after": cols_after,
            "cols_delta": cols_after - cols_before,
            "cells_modified": cells_modified
        }

    def drop_missing_rows(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        how: str = "any"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Drop rows containing missing values.
        
        Args:
            df: Input DataFrame.
            columns: Specific columns to evaluate. If None, checks all columns.
            how: 'any' or 'all'.
        """
        df_clean = df.copy()
        subset = columns if columns else None
        before_count = len(df_clean)
        df_clean = df_clean.dropna(subset=subset, how=how).reset_index(drop=True)
        dropped_count = before_count - len(df_clean)

        col_str = f"in columns [{', '.join(columns)}]" if columns else "across all columns"
        record = self._create_change_record(
            action="Drop Missing Rows",
            details=f"Dropped {dropped_count} rows with missing values {col_str} (mode: '{how}').",
            df_before=df,
            df_after=df_clean
        )
        return df_clean, record

    def drop_missing_columns(
        self,
        df: pd.DataFrame,
        threshold_pct: float = 50.0
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Drop columns with missing value percentage exceeding threshold_pct."""
        df_clean = df.copy()
        total_rows = len(df_clean)
        if total_rows == 0:
            return df_clean, self._create_change_record("Drop Missing Columns", "Dataset is empty.", df, df_clean)

        missing_pcts = (df_clean.isna().sum() / total_rows) * 100
        cols_to_drop = missing_pcts[missing_pcts >= threshold_pct].index.tolist()

        df_clean = df_clean.drop(columns=cols_to_drop)

        record = self._create_change_record(
            action="Drop Missing Columns",
            details=f"Dropped {len(cols_to_drop)} columns exceeding {threshold_pct}% missing threshold: {cols_to_drop}.",
            df_before=df,
            df_after=df_clean
        )
        return df_clean, record

    def impute_missing(
        self,
        df: pd.DataFrame,
        column: str,
        strategy: str = "mean",
        custom_value: Optional[Any] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Impute missing values in a column.
        
        Strategies: 'mean', 'median', 'mode', 'constant' (uses custom_value).
        """
        df_clean = df.copy()
        if column not in df_clean.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        missing_mask = df_clean[column].isna()
        missing_count = int(missing_mask.sum())

        if missing_count == 0:
            return df_clean, self._create_change_record(
                "Impute Missing", f"Column '{column}' has 0 missing values.", df, df_clean
            )

        fill_val = None
        series = df_clean[column].dropna()

        if strategy == "mean":
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError(f"Cannot calculate mean on non-numeric column '{column}'.")
            fill_val = series.mean()
        elif strategy == "median":
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError(f"Cannot calculate median on non-numeric column '{column}'.")
            fill_val = series.median()
        elif strategy == "mode":
            mode_vals = series.mode()
            if not mode_vals.empty:
                fill_val = mode_vals.iloc[0]
            else:
                fill_val = "Unknown"
        elif strategy == "constant":
            fill_val = custom_value
        else:
            raise ValueError(f"Unsupported strategy '{strategy}'.")

        df_clean[column] = df_clean[column].fillna(fill_val)

        record = self._create_change_record(
            action=f"Impute Missing ({strategy.capitalize()})",
            details=f"Imputed {missing_count} missing values in '{column}' with '{fill_val}'.",
            df_before=df,
            df_after=df_clean,
            cells_modified=missing_count
        )
        return df_clean, record

    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = "first"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove duplicate records."""
        df_clean = df.copy()
        before_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
        dropped_count = before_count - len(df_clean)

        sub_str = f"based on columns {subset}" if subset else "across all columns"
        record = self._create_change_record(
            action="Remove Duplicates",
            details=f"Removed {dropped_count} duplicate rows {sub_str} (retained {keep} occurrence).",
            df_before=df,
            df_after=df_clean
        )
        return df_clean, record

    def remove_outliers_iqr(
        self,
        df: pd.DataFrame,
        column: str,
        multiplier: float = 1.5
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove rows containing IQR outliers in the specified column."""
        df_clean = df.copy()
        if column not in df_clean.columns or not pd.api.types.is_numeric_dtype(df_clean[column]):
            raise ValueError(f"Column '{column}' must be a numerical column.")

        series = df_clean[column].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (multiplier * iqr)
        upper_bound = q3 + (multiplier * iqr)

        outlier_mask = (df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)
        outlier_count = int(outlier_mask.sum())

        df_clean = df_clean[~outlier_mask].reset_index(drop=True)

        record = self._create_change_record(
            action="Remove Outliers (IQR)",
            details=(
                f"Removed {outlier_count} outlier rows from '{column}' outside range "
                f"[{lower_bound:.2f}, {upper_bound:.2f}] (IQR multiplier: {multiplier})."
            ),
            df_before=df,
            df_after=df_clean
        )
        return df_clean, record

    def cap_outliers_iqr(
        self,
        df: pd.DataFrame,
        column: str,
        multiplier: float = 1.5
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Winsorize (cap) outliers to IQR lower and upper bounds."""
        df_clean = df.copy()
        if column not in df_clean.columns or not pd.api.types.is_numeric_dtype(df_clean[column]):
            raise ValueError(f"Column '{column}' must be a numerical column.")

        series = df_clean[column].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - (multiplier * iqr)
        upper = q3 + (multiplier * iqr)

        capped_lower = (df_clean[column] < lower).sum()
        capped_upper = (df_clean[column] > upper).sum()
        total_capped = int(capped_lower + capped_upper)

        df_clean[column] = df_clean[column].clip(lower=lower, upper=upper)

        record = self._create_change_record(
            action="Cap Outliers (Winsorize)",
            details=f"Capped {total_capped} outliers in '{column}' to bounds [{lower:.2f}, {upper:.2f}].",
            df_before=df,
            df_after=df_clean,
            cells_modified=total_capped
        )
        return df_clean, record

    def convert_data_type(
        self,
        df: pd.DataFrame,
        column: str,
        target_type: str,
        datetime_format: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Convert column data type with robust coercion.
        Target types: 'integer', 'float', 'string', 'datetime'.
        """
        df_clean = df.copy()
        if column not in df_clean.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        prev_type = str(df_clean[column].dtype)

        if target_type == "integer":
            # Strip potential commas, currency, spaces
            if df_clean[column].dtype == object or pd.api.types.is_string_dtype(df_clean[column]):
                clean_str = df_clean[column].astype(str).str.replace(r"[$,€£% ]", "", regex=True)
                clean_num = pd.to_numeric(clean_str, errors="coerce")
            else:
                clean_num = pd.to_numeric(df_clean[column], errors="coerce")
            # Use nullable integer Int64 to preserve NaNs without crashing
            df_clean[column] = clean_num.round().astype("Int64")

        elif target_type == "float":
            if df_clean[column].dtype == object or pd.api.types.is_string_dtype(df_clean[column]):
                clean_str = df_clean[column].astype(str).str.replace(r"[$,€£% ]", "", regex=True)
                df_clean[column] = pd.to_numeric(clean_str, errors="coerce")
            else:
                df_clean[column] = pd.to_numeric(df_clean[column], errors="coerce")

        elif target_type == "string":
            df_clean[column] = df_clean[column].astype(str)

        elif target_type == "datetime":
            if datetime_format:
                df_clean[column] = pd.to_datetime(df_clean[column], format=datetime_format, errors="coerce")
            else:
                df_clean[column] = pd.to_datetime(df_clean[column], errors="coerce")
        else:
            raise ValueError(f"Unsupported target type '{target_type}'.")

        new_type = str(df_clean[column].dtype)
        record = self._create_change_record(
            action=f"Convert Data Type ({target_type})",
            details=f"Converted '{column}' from {prev_type} to {new_type}.",
            df_before=df,
            df_after=df_clean,
            cells_modified=len(df_clean)
        )
        return df_clean, record

    def standardize_text(
        self,
        df: pd.DataFrame,
        column: str,
        strip_whitespace: bool = True,
        casing: str = "title"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Standardize text casing and strip leading/trailing whitespace.
        Casing options: 'title', 'lower', 'upper', 'none'.
        """
        df_clean = df.copy()
        if column not in df_clean.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        series = df_clean[column].dropna().astype(str)
        if strip_whitespace:
            series = series.str.strip()

        if casing == "title":
            series = series.str.title()
        elif casing == "lower":
            series = series.str.lower()
        elif casing == "upper":
            series = series.str.upper()

        df_clean.loc[series.index, column] = series

        record = self._create_change_record(
            action=f"Standardize Text ({casing.capitalize()})",
            details=f"Standardized text in '{column}': stripped whitespace={strip_whitespace}, casing={casing}.",
            df_before=df,
            df_after=df_clean,
            cells_modified=len(series)
        )
        return df_clean, record

    def replace_placeholders(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        placeholders: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Replace string placeholders ('?', 'NA', 'null', etc.) with np.nan."""
        df_clean = df.copy()
        target_cols = columns if columns else df_clean.select_dtypes(include=["object", "string"]).columns.tolist()
        tokens = placeholders if placeholders else ["?", "na", "n/a", "null", "none", "unknown", "nan", "-999", "9999", "nil"]
        tokens_lower = {t.lower() for t in tokens}

        replaced_count = 0
        for col in target_cols:
            if col in df_clean.columns:
                mask = df_clean[col].dropna().astype(str).str.strip().str.lower().isin(tokens_lower)
                col_replaced = int(mask.sum())
                if col_replaced > 0:
                    indices_to_replace = df_clean[col].dropna().index[mask]
                    df_clean.loc[indices_to_replace, col] = np.nan
                    replaced_count += col_replaced

        record = self._create_change_record(
            action="Replace Placeholders with NaN",
            details=f"Replaced {replaced_count} placeholder tokens across columns with standard NaN.",
            df_before=df,
            df_after=df_clean,
            cells_modified=replaced_count
        )
        return df_clean, record

    def drop_columns(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Drop selected columns from the dataset."""
        df_clean = df.copy()
        existing = [c for c in columns if c in df_clean.columns]
        df_clean = df_clean.drop(columns=existing)

        record = self._create_change_record(
            action="Drop Columns",
            details=f"Dropped {len(existing)} column(s): {existing}.",
            df_before=df,
            df_after=df_clean
        )
        return df_clean, record

