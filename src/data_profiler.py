"""
Data Profiler Module for Intelligent Dataset Diagnosis and Quality Analysis System.
Generates comprehensive statistical profiles for numerical, categorical, and datetime attributes.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class DataProfiler:
    """Profiles datasets providing distribution, cardinality, and summary metrics."""

    def profile_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate complete dataset and column-level statistical profile.
        
        Args:
            df: Input pandas DataFrame.
            
        Returns:
            Dictionary with overview, column profiles, and correlation matrix.
        """
        total_rows = len(df)
        total_cols = len(df.columns)
        total_cells = total_rows * total_cols
        total_missing = int(df.isna().sum().sum())
        missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0

        duplicate_rows = int(df.duplicated().sum())
        duplicate_pct = round((duplicate_rows / total_rows) * 100, 2) if total_rows > 0 else 0.0

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        date_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
        bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

        # Profiling columns
        columns_profile: Dict[str, Dict[str, Any]] = {}
        for col in df.columns:
            columns_profile[str(col)] = self._profile_column(df[col], total_rows)

        # Correlation matrix for numerical
        corr_matrix: Dict[str, Dict[str, float]] = {}
        if len(num_cols) >= 2:
            try:
                corr_df = df[num_cols].corr()
                corr_matrix = {
                    str(c): {str(r): round(float(val), 3) if not np.isnan(val) else 0.0 for r, val in corr_df[c].items()}
                    for c in corr_df.columns
                }
            except Exception:
                corr_matrix = {}

        return {
            "overview": {
                "total_rows": total_rows,
                "total_columns": total_cols,
                "total_cells": total_cells,
                "total_missing_cells": total_missing,
                "missing_percentage": missing_pct,
                "duplicate_rows_count": duplicate_rows,
                "duplicate_percentage": duplicate_pct,
                "numerical_columns_count": len(num_cols),
                "categorical_columns_count": len(cat_cols),
                "datetime_columns_count": len(date_cols),
                "boolean_columns_count": len(bool_cols),
                "numerical_column_names": [str(c) for c in num_cols],
                "categorical_column_names": [str(c) for c in cat_cols]
            },
            "columns": columns_profile,
            "correlation_matrix": corr_matrix
        }

    def _profile_column(self, series: pd.Series, total_rows: int) -> Dict[str, Any]:
        """Profile an individual column."""
        missing_count = int(series.isna().sum())
        missing_pct = round((missing_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))
        is_constant = unique_count <= 1
        is_high_cardinality = (unique_count / total_rows > 0.8) and (unique_count > 20) if total_rows > 0 else False

        profile: Dict[str, Any] = {
            "name": str(series.name),
            "dtype": str(series.dtype),
            "total_count": total_rows,
            "non_null_count": total_rows - missing_count,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": unique_count,
            "is_constant": is_constant,
            "is_high_cardinality": is_high_cardinality
        }

        if pd.api.types.is_numeric_dtype(series):
            valid_vals = series.dropna()
            profile["kind"] = "numerical"
            if len(valid_vals) > 0:
                profile.update({
                    "mean": round(float(valid_vals.mean()), 3),
                    "std": round(float(valid_vals.std(ddof=1)), 3) if len(valid_vals) > 1 else 0.0,
                    "min": round(float(valid_vals.min()), 3),
                    "q25": round(float(valid_vals.quantile(0.25)), 3),
                    "median": round(float(valid_vals.median()), 3),
                    "q75": round(float(valid_vals.quantile(0.75)), 3),
                    "max": round(float(valid_vals.max()), 3),
                    "skewness": round(float(valid_vals.skew()), 3) if len(valid_vals) > 2 else 0.0,
                    "zeros_count": int((valid_vals == 0).sum()),
                    "negatives_count": int((valid_vals < 0).sum())
                })
        elif pd.api.types.is_datetime64_any_dtype(series):
            valid_vals = series.dropna()
            profile["kind"] = "datetime"
            if len(valid_vals) > 0:
                profile.update({
                    "min_date": str(valid_vals.min()),
                    "max_date": str(valid_vals.max())
                })
        else:
            # Categorical or string
            valid_vals = series.dropna().astype(str)
            profile["kind"] = "categorical"
            if len(valid_vals) > 0:
                top_counts = valid_vals.value_counts().head(5)
                profile["top_values"] = [
                    {"value": str(k), "count": int(v), "pct": round((v / len(valid_vals)) * 100, 2)}
                    for k, v in top_counts.items()
                ]
            else:
                profile["top_values"] = []

        return profile

