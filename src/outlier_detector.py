"""
Outlier Detection Module for Intelligent Dataset Diagnosis and Quality Analysis System.
Implements Interquartile Range (IQR) and Z-score methods with detailed statistical reporting.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class OutlierDetector:
    """Detects statistical outliers in numerical columns using IQR and Z-Score techniques."""

    def __init__(self, iqr_multiplier: float = 1.5, z_threshold: float = 3.0):
        self.iqr_multiplier = iqr_multiplier
        self.z_threshold = z_threshold

    def detect_outliers_iqr(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        multiplier: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Detect outliers across numerical columns using the Interquartile Range (IQR) method.

        Formula:
            IQR = Q3 - Q1
            Lower Bound = Q1 - (multiplier * IQR)
            Upper Bound = Q3 + (multiplier * IQR)
            Outlier: value < Lower Bound OR value > Upper Bound

        Args:
            df: Input DataFrame.
            columns: Specific numerical columns to check. If None, checks all numerical columns.
            multiplier: IQR multiplier (default 1.5).

        Returns:
            Dictionary containing dataset-level and column-level outlier summaries.
        """
        k = multiplier if multiplier is not None else self.iqr_multiplier
        if columns is None:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            num_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

        column_reports: Dict[str, Dict[str, Any]] = {}
        all_outlier_indices = set()

        for col in num_cols:
            series = df[col].dropna()
            if len(series) < 4:
                # Not enough points for reliable IQR
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1

            lower_bound = q1 - (k * iqr)
            upper_bound = q3 + (k * iqr)

            # Mask of outliers in non-null data
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_indices = df[outlier_mask].index.tolist()
            outlier_values = df.loc[outlier_mask, col].tolist()

            all_outlier_indices.update(outlier_indices)

            column_reports[col] = {
                "q1": round(q1, 3),
                "q3": round(q3, 3),
                "iqr": round(iqr, 3),
                "lower_bound": round(lower_bound, 3),
                "upper_bound": round(upper_bound, 3),
                "outlier_count": len(outlier_indices),
                "outlier_pct": round((len(outlier_indices) / len(df)) * 100, 2) if len(df) > 0 else 0.0,
                "outlier_indices": outlier_indices,
                "sample_outliers": outlier_values[:10]  # Cap preview at 10 items
            }

        total_rows = len(df)
        outlier_rows_count = len(all_outlier_indices)
        outlier_rows_pct = round((outlier_rows_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

        return {
            "method": "IQR",
            "multiplier": k,
            "columns_analyzed": num_cols,
            "columns_with_outliers": [col for col, rep in column_reports.items() if rep["outlier_count"] > 0],
            "total_outlier_cells": sum(rep["outlier_count"] for rep in column_reports.values()),
            "outlier_rows_count": outlier_rows_count,
            "outlier_rows_percentage": outlier_rows_pct,
            "affected_row_indices": sorted(list(all_outlier_indices)),
            "column_reports": column_reports
        }

    def detect_outliers_zscore(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Detect outliers using Z-Score (|z| > threshold)."""
        thresh = threshold if threshold is not None else self.z_threshold
        if columns is None:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            num_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

        column_reports: Dict[str, Dict[str, Any]] = {}
        all_outlier_indices = set()

        for col in num_cols:
            series = df[col].dropna()
            if len(series) < 3:
                continue

            mean = float(series.mean())
            std = float(series.std(ddof=1))
            if std == 0:
                continue

            z_scores = (df[col] - mean) / std
            outlier_mask = z_scores.abs() > thresh
            outlier_indices = df[outlier_mask].index.tolist()

            all_outlier_indices.update(outlier_indices)

            column_reports[col] = {
                "mean": round(mean, 3),
                "std": round(std, 3),
                "threshold": thresh,
                "outlier_count": len(outlier_indices),
                "outlier_pct": round((len(outlier_indices) / len(df)) * 100, 2) if len(df) > 0 else 0.0,
                "outlier_indices": outlier_indices
            }

        return {
            "method": "Z-Score",
            "threshold": thresh,
            "columns_analyzed": num_cols,
            "columns_with_outliers": [col for col, rep in column_reports.items() if rep["outlier_count"] > 0],
            "total_outlier_cells": sum(rep["outlier_count"] for rep in column_reports.values()),
            "outlier_rows_count": len(all_outlier_indices),
            "affected_row_indices": sorted(list(all_outlier_indices)),
            "column_reports": column_reports
        }

    @staticmethod
    def get_outlier_rows(df: pd.DataFrame, outlier_indices: List[Any]) -> pd.DataFrame:
        """Extract slice of DataFrame containing the outlier records for review."""
        valid_indices = [idx for idx in outlier_indices if idx in df.index]
        return df.loc[valid_indices]

