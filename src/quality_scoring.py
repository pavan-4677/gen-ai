"""
Quality Scoring Engine for Intelligent Dataset Diagnosis and Quality Analysis System.
Calculates transparent, explainable scores across Completeness, Uniqueness, Validity,
and Consistency dimensions on a 0-100 scale.
"""

from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd


class QualityScoringEngine:
    """
    Transparent quality scoring engine.
    Calculates 4 dimensions: Completeness, Uniqueness, Validity, Consistency.
    Computes weighted overall score and generates plain-language explanations.
    """

    PLACEHOLDER_STRINGS: Set[str] = {
        "?", "na", "n/a", "null", "none", "unknown", "nan", "-999", "9999", "inf", "-inf", "missing", "nil"
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        custom_placeholders: Optional[Set[str]] = None
    ):
        self.weights = weights or {
            "completeness": 0.25,
            "uniqueness": 0.25,
            "validity": 0.25,
            "consistency": 0.25
        }
        self.placeholders = set(self.PLACEHOLDER_STRINGS)
        if custom_placeholders:
            self.placeholders.update(custom_placeholders)

    def calculate_scores(
        self,
        df: pd.DataFrame,
        outlier_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compute all quality dimensions and overall score.
        
        Args:
            df: Dataset DataFrame.
            outlier_summary: Precomputed outlier summary from OutlierDetector.
            
        Returns:
            Dictionary containing individual dimension scores, overall score,
            deduction reasons, and methodology documentation.
        """
        total_rows = len(df)
        total_cols = len(df.columns)
        total_cells = total_rows * total_cols

        if total_cells == 0:
            return self._empty_dataset_result()

        # 1. Completeness Score
        completeness_res = self._eval_completeness(df, total_cells)

        # 2. Uniqueness Score
        uniqueness_res = self._eval_uniqueness(df, total_rows)

        # 3. Validity Score
        validity_res = self._eval_validity(df, total_cells)

        # 4. Consistency Score
        consistency_res = self._eval_consistency(df, total_cells, outlier_summary)

        # Configurable weighted overall score
        w_comp = self.weights.get("completeness", 0.25)
        w_uniq = self.weights.get("uniqueness", 0.25)
        w_val = self.weights.get("validity", 0.25)
        w_cons = self.weights.get("consistency", 0.25)
        total_weight = w_comp + w_uniq + w_val + w_cons

        if total_weight <= 0:
            total_weight = 1.0

        overall_score = (
            (completeness_res["score"] * w_comp) +
            (uniqueness_res["score"] * w_uniq) +
            (validity_res["score"] * w_val) +
            (consistency_res["score"] * w_cons)
        ) / total_weight

        overall_score = round(max(0.0, min(100.0, overall_score)), 2)

        # Grade classification
        if overall_score >= 90:
            grade = "A (Excellent)"
            status = "Ready for Modeling & Production"
        elif overall_score >= 75:
            grade = "B (Good)"
            status = "Minor Cleaning Recommended"
        elif overall_score >= 50:
            grade = "C (Fair)"
            status = "Moderate Cleaning Required"
        else:
            grade = "D (Poor)"
            status = "Critical Data Issues - High Risk"

        return {
            "overall_score": overall_score,
            "grade": grade,
            "status": status,
            "weights": {
                "completeness": w_comp,
                "uniqueness": w_uniq,
                "validity": w_val,
                "consistency": w_cons
            },
            "dimensions": {
                "completeness": completeness_res,
                "uniqueness": uniqueness_res,
                "validity": validity_res,
                "consistency": consistency_res
            },
            "summary_deductions": (
                completeness_res["deductions"] +
                uniqueness_res["deductions"] +
                validity_res["deductions"] +
                consistency_res["deductions"]
            ),
            "limitations": (
                "This score is an automated statistical assessment based on completeness, "
                "uniqueness, syntactic validity, and formatting consistency. It does not replace "
                "domain-specific business rules, external reference validation, or semantic verification."
            )
        }

    def _eval_completeness(self, df: pd.DataFrame, total_cells: int) -> Dict[str, Any]:
        """Completeness: 100 * (non-null cells / total cells)."""
        missing_count = int(df.isna().sum().sum())
        missing_pct = (missing_count / total_cells) * 100
        score = round(max(0.0, 100.0 - missing_pct), 2)

        deductions = []
        if missing_count > 0:
            top_missing = df.isna().sum().sort_values(ascending=False)
            top_missing = top_missing[top_missing > 0].head(3)
            col_details = ", ".join([f"{col} ({cnt} nulls)" for col, cnt in top_missing.items()])
            deductions.append(
                f"Completeness deducted {missing_pct:.1f}% due to {missing_count} missing cells (Highest in: {col_details})."
            )

        return {
            "score": score,
            "missing_cells": missing_count,
            "missing_percentage": round(missing_pct, 2),
            "deductions": deductions
        }

    def _eval_uniqueness(self, df: pd.DataFrame, total_rows: int) -> Dict[str, Any]:
        """Uniqueness: 100 * (1 - duplicate rows / total rows)."""
        duplicate_count = int(df.duplicated().sum())
        dup_pct = (duplicate_count / total_rows) * 100 if total_rows > 0 else 0.0
        score = round(max(0.0, 100.0 - dup_pct), 2)

        deductions = []
        if duplicate_count > 0:
            deductions.append(
                f"Uniqueness deducted {dup_pct:.1f}% due to {duplicate_count} exact duplicate rows."
            )

        return {
            "score": score,
            "duplicate_rows": duplicate_count,
            "duplicate_percentage": round(dup_pct, 2),
            "deductions": deductions
        }

    def _eval_validity(self, df: pd.DataFrame, total_cells: int) -> Dict[str, Any]:
        """Validity: Checks for string placeholders, negative values where invalid, and corrupt tokens."""
        invalid_count = 0
        reasons = []

        # Check for placeholder string tokens in object/string columns
        for col in df.select_dtypes(include=["object", "string"]).columns:
            str_series = df[col].dropna().astype(str).str.strip().str.lower()
            matched = str_series.isin(self.placeholders).sum()
            if matched > 0:
                invalid_count += matched
                reasons.append(f"Column '{col}' contains {matched} placeholder values (e.g. '?', 'null', 'unknown').")

        # Check for negative values in columns with 'age', 'year', 'salary', 'income', 'price', 'count' in name
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            col_lower = str(col).lower()
            if any(term in col_lower for term in ["age", "price", "income", "salary", "tenure", "year", "count"]):
                negs = (df[col] < 0).sum()
                if negs > 0:
                    invalid_count += negs
                    reasons.append(f"Column '{col}' contains {negs} invalid negative values.")

        invalid_pct = (invalid_count / total_cells) * 100 if total_cells > 0 else 0.0
        score = round(max(0.0, 100.0 - (invalid_pct * 2.0)), 2)  # Weighted impact on validity

        return {
            "score": score,
            "invalid_values_count": int(invalid_count),
            "invalid_percentage": round(invalid_pct, 2),
            "deductions": reasons[:5]  # Cap top reasons
        }

    def _eval_consistency(
        self,
        df: pd.DataFrame,
        total_cells: int,
        outlier_summary: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Consistency: Evaluates text casing variance, leading/trailing whitespace, and extreme outliers."""
        penalty_count = 0
        deductions = []

        # 1. Whitespace and mixed casing in categorical columns
        for col in df.select_dtypes(include=["object", "string"]).columns:
            series = df[col].dropna().astype(str)
            if len(series) == 0:
                continue
            
            # Whitespace check
            has_whitespace = (series.str.strip() != series).sum()
            if has_whitespace > 0:
                penalty_count += has_whitespace
                deductions.append(f"Column '{col}' has {has_whitespace} cells with untrimmed whitespace.")

            # Case inconsistency check: unique values in original vs unique values in lower
            uniques_orig = series.nunique()
            uniques_lower = series.str.strip().str.lower().nunique()
            if uniques_orig > uniques_lower:
                casing_diff = uniques_orig - uniques_lower
                penalty_count += (casing_diff * 5)
                deductions.append(f"Column '{col}' has inconsistent text casing (e.g. 'Male' vs 'male').")

        # 2. Outliers penalty
        if outlier_summary and "total_outlier_cells" in outlier_summary:
            outliers = outlier_summary["total_outlier_cells"]
            if outliers > 0:
                penalty_count += (outliers // 2)
                deductions.append(
                    f"Statistical inconsistency: {outliers} outlier values detected across {len(outlier_summary.get('columns_with_outliers', []))} numerical features."
                )

        penalty_pct = (penalty_count / total_cells) * 100 if total_cells > 0 else 0.0
        score = round(max(0.0, 100.0 - (penalty_pct * 1.5)), 2)

        return {
            "score": score,
            "inconsistency_points": int(penalty_count),
            "deductions": deductions[:5]
        }

    def _empty_dataset_result(self) -> Dict[str, Any]:
        """Fallback for empty dataset."""
        return {
            "overall_score": 0.0,
            "grade": "N/A",
            "status": "Empty Dataset",
            "weights": self.weights,
            "dimensions": {
                "completeness": {"score": 0.0, "deductions": ["Dataset is empty"]},
                "uniqueness": {"score": 0.0, "deductions": []},
                "validity": {"score": 0.0, "deductions": []},
                "consistency": {"score": 0.0, "deductions": []}
            },
            "summary_deductions": ["Dataset contains no rows or columns."],
            "limitations": "Empty dataset cannot be scored."
        }

