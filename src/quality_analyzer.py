"""
Quality Analyzer Module for Intelligent Dataset Diagnosis and Quality Analysis System.
Orchestrates statistical profiling, outlier detection, data integrity checks, and scoring.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.data_profiler import DataProfiler
from src.outlier_detector import OutlierDetector
from src.quality_scoring import QualityScoringEngine


class QualityAnalyzer:
    """Comprehensive diagnosis engine analyzing data quality across all required dimensions."""

    def __init__(
        self,
        iqr_multiplier: float = 1.5,
        scoring_weights: Optional[Dict[str, float]] = None
    ):
        self.profiler = DataProfiler()
        self.outlier_detector = OutlierDetector(iqr_multiplier=iqr_multiplier)
        self.scoring_engine = QualityScoringEngine(weights=scoring_weights)

    def analyze(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict[str, Any]:
        """
        Run all 10 diagnostic checks and calculate quality scores.
        
        Args:
            df: Input dataset.
            dataset_name: Display label for the dataset.
            
        Returns:
            Dictionary containing complete diagnosis results and quality scores.
        """
        if df is None or df.empty:
            return {"error": "Cannot analyze an empty dataset."}

        # 1. Base profiling (overview, column metrics, correlation)
        profile = self.profiler.profile_dataset(df)

        # 2. Outlier detection (IQR method)
        outlier_res = self.outlier_detector.detect_outliers_iqr(df)

        # 3. Missing values analysis
        missing_res = self._analyze_missing_values(df)

        # 4. Duplicate rows analysis
        duplicate_res = self._analyze_duplicates(df)

        # 5. Data type & type inference analysis
        dtype_res = self._analyze_data_types(df)

        # 6. Invalid & placeholder values
        invalid_res = self._analyze_invalid_values(df)

        # 7. Constant & high-cardinality columns
        cardinality_res = self._analyze_cardinality(df)

        # 8. Text consistency checks
        consistency_res = self._analyze_text_consistency(df)

        # 9. Calculate transparent data quality score
        quality_scores = self.scoring_engine.calculate_scores(df, outlier_summary=outlier_res)

        # 10. Compile list of actionable issues
        detected_issues = self._compile_issues(
            missing_res, duplicate_res, outlier_res, dtype_res, invalid_res, cardinality_res, consistency_res
        )

        return {
            "dataset_name": dataset_name,
            "profile": profile,
            "missing_analysis": missing_res,
            "duplicate_analysis": duplicate_res,
            "outlier_analysis": outlier_res,
            "dtype_analysis": dtype_res,
            "invalid_values_analysis": invalid_res,
            "cardinality_analysis": cardinality_res,
            "consistency_analysis": consistency_res,
            "quality_scores": quality_scores,
            "detected_issues": detected_issues,
            "total_issues_count": len(detected_issues)
        }

    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detailed missing value breakdown."""
        total_rows = len(df)
        missing_counts = df.isna().sum()
        cols_with_missing = missing_counts[missing_counts > 0].to_dict()

        col_details = []
        for col, cnt in cols_with_missing.items():
            pct = round((cnt / total_rows) * 100, 2) if total_rows > 0 else 0.0
            col_details.append({
                "column": str(col),
                "missing_count": int(cnt),
                "missing_percentage": pct,
                "severity": "High" if pct > 30 else ("Medium" if pct > 10 else "Low")
            })

        col_details.sort(key=lambda x: x["missing_count"], reverse=True)

        rows_with_any_missing = int(df.isna().any(axis=1).sum())
        rows_missing_pct = round((rows_with_any_missing / total_rows) * 100, 2) if total_rows > 0 else 0.0

        return {
            "total_missing_cells": int(missing_counts.sum()),
            "columns_with_missing_count": len(cols_with_missing),
            "rows_with_missing": rows_with_any_missing,
            "rows_with_missing_pct": rows_missing_pct,
            "column_breakdown": col_details
        }

    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Exact duplicate detection and row inspection."""
        duplicate_mask = df.duplicated(keep=False)
        dup_count = int(df.duplicated(keep="first").sum())
        total_affected_rows = int(duplicate_mask.sum())
        dup_indices = df[duplicate_mask].index.tolist()

        return {
            "duplicate_rows_count": dup_count,
            "total_duplicate_instances": total_affected_rows,
            "duplicate_percentage": round((dup_count / len(df)) * 100, 2) if len(df) > 0 else 0.0,
            "duplicate_indices": dup_indices
        }

    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze actual dtypes and infer potential type mismatches."""
        inferred_types = {}
        type_warnings = []

        for col in df.columns:
            series = df[col].dropna()
            current_dtype = str(series.dtype)
            
            # Check if object column could be numeric
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                cleaned_str = series.astype(str).str.replace(r"[$,€£% ]", "", regex=True)
                coerced_num = pd.to_numeric(cleaned_str, errors="coerce")
                valid_num_ratio = coerced_num.notna().sum() / len(series) if len(series) > 0 else 0.0
                
                if valid_num_ratio >= 0.8:
                    type_warnings.append({
                        "column": str(col),
                        "current_dtype": current_dtype,
                        "suggested_dtype": "numerical (float/int)",
                        "reason": f"{valid_num_ratio * 100:.1f}% of values can be converted to numbers (possibly contains currency symbols or dirty strings)."
                    })
                else:
                    # Check if it could be datetime
                    try:
                        coerced_dates = pd.to_datetime(series, errors="coerce", format="mixed")
                        valid_date_ratio = coerced_dates.notna().sum() / len(series) if len(series) > 0 else 0.0
                        if valid_date_ratio >= 0.8:
                            type_warnings.append({
                                "column": str(col),
                                "current_dtype": current_dtype,
                                "suggested_dtype": "datetime",
                                "reason": f"{valid_date_ratio * 100:.1f}% of values follow recognizable date/time formats."
                            })
                    except Exception:
                        pass

            inferred_types[str(col)] = current_dtype

        return {
            "column_types": inferred_types,
            "type_mismatch_warnings": type_warnings
        }

    def _analyze_invalid_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect placeholder string tokens ('?', 'null', 'unknown') and impossible numericals."""
        placeholders_detected = []
        placeholder_tokens = {"?", "na", "n/a", "null", "none", "unknown", "nan", "-999", "9999", "nil"}

        for col in df.select_dtypes(include=["object", "string"]).columns:
            series = df[col].dropna().astype(str).str.strip().str.lower()
            mask = series.isin(placeholder_tokens)
            count = int(mask.sum())
            if count > 0:
                found_tokens = sorted(list(series[mask].unique()))
                placeholders_detected.append({
                    "column": str(col),
                    "placeholder_count": count,
                    "tokens_found": found_tokens
                })

        negative_warnings = []
        for col in df.select_dtypes(include=[np.number]).columns:
            col_l = str(col).lower()
            if any(term in col_l for term in ["age", "salary", "income", "price", "revenue", "tenure", "cost"]):
                neg_count = int((df[col] < 0).sum())
                if neg_count > 0:
                    negative_warnings.append({
                        "column": str(col),
                        "negative_count": neg_count,
                        "min_value": float(df[col].min())
                    })

        return {
            "placeholder_issues": placeholders_detected,
            "negative_value_issues": negative_warnings
        }

    def _analyze_cardinality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect constant (zero variance) and high-cardinality candidate ID columns."""
        total_rows = len(df)
        constant_columns = []
        high_cardinality_columns = []

        for col in df.columns:
            uniques = df[col].nunique(dropna=True)
            if uniques <= 1:
                constant_columns.append({
                    "column": str(col),
                    "unique_values": uniques,
                    "constant_value": str(df[col].dropna().iloc[0]) if uniques == 1 else "None (All Null)"
                })
            elif uniques > 20 and (uniques / total_rows > 0.8) and pd.api.types.is_object_dtype(df[col]):
                high_cardinality_columns.append({
                    "column": str(col),
                    "unique_values": uniques,
                    "cardinality_ratio": round((uniques / total_rows) * 100, 2),
                    "inference": "Likely an identifier, primary key, or unique text field."
                })

        return {
            "constant_columns": constant_columns,
            "high_cardinality_columns": high_cardinality_columns
        }

    def _analyze_text_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect untrimmed spaces and inconsistent capitalization across text columns."""
        text_issues = []
        for col in df.select_dtypes(include=["object", "string"]).columns:
            series = df[col].dropna().astype(str)
            if len(series) == 0:
                continue

            whitespace_count = int((series.str.strip() != series).sum())
            uniques_orig = series.nunique()
            uniques_clean = series.str.strip().str.lower().nunique()
            casing_discrepancy = uniques_orig > uniques_clean

            if whitespace_count > 0 or casing_discrepancy:
                text_issues.append({
                    "column": str(col),
                    "whitespace_count": whitespace_count,
                    "casing_inconsistency": casing_discrepancy,
                    "unique_variants": uniques_orig,
                    "canonical_variants": uniques_clean
                })

        return {"text_issues": text_issues}

    def _compile_issues(
        self,
        missing: Dict[str, Any],
        duplicates: Dict[str, Any],
        outliers: Dict[str, Any],
        dtypes: Dict[str, Any],
        invalids: Dict[str, Any],
        cardinality: Dict[str, Any],
        consistency: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Combine all detected issues into a prioritized list for Gen AI and user reporting."""
        issues = []

        # 1. Missing values
        for item in missing.get("column_breakdown", []):
            issues.append({
                "category": "Missing Values",
                "severity": item["severity"],
                "column": item["column"],
                "description": f"Column '{item['column']}' has {item['missing_count']} missing values ({item['missing_percentage']}% null).",
                "recommendation": "Impute with mean/median for numerical or mode for categorical, or drop if null ratio is excessive (>50%)."
            })

        # 2. Duplicates
        if duplicates.get("duplicate_rows_count", 0) > 0:
            issues.append({
                "category": "Duplicates",
                "severity": "Medium",
                "column": "Dataset Rows",
                "description": f"Detected {duplicates['duplicate_rows_count']} exact duplicate rows ({duplicates['duplicate_percentage']}% of dataset).",
                "recommendation": "Remove duplicate rows to prevent overfitting and sample inflation."
            })

        # 3. Outliers
        for col in outliers.get("columns_with_outliers", []):
            rep = outliers["column_reports"][col]
            issues.append({
                "category": "Statistical Outliers",
                "severity": "Medium" if rep["outlier_pct"] < 15 else "High",
                "column": col,
                "description": f"Column '{col}' has {rep['outlier_count']} IQR outliers ({rep['outlier_pct']}% of records). Lower bound: {rep['lower_bound']}, Upper bound: {rep['upper_bound']}.",
                "recommendation": "Inspect affected records; apply Winsorization (capping) or filter out unphysical anomalies."
            })

        # 4. Data types
        for warning in dtypes.get("type_mismatch_warnings", []):
            issues.append({
                "category": "Data Type Mismatch",
                "severity": "Medium",
                "column": warning["column"],
                "description": f"Column '{warning['column']}' is currently '{warning['current_dtype']}' but {warning['reason']}",
                "recommendation": f"Convert column to {warning['suggested_dtype']}."
            })

        # 5. Invalid values / placeholders
        for item in invalids.get("placeholder_issues", []):
            issues.append({
                "category": "Placeholder Values",
                "severity": "Medium",
                "column": item["column"],
                "description": f"Column '{item['column']}' contains {item['placeholder_count']} string placeholder tokens: {item['tokens_found']}.",
                "recommendation": "Standardize placeholders to actual NaN/null and then impute or drop."
            })

        for item in invalids.get("negative_value_issues", []):
            issues.append({
                "category": "Domain Rule Violation",
                "severity": "High",
                "column": item["column"],
                "description": f"Column '{item['column']}' has {item['negative_count']} negative values (min: {item['min_value']}) which violates domain constraints.",
                "recommendation": "Filter out negative records or correct erroneous signs."
            })

        # 6. Cardinality
        for item in cardinality.get("constant_columns", []):
            issues.append({
                "category": "Constant Column",
                "severity": "Low",
                "column": item["column"],
                "description": f"Column '{item['column']}' is constant with value: '{item['constant_value']}'.",
                "recommendation": "Remove constant column as it provides zero variance/information for modeling."
            })

        # 7. Text consistency
        for item in consistency.get("text_issues", []):
            details = []
            if item["whitespace_count"] > 0:
                details.append(f"{item['whitespace_count']} untrimmed whitespace cells")
            if item["casing_inconsistency"]:
                details.append(f"inconsistent capitalization ({item['unique_variants']} variants instead of {item['canonical_variants']})")
            issues.append({
                "category": "Text Consistency",
                "severity": "Low",
                "column": item["column"],
                "description": f"Column '{item['column']}' has {', '.join(details)}.",
                "recommendation": "Strip leading/trailing whitespace and standardize text casing (Title Case / Lowercase)."
            })

        return issues
