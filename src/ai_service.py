"""
Generative AI Service for Intelligent Dataset Diagnosis and Quality Analysis System.
Provides structured prompt synthesis, OpenAI-compatible API integration, prompt-injection defense,
dedicated anomaly analyzers, ML readiness evaluators, and an intelligent offline fallback diagnosis engine.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional
import requests


class AIService:
    """
    Enterprise-grade Generative AI Service for Tabular Data Quality.
    Communicates with OpenAI-compatible endpoints or seamlessly falls back to a deterministic,
    expert heuristic reasoning engine grounded strictly in verified dataset metrics.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1"
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")

    def is_api_configured(self) -> bool:
        """Check if a non-placeholder API key is configured."""
        return bool(self.api_key and self.api_key.strip() and not self.api_key.startswith("your_"))

    # =========================================================================
    # Structured Metadata Extraction (Zero Raw Data Transmission)
    # =========================================================================
    def create_structured_summary(self, diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract only necessary metadata, statistics, and issue summaries for the LLM.
        Raw individual rows are NEVER transmitted. Untrusted strings are strictly sanitized.
        """
        profile = diagnosis.get("profile", {}).get("overview", {})
        scores = diagnosis.get("quality_scores", {})
        dimensions = scores.get("dimensions", {})
        
        # Missing values breakdown
        missing_info = []
        for col_data in diagnosis.get("missing_analysis", {}).get("column_breakdown", []):
            missing_info.append({
                "column": self._sanitize_text(col_data["column"]),
                "null_count": col_data["missing_count"],
                "null_percentage": col_data["missing_percentage"],
                "severity": col_data.get("severity", "Low")
            })

        # Outlier breakdown
        outlier_info = []
        outlier_rep = diagnosis.get("outlier_analysis", {}).get("column_reports", {})
        for col, rep in outlier_rep.items():
            if rep.get("outlier_count", 0) > 0:
                outlier_info.append({
                    "column": self._sanitize_text(col),
                    "outlier_count": rep["outlier_count"],
                    "outlier_percentage": rep["outlier_pct"],
                    "iqr": rep.get("iqr", 0),
                    "bounds": f"[{rep.get('lower_bound')} to {rep.get('upper_bound')}]",
                    "sample_values": [round(float(v), 2) for v in rep.get("sample_outliers", [])[:5]]
                })

        # Data type mismatches and sentinel tokens
        dtype_warnings = diagnosis.get("dtype_analysis", {}).get("type_mismatch_warnings", [])
        placeholders = diagnosis.get("invalid_values_analysis", {}).get("placeholder_issues", [])
        negative_issues = diagnosis.get("invalid_values_analysis", {}).get("negative_value_issues", [])

        # Detected actionable issues
        sanitized_issues = []
        for issue in diagnosis.get("detected_issues", [])[:15]:
            sanitized_issues.append({
                "category": issue.get("category"),
                "severity": issue.get("severity"),
                "column": self._sanitize_text(issue.get("column", "")),
                "description": self._sanitize_text(issue.get("description", "")),
                "recommendation": self._sanitize_text(issue.get("recommendation", ""))
            })

        return {
            "dataset_name": self._sanitize_text(diagnosis.get("dataset_name", "Uploaded Dataset")),
            "dimensions": {
                "rows": profile.get("total_rows", 0),
                "columns": profile.get("total_columns", 0),
                "total_cells": profile.get("total_cells", 0),
                "numerical_features_count": profile.get("numerical_columns_count", 0),
                "categorical_features_count": profile.get("categorical_columns_count", 0),
                "datetime_features_count": profile.get("datetime_columns_count", 0),
                "numerical_columns": [self._sanitize_text(c) for c in profile.get("numerical_column_names", [])],
                "categorical_columns": [self._sanitize_text(c) for c in profile.get("categorical_column_names", [])]
            },
            "scores": {
                "overall_score": scores.get("overall_score", 0),
                "grade": scores.get("grade", "N/A"),
                "status": scores.get("status", "N/A"),
                "completeness": dimensions.get("completeness", {}).get("score", 0),
                "uniqueness": dimensions.get("uniqueness", {}).get("score", 0),
                "validity": dimensions.get("validity", {}).get("score", 0),
                "consistency": dimensions.get("consistency", {}).get("score", 0)
            },
            "missing_summary": {
                "total_missing_cells": diagnosis.get("missing_analysis", {}).get("total_missing_cells", 0),
                "missing_percentage": profile.get("missing_percentage", 0),
                "columns_with_missing_count": diagnosis.get("missing_analysis", {}).get("columns_with_missing_count", 0),
                "column_breakdown": missing_info
            },
            "duplicate_summary": {
                "count": diagnosis.get("duplicate_analysis", {}).get("duplicate_rows_count", 0),
                "percentage": diagnosis.get("duplicate_analysis", {}).get("duplicate_percentage", 0)
            },
            "outlier_summary": {
                "outlier_rows_count": diagnosis.get("outlier_analysis", {}).get("outlier_rows_count", 0),
                "outlier_rows_percentage": diagnosis.get("outlier_analysis", {}).get("outlier_rows_percentage", 0),
                "columns_with_outliers_count": len(outlier_info),
                "columns": outlier_info
            },
            "schema_anomalies": {
                "type_warnings": dtype_warnings,
                "placeholders": placeholders,
                "domain_negatives": negative_issues
            },
            "detected_issues": sanitized_issues
        }

    # =========================================================================
    # 1. AI-Generated Dataset Summary
    # =========================================================================
    def generate_dataset_summary(self, diagnosis: Dict[str, Any]) -> str:
        """Generate a natural-language executive summary of the dataset structure and characteristics."""
        summary = self.create_structured_summary(diagnosis)

        if self.is_api_configured():
            try:
                system_prompt = (
                    "You are a Senior Principal Data Quality Engineer. "
                    "Write a professional, concise executive dataset summary based strictly on the provided JSON metadata. "
                    "Highlight: 1) Dataset scale and composition, 2) Dominant data types, 3) Overall health grade, and 4) Top structural traits. "
                    "Never invent numbers or columns not present in the summary."
                )
                user_prompt = f"Dataset Metadata:\n```json\n{json.dumps(summary, indent=2)}\n```"
                resp = self._call_llm_api([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ])
                if resp:
                    return resp
            except Exception:
                pass  # Fallback to local rule-based generation

        # Deterministic Fallback
        dims = summary["dimensions"]
        scores = summary["scores"]
        num_cols = dims.get("numerical_columns", [])
        cat_cols = dims.get("categorical_columns", [])
        num_str = f"numerical features ({', '.join(num_cols[:4])}{'...' if len(num_cols) > 4 else ''})" if num_cols else "no numerical features"
        cat_str = f"categorical attributes ({', '.join(cat_cols[:4])}{'...' if len(cat_cols) > 4 else ''})" if cat_cols else "no categorical attributes"

        return (
            f"The dataset **'{summary['dataset_name']}'** consists of **{dims['rows']:,} records** across **{dims['columns']} attributes**, "
            f"comprising a total volume of **{dims['total_cells']:,} data cells**. Structurally, the dataset features {len(num_cols)} {num_str} "
            f"and {len(cat_cols)} {cat_str}.\n\n"
            f"From an automated data quality perspective, the dataset achieves an **Overall Quality Score of {scores['overall_score']}/100 ({scores['grade']})**, "
            f"reflecting **{scores['completeness']}% completeness**, **{scores['uniqueness']}% uniqueness**, **{scores['validity']}% validity**, "
            f"and **{scores['consistency']}% formatting consistency**. Its current operational readiness is categorized as: **{scores['status']}**."
        )

    # =========================================================================
    # 2. AI Explanation of Missing Values, Duplicates, and Outliers
    # =========================================================================
    def explain_anomalies(self, diagnosis: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate deep-dive technical explanations for the three critical anomaly families:
        Missing Values, Record Duplication, and Statistical Outliers.
        """
        summary = self.create_structured_summary(diagnosis)
        
        # 1. Missing Values Explanation
        miss_summary = summary["missing_summary"]
        if miss_summary["total_missing_cells"] == 0:
            miss_text = "✅ **Zero Missing Values**: The dataset is 100% complete across all attributes. No records will be dropped due to missingness during model training."
        else:
            col_bullets = []
            for m in miss_summary["column_breakdown"][:6]:
                col_bullets.append(f"- **`{m['column']}`**: **{m['null_count']} missing rows** ({m['null_percentage']}% null) • Severity: `{m['severity']}`")
            col_list_str = "\n".join(col_bullets)
            miss_text = (
                f"⚠️ **Detected {miss_summary['total_missing_cells']} Missing Data Points** across {miss_summary['columns_with_missing_count']} columns "
                f"({miss_summary['missing_percentage']}% overall dataset sparsity).\n\n"
                f"{col_list_str}\n\n"
                f"**Technical Impact**: Missing values cause native estimators (e.g., standard scikit-learn Linear Regression, SVMs, and neural networks) "
                f"to fail on ingestion without explicit imputation. High null sparsity (>40%) in a single feature indicates systemic collection loss or optional input fields."
            )

        # 2. Duplicate Records Explanation
        dup_summary = summary["duplicate_summary"]
        if dup_summary["count"] == 0:
            dup_text = "✅ **100% Record Uniqueness**: Zero duplicate rows identified. Every record represents a distinct observation."
        else:
            dup_text = (
                f"⚠️ **Detected {dup_summary['count']} Exact Duplicate Rows** ({dup_summary['percentage']}% of the dataset).\n\n"
                f"**Technical Impact**: Duplicate rows artificially inflate statistical confidence, bias metric evaluations (such as AUC-ROC or R²), "
                f"and cause data leakage if identical records cross the training and validation splits. Deduplication should be executed prior to modeling."
            )

        # 3. Outlier Explanation (IQR Method)
        out_summary = summary["outlier_summary"]
        if out_summary["columns_with_outliers_count"] == 0:
            out_text = "✅ **Zero Statistical Anomalies**: No numerical values exceed standard Tukey Interquartile Range bounds ($Q_1 - 1.5\\times\\text{IQR}$ to $Q_3 + 1.5\\times\\text{IQR}$)."
        else:
            out_bullets = []
            for o in out_summary["columns"][:6]:
                sample_str = f" (Sample anomalies: {o['sample_values']})" if o['sample_values'] else ""
                out_bullets.append(f"- **`{o['column']}`**: **{o['outlier_count']} outliers** ({o['outlier_percentage']}%) outside valid range `{o['bounds']}`{sample_str}")
            out_list_str = "\n".join(out_bullets)
            out_text = (
                f"⚠️ **Detected {out_summary['outlier_rows_count']} Outlier Records** ({out_summary['outlier_rows_percentage']}% of total rows) "
                f"across {out_summary['columns_with_outliers_count']} numerical attributes using the Interquartile Range (IQR 1.5x) method.\n\n"
                f"{out_list_str}\n\n"
                f"**Technical Impact**: Extreme outliers distort parameter estimation in OLS regression, distort variance calculations, and destabilize gradient descent updates. "
                f"Applying **Winsorization (capping to IQR bounds)** is strongly recommended to preserve observation count while mitigating distortion."
            )

        return {
            "missing_values_explanation": miss_text,
            "duplicates_explanation": dup_text,
            "outliers_explanation": out_text
        }

    # =========================================================================
    # 3. AI-Based Data Cleaning Recommendations
    # =========================================================================
    def generate_cleaning_recommendations(self, diagnosis: Dict[str, Any]) -> str:
        """Synthesize prioritized, step-by-step cleaning recommendations matching the Cleaning Studio."""
        summary = self.create_structured_summary(diagnosis)
        miss_summary = summary["missing_summary"]
        dup_summary = summary["duplicate_summary"]
        out_summary = summary["outlier_summary"]
        schemas = summary.get("schema_anomalies", {})

        recommendations = []
        step_idx = 1

        # Step A: Deduplication
        if dup_summary["count"] > 0:
            recommendations.append(
                f"**Step {step_idx}: Row Deduplication**  \n"
                f"Remove the **{dup_summary['count']} duplicate rows** in the Data Cleaning Studio. "
                f"Use `keep='first'` to maintain the chronological baseline observation."
            )
            step_idx += 1

        # Step B: Sentinel token replacement
        if schemas.get("placeholders"):
            cols = [p["column"] for p in schemas["placeholders"]]
            recommendations.append(
                f"**Step {step_idx}: Standardize Placeholder Tokens**  \n"
                f"Convert string tokens (such as `?`, `NULL`, `unknown`) to standard `NaN` across columns: `{', '.join(cols)}`. "
                f"This ensures missingness can be imputed numerically."
            )
            step_idx += 1

        # Step C: Missing value treatment
        if miss_summary["total_missing_cells"] > 0:
            impute_actions = []
            for m in miss_summary["column_breakdown"]:
                col = m["column"]
                pct = m["null_percentage"]
                if pct >= 50.0:
                    impute_actions.append(f"- Drop column `{col}` ({pct}% null exceeds 50% threshold).")
                elif col in summary["dimensions"]["numerical_columns"]:
                    impute_actions.append(f"- Impute `{col}` ({pct}% null) using **median** to prevent skewness from potential outliers.")
                else:
                    impute_actions.append(f"- Impute `{col}` ({pct}% null) using **mode** or assign an `'Unknown'` category.")
            recommendations.append(
                f"**Step {step_idx}: Missing Value Imputation**  \n" + "\n".join(impute_actions)
            )
            step_idx += 1

        # Step D: Outlier handling
        if out_summary["columns_with_outliers_count"] > 0:
            out_actions = [f"- Cap `{o['column']}` to range `{o['bounds']}`" for o in out_summary["columns"][:4]]
            recommendations.append(
                f"**Step {step_idx}: Outlier Mitigation (Winsorization)**  \n"
                f"Apply IQR capping (Winsorization) to numerical attributes to restrain extreme values without losing valuable records:\n" +
                "\n".join(out_actions)
            )
            step_idx += 1

        # Step E: Schema & Type conversions
        if schemas.get("type_warnings"):
            type_actions = [f"- Convert `{w['column']}` to `{w['suggested_dtype']}` ({w['reason']})" for w in schemas["type_warnings"]]
            recommendations.append(
                f"**Step {step_idx}: Schema Type Casting**  \n" + "\n".join(type_actions)
            )
            step_idx += 1

        if not recommendations:
            return "🎉 **No Mandatory Cleaning Steps Required**: The dataset is clean, deduplicated, and fully populated. You may proceed directly to modeling."

        return "\n\n".join(recommendations)

    # =========================================================================
    # 4. Machine Learning Readiness Assessment
    # =========================================================================
    def assess_ml_readiness(self, diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate dataset suitability for machine learning model architectures
        (Linear Models, Tree-Based Models, Neural Networks) and outline critical risks.
        """
        summary = self.create_structured_summary(diagnosis)
        scores = summary["scores"]
        overall = scores["overall_score"]
        miss_count = summary["missing_summary"]["total_missing_cells"]
        dup_count = summary["duplicate_summary"]["count"]
        out_count = summary["outlier_summary"]["outlier_rows_count"]

        if overall >= 85:
            readiness_status = "HIGH"
            badge = "🟢 Ready for Production ML"
            summary_eval = "The dataset demonstrates strong baseline integrity with minimal data hygiene defects. Feature engineering can proceed directly."
        elif overall >= 65:
            readiness_status = "MODERATE"
            badge = "🟡 Preprocessing Required"
            summary_eval = "The dataset has moderate defects (missing values and/or outliers) that must be remediated in the Cleaning Studio to avoid estimator bias."
        else:
            readiness_status = "LOW"
            badge = "🔴 Critical Data Defects (High Risk)"
            summary_eval = "Direct model training is strongly discouraged. Unhandled nulls, duplicates, and unvalidated domains will induce severe variance and data leakage."

        # Model Family Suitability
        model_suitability = {
            "tree_based": {
                "name": "Tree-Based Models (Random Forest, XGBoost, LightGBM)",
                "compatibility": "High" if dup_count == 0 else "Moderate",
                "notes": "Naturally robust to unscaled numerical features and monotonic outliers. However, duplicate rows will still bias tree partition frequencies."
            },
            "linear_models": {
                "name": "Linear Models (Logistic / Ridge / Lasso Regression)",
                "compatibility": "Low" if out_count > 0 or miss_count > 0 else "High",
                "notes": "Highly sensitive to un-Winsorized outliers (distorts OLS regression hyperplanes) and requires 100% imputation of missing cells."
            },
            "neural_networks": {
                "name": "Neural Networks & Deep Learning",
                "compatibility": "Low" if out_count > 0 or miss_count > 0 else "High",
                "notes": "Uncapped outliers trigger exploding gradients during backpropagation. Requires scaling (StandardScaler/MinMax) and complete imputation."
            }
        }

        # Specific Preprocessing Checklist
        checklist = [
            {"item": "Deduplicate Rows", "status": "Done" if dup_count == 0 else "Pending Action", "critical": dup_count > 0},
            {"item": "Impute Missing Cells", "status": "Done" if miss_count == 0 else "Pending Action", "critical": miss_count > 0},
            {"item": "Cap Statistical Outliers", "status": "Done" if out_count == 0 else "Pending Action", "critical": out_count > 0},
            {"item": "Standardize Text Casing", "status": "Done" if scores["consistency"] >= 95 else "Recommended", "critical": False}
        ]

        return {
            "status": readiness_status,
            "badge": badge,
            "overall_score": overall,
            "summary_evaluation": summary_eval,
            "model_suitability": model_suitability,
            "checklist": checklist
        }

    # =========================================================================
    # 5. Full AI Quality Report Generation
    # =========================================================================
    def generate_full_quality_report(
        self,
        diagnosis: Dict[str, Any],
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        Generate in-depth AI explanation of data quality, issues, causes, and ML recommendations.
        Uses OpenAI API if configured; otherwise seamlessly uses the deterministic expert report.
        """
        structured_summary = self.create_structured_summary(diagnosis)

        if self.is_api_configured():
            try:
                system_prompt = (
                    "You are a Senior Principal AI/ML Engineer and Data Quality Auditor. "
                    "Analyze the provided structured dataset diagnostics JSON. "
                    "Strict Instructions:\n"
                    "1. Rely ONLY on the facts and statistics provided in the JSON.\n"
                    "2. NEVER invent fake numbers, column names, or cleaning steps.\n"
                    "3. Clearly distinguish factual diagnosis from cleaning recommendations.\n"
                    "4. Structure your response into:\n"
                    "   - 1. Executive Summary & Health Grade\n"
                    "   - 2. Critical Quality Bottlenecks (Missing, Duplicates, Outliers)\n"
                    "   - 3. Root Cause Hypotheses\n"
                    "   - 4. Actionable Step-by-Step Data Cleaning Strategy\n"
                    "   - 5. Machine Learning Readiness & Risk Evaluation\n"
                    "5. Keep the language crisp, professional, and accessible."
                )

                user_prompt = f"Dataset Quality Analysis Data:\n```json\n{json.dumps(structured_summary, indent=2)}\n```"
                if custom_instructions:
                    user_prompt += f"\n\nAdditional Guidance: {self._sanitize_text(custom_instructions)}"

                response = self._call_llm_api([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ])
                if response:
                    return response
            except Exception as e:
                return self._generate_rule_based_explanation(structured_summary, error_note=str(e))

        # Fallback mode
        return self._generate_rule_based_explanation(structured_summary)

    # Legacy method compatibility
    def generate_comprehensive_explanation(self, diagnosis: Dict[str, Any], custom_instructions: Optional[str] = None) -> str:
        return self.generate_full_quality_report(diagnosis, custom_instructions)

    # =========================================================================
    # 6. Interactive Chatbot for Dataset Questions
    # =========================================================================
    def answer_user_query(
        self,
        user_query: str,
        chat_history: List[Dict[str, str]],
        diagnosis: Dict[str, Any]
    ) -> str:
        """
        Answer user questions in the interactive AI chat assistant.
        Uses live OpenAI completion if configured, or falls back to the deterministic Q&A engine.
        """
        structured_summary = self.create_structured_summary(diagnosis)
        sanitized_query = self._sanitize_text(user_query)

        if self.is_api_configured():
            try:
                system_prompt = (
                    "You are an Intelligent Tabular Data Quality Assistant. "
                    "You have direct access to the diagnosed dataset metrics in the JSON below. "
                    "Instructions:\n"
                    "1. Answer the user's questions accurately using ONLY these verified statistics.\n"
                    "2. NEVER invent numerical values, column names, or claims.\n"
                    "3. If asked about ML readiness, reference the computed health score and detected defects.\n"
                    "4. If asked about data cleaning, reference exact strategies (mean/median, mode, Winsorization).\n"
                    f"\nDataset Diagnostics Metadata:\n{json.dumps(structured_summary, indent=2)}"
                )

                messages = [{"role": "system", "content": system_prompt}]
                for msg in chat_history[-6:]:
                    messages.append({"role": msg["role"], "content": self._sanitize_text(msg["content"])})
                messages.append({"role": "user", "content": sanitized_query})

                response = self._call_llm_api(messages)
                if response:
                    return response
            except Exception:
                pass  # Fallback to local rule-based answering engine

        return self._generate_rule_based_chat_answer(sanitized_query, structured_summary)

    # =========================================================================
    # LLM API Caller with Robust Error Handling
    # =========================================================================
    def _call_llm_api(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Make HTTP POST call to OpenAI-compatible endpoint with explicit error handling."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1400
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=25)
        except requests.exceptions.Timeout:
            raise RuntimeError("API request timed out (25s limit reached). Switching to offline deterministic engine.")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Unable to reach LLM API endpoint. Verify network connectivity or base URL.")
        except Exception as e:
            raise RuntimeError(f"Unexpected connection error: {str(e)}")

        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        elif resp.status_code == 401:
            raise RuntimeError("Invalid or expired API Key (HTTP 401). Verify credentials in Settings.")
        elif resp.status_code == 429:
            raise RuntimeError("Rate limit or quota exceeded (HTTP 429). Check your provider billing or limits.")
        else:
            raise RuntimeError(f"LLM API returned status {resp.status_code}: {resp.text}")

    # =========================================================================
    # Deterministic Rule-Based Fallback Synthesis
    # =========================================================================
    def _generate_rule_based_explanation(
        self,
        summary: Dict[str, Any],
        error_note: Optional[str] = None
    ) -> str:
        """High-grade deterministic rule-based analysis synthesis when API is offline."""
        scores = summary.get("scores", {})
        overall = scores.get("overall_score", 0)
        grade = scores.get("grade", "N/A")
        dims = summary.get("dimensions", {})
        missing_list = summary.get("missing_summary", {}).get("column_breakdown", [])
        outliers_list = summary.get("outlier_summary", {}).get("columns", [])
        duplicates = summary.get("duplicate_summary", {})

        note = ""
        if error_note:
            note = f"> *Note: Switched to deterministic diagnosis engine ({error_note}). Non-AI analysis remains 100% operational.*\n\n"
        elif not self.is_api_configured():
            note = "> *Note: Running in Deterministic Heuristic Engine Mode (No live OpenAI API key detected). Real statistical insights generated below. Add an API key in Settings to activate dynamic LLM completions.*\n\n"

        missing_text = ""
        if missing_list:
            missing_text = "\n".join([f"- **`{m['column']}`**: {m['null_count']} missing rows ({m['null_percentage']}%) • Severity: `{m.get('severity', 'Low')}`" for m in missing_list])
        else:
            missing_text = "- No missing values detected across attributes (100% complete)."

        outlier_text = ""
        if outliers_list:
            outlier_text = "\n".join([f"- **`{o['column']}`**: {o['outlier_count']} outliers ({o['outlier_percentage']}%) outside valid range {o['bounds']}" for o in outliers_list])
        else:
            outlier_text = "- No extreme IQR statistical anomalies found."

        duplicate_text = (
            f"- **{duplicates.get('count', 0)} exact duplicate rows** ({duplicates.get('percentage', 0)}% of dataset)."
            if duplicates.get("count", 0) > 0 else
            "- No duplicate records identified (100% unique records)."
        )

        ml_readiness = (
            "✅ **High Machine Learning Readiness**: The dataset demonstrates strong baseline integrity with minimal anomalies. Direct feature engineering can proceed."
            if overall >= 85 else
            ("⚠️ **Moderate Readiness**: Preprocessing required. Missing values and distribution skews must be addressed in the Cleaning Studio before model training to avoid bias."
             if overall >= 65 else
             "🚨 **Not Ready for ML**: Severe data hygiene issues detected (missing data, duplicates, and unvalidated types). Training models directly will lead to high variance, data leakage, and degraded generalization.")
        )

        report = f"""{note}### 1. Executive Summary & Health Grade
The dataset **'{summary.get('dataset_name')}'** comprises **{dims.get('rows', 0):,} records** and **{dims.get('columns', 0)} attributes** ({dims.get('total_cells', 0):,} total data points).

- **Overall Data Quality Score**: **{overall}/100** ({grade})
- **Completeness Dimension**: {scores.get('completeness', 0)}/100
- **Uniqueness Dimension**: {scores.get('uniqueness', 0)}/100
- **Validity Dimension**: {scores.get('validity', 0)}/100
- **Consistency Dimension**: {scores.get('consistency', 0)}/100

---

### 2. Critical Quality Findings
#### Missing Values
{missing_text}

#### Record Duplication
{duplicate_text}

#### Statistical Outliers (IQR Method)
{outlier_text}

---

### 3. Probable Root Causes
1. **Upstream Ingestion Gaps**: Missing values in customer or operational fields typically originate from optional user form inputs or dropped upstream pipeline payloads.
2. **Batch Ingestion Repetition**: Duplicate records frequently indicate un-idempotent ETL sync jobs or re-queued webhook retries.
3. **Data Entry & Typing Errors**: Outliers and sentinel tokens (e.g. negative ages or string placeholders like '?' and 'NULL') indicate insufficient client-side validation during data collection.

---

### 4. Recommended Step-by-Step Cleaning Roadmap
1. **Deduplication**: Execute exact row deduplication to remove the {duplicates.get('count', 0)} redundant rows.
2. **Missing Value Treatment**:
   - For numerical columns with moderate missingness (<20%), apply **median imputation** to resist skewness.
   - For categorical attributes, apply **mode imputation** or flag with `'Unknown'`.
   - Any column with >50% null ratio should be evaluated for elimination.
3. **Outlier Mitigation**: Review detected anomalies. Apply **Winsorization (capping to IQR bounds)** for legitimate extreme values, or filter out unphysical readings.
4. **Text & Type Standardization**: Standardize capitalization and strip extraneous whitespace across text columns to unify category groupings.

---

### 5. Machine Learning Readiness & Risk Evaluation
{ml_readiness}
"""
        return report

    def _generate_rule_based_chat_answer(self, query: str, summary: Dict[str, Any]) -> str:
        """Provide direct, context-grounded answers to common diagnostic questions."""
        q = query.lower()
        scores = summary.get("scores", {})
        missing = summary.get("missing_summary", {}).get("column_breakdown", [])
        outliers = summary.get("outlier_summary", {}).get("columns", [])
        duplicates = summary.get("duplicate_summary", {})
        overall = scores.get("overall_score", 0)

        if any(term in q for term in ["summary", "overview", "describe", "what is this dataset"]):
            dims = summary.get("dimensions", {})
            return (
                f"**Dataset Summary for '{summary.get('dataset_name')}'**:\n\n"
                f"- **Dimensions**: {dims.get('rows', 0):,} records × {dims.get('columns', 0)} columns ({dims.get('total_cells', 0):,} data points)\n"
                f"- **Data Quality Score**: **{overall}/100** ({scores.get('grade')})\n"
                f"- **Feature Breakdown**: {dims.get('numerical_features_count', 0)} numerical, {dims.get('categorical_features_count', 0)} categorical\n"
                f"- **Completeness**: {scores.get('completeness')}% ({len(missing)} columns with nulls)\n"
                f"- **Uniqueness**: {scores.get('uniqueness')}% ({duplicates.get('count', 0)} duplicates)\n"
                f"- **Anomalies**: {len(outliers)} numerical columns with statistical outliers (IQR method)."
            )

        elif any(term in q for term in ["wrong", "issue", "problem", "what is wrong"]):
            issues = summary.get("detected_issues", [])
            if not issues:
                return f"Your dataset looks very clean! Overall quality score is **{overall}/100**. No major red flags were detected."
            issue_bullets = "\n".join([f"- **{i['category']}** in `{i['column']}`: {i['description']}" for i in issues[:6]])
            return f"Here is the breakdown of issues detected in **'{summary.get('dataset_name')}'**:\n\n{issue_bullets}\n\nOverall Quality Score: **{overall}/100**."

        elif any(term in q for term in ["why", "low", "score", "grade"]):
            return (
                f"Your overall score is **{overall}/100** ({scores.get('grade')}).\n\n"
                f"Points were evaluated across 4 core dimensions:\n"
                f"- **Completeness ({scores.get('completeness')}/100)**: Reflects {len(missing)} columns with missing cells.\n"
                f"- **Uniqueness ({scores.get('uniqueness')}/100)**: Impacted by {duplicates.get('count', 0)} duplicate rows ({duplicates.get('percentage')}%\n"
                f"- **Validity ({scores.get('validity')}/100)**: Reflects invalid formatting, negative values in positive domains, or placeholder strings.\n"
                f"- **Consistency ({scores.get('consistency')}/100)**: Affected by text casing variations and statistical outliers."
            )

        elif "missing" in q:
            if not missing:
                return "Good news! There are **0 missing values** in this dataset. Completeness is at 100%."
            m_list = "\n".join([f"- **`{m['column']}`**: {m['null_count']} missing rows ({m['null_percentage']}%)" for m in missing])
            return (
                f"The following columns have missing values:\n\n{m_list}\n\n"
                f"**Recommendation**: Use median imputation for numeric features with skewed distributions, "
                f"mode for categoricals, or drop rows if missingness is concentrated in non-critical records."
            )

        elif "duplicate" in q:
            if duplicates.get("count", 0) == 0:
                return "Zero duplicate rows exist in this dataset. Uniqueness is 100%."
            return (
                f"Detected **{duplicates.get('count')} duplicate rows** ({duplicates.get('percentage')}% of the dataset).\n\n"
                "**Recommendation**: Execute row deduplication in the Data Cleaning Studio. Duplicate rows inflate sample sizes and can cause data leakage between train and test splits."
            )

        elif "outlier" in q:
            if not outliers:
                return "No statistical outliers were detected using the Interquartile Range (IQR 1.5x) method."
            o_list = "\n".join([f"- **`{o['column']}`**: {o['outlier_count']} outliers ({o['outlier_percentage']}%) outside {o['bounds']}" for o in outliers])
            return (
                f"Statistical outliers were detected in {len(outliers)} numerical columns:\n\n{o_list}\n\n"
                f"**Handling Strategy**: Use the Data Cleaning Studio to either remove the rows or cap them (Winsorize) "
                f"to the IQR bounds so you don't discard valid high-value observations."
            )

        elif any(term in q for term in ["ml", "machine learning", "ready", "model"]):
            if overall >= 85:
                return (
                    f"**Yes**, the dataset has an overall score of **{overall}/100**, making it well-suited for machine learning. "
                    "You can proceed directly with feature encoding and scaling."
                )
            else:
                return (
                    f"**Not yet**. With an overall quality score of **{overall}/100**, there are {len(missing)} missing columns "
                    f"and {duplicates.get('count', 0)} duplicate records. Feeding raw data with these defects into models can cause "
                    "data leakage and estimation bias. Run the recommended steps in the Data Cleaning Studio first."
                )

        elif any(term in q for term in ["preprocessing", "clean", "steps", "how should i"]):
            return (
                "Here are the recommended preprocessing steps in order:\n\n"
                "1. **Deduplication**: Remove duplicate rows in the Data Cleaning Studio.\n"
                "2. **Standardize Placeholders**: Convert tokens like `'?'` or `'NULL'` into standard nulls.\n"
                "3. **Missing Value Imputation**: Impute numerical columns using median and categorical using mode.\n"
                "4. **Outlier Treatment**: Cap extreme outliers to upper/lower IQR bounds (Winsorization).\n"
                "5. **Formatting**: Trim whitespace and standardize casing on text attributes."
            )

        elif any(term in q for term in ["simple", "english", "explain"]):
            return (
                f"In simple terms: You have a table with **{summary['dimensions']['rows']} rows** and **{summary['dimensions']['columns']} columns**. "
                f"Its overall health is rated **{scores.get('grade')} ({overall} out of 100)**. "
                f"It contains {duplicates.get('count', 0)} repeated rows, {len(missing)} columns where some values are left blank, "
                f"and {len(outliers)} columns with unusually high or low numbers. "
                "Cleaning these up will make your charts and AI predictions much more trustworthy."
            )

        else:
            return (
                f"Regarding your query about **'{query}'**: The dataset **'{summary.get('dataset_name')}'** currently has "
                f"an overall quality score of **{overall}/100**, {len(missing)} columns with missing entries, and "
                f"{duplicates.get('count', 0)} duplicate rows. You can ask me specific questions about missing values, "
                "outliers, ML readiness, or recommended cleaning operations!"
            )

    @staticmethod
    def _sanitize_text(text: Any) -> str:
        """Sanitize text against prompt injection or malicious formatting."""
        if not isinstance(text, str):
            text = str(text)
        # Strip potential instruction overrides or tags
        text = re.sub(r"[<>{}\\]", "", text)
        return text[:250].strip()
