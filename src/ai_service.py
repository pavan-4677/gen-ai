"""
Generative AI Service for Intelligent Dataset Diagnosis and Quality Analysis System.
Provides structured prompt synthesis, OpenAI-compatible API integration, prompt-injection defense,
and an intelligent offline fallback diagnosis engine.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional
import requests


class AIService:
    """Manages LLM communication, structured prompt generation, and offline fallback intelligence."""

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
        """Check if an API key is set."""
        return bool(self.api_key and self.api_key.strip() and not self.api_key.startswith("your_"))

    def create_structured_summary(self, diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract only necessary metadata, statistics, and issue summaries for the LLM.
        Raw rows are never forwarded. Untrusted strings are sanitized.
        """
        profile = diagnosis.get("profile", {}).get("overview", {})
        scores = diagnosis.get("quality_scores", {})
        dimensions = scores.get("dimensions", {})
        
        # Extract top missing
        missing_info = []
        for col_data in diagnosis.get("missing_analysis", {}).get("column_breakdown", [])[:5]:
            missing_info.append({
                "column": self._sanitize_text(col_data["column"]),
                "null_count": col_data["missing_count"],
                "null_percentage": col_data["missing_percentage"]
            })

        # Extract outliers
        outlier_info = []
        outlier_rep = diagnosis.get("outlier_analysis", {}).get("column_reports", {})
        for col, rep in list(outlier_rep.items())[:5]:
            if rep.get("outlier_count", 0) > 0:
                outlier_info.append({
                    "column": self._sanitize_text(col),
                    "outlier_count": rep["outlier_count"],
                    "outlier_percentage": rep["outlier_pct"],
                    "bounds": f"[{rep['lower_bound']} to {rep['upper_bound']}]"
                })

        # Extract detected issues
        sanitized_issues = []
        for issue in diagnosis.get("detected_issues", [])[:10]:
            sanitized_issues.append({
                "category": issue.get("category"),
                "severity": issue.get("severity"),
                "column": self._sanitize_text(issue.get("column", "")),
                "description": self._sanitize_text(issue.get("description", ""))
            })

        return {
            "dataset_name": self._sanitize_text(diagnosis.get("dataset_name", "Uploaded Dataset")),
            "dimensions": {
                "rows": profile.get("total_rows", 0),
                "columns": profile.get("total_columns", 0),
                "total_cells": profile.get("total_cells", 0)
            },
            "scores": {
                "overall_score": scores.get("overall_score", 0),
                "grade": scores.get("grade", "N/A"),
                "completeness": dimensions.get("completeness", {}).get("score", 0),
                "uniqueness": dimensions.get("uniqueness", {}).get("score", 0),
                "validity": dimensions.get("validity", {}).get("score", 0),
                "consistency": dimensions.get("consistency", {}).get("score", 0)
            },
            "duplicates": {
                "count": diagnosis.get("duplicate_analysis", {}).get("duplicate_rows_count", 0),
                "percentage": diagnosis.get("duplicate_analysis", {}).get("duplicate_percentage", 0)
            },
            "missing_values": missing_info,
            "outliers": outlier_info,
            "detected_issues": sanitized_issues
        }

    def generate_comprehensive_explanation(
        self,
        diagnosis: Dict[str, Any],
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        Generate in-depth AI explanation of data quality, issues, causes, and ML recommendations.
        Uses OpenAI API if configured; otherwise provides deterministic expert report.
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
                # Log and fallback smoothly
                return self._generate_rule_based_explanation(structured_summary, error_note=str(e))

        # Fallback mode
        return self._generate_rule_based_explanation(structured_summary)

    def answer_user_query(
        self,
        user_query: str,
        chat_history: List[Dict[str, str]],
        diagnosis: Dict[str, Any]
    ) -> str:
        """
        Answer user questions in the interactive AI chat assistant.
        """
        structured_summary = self.create_structured_summary(diagnosis)
        sanitized_query = self._sanitize_text(user_query)

        if self.is_api_configured():
            try:
                system_prompt = (
                    "You are an Intelligent Dataset Quality Assistant. You have access to the diagnosed dataset "
                    "metrics in the JSON below. Answer the user's questions accurately using only these actual facts. "
                    "Never invent statistics. If the user asks whether something was cleaned, check if the dataset "
                    "was cleaned; do not claim actions that were not performed. "
                    f"\n\nDataset Summary:\n{json.dumps(structured_summary, indent=2)}"
                )

                messages = [{"role": "system", "content": system_prompt}]
                # Append last 6 messages of conversation context
                for msg in chat_history[-6:]:
                    messages.append({"role": msg["role"], "content": self._sanitize_text(msg["content"])})
                messages.append({"role": "user", "content": sanitized_query})

                response = self._call_llm_api(messages)
                if response:
                    return response
            except Exception:
                pass  # Fallback to local answering engine

        return self._generate_rule_based_chat_answer(sanitized_query, structured_summary)

    def _call_llm_api(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Make HTTP POST call to OpenAI-compatible endpoint."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1200
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=25)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        else:
            raise RuntimeError(f"LLM API returned status {resp.status_code}: {resp.text}")

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
        missing_list = summary.get("missing_values", [])
        outliers_list = summary.get("outliers", [])
        duplicates = summary.get("duplicates", {})

        note = ""
        if error_note:
            note = f"> *Note: Switched to deterministic diagnosis engine ({error_note}). Non-AI analysis remains 100% operational.*\n\n"
        elif not self.is_api_configured():
            note = "> *Note: Running in Rule-Based Diagnostic Engine Mode (no LLM API key configured). Real statistical insights generated below. Add an API key in Settings to activate interactive LLM completions.*\n\n"

        missing_text = ""
        if missing_list:
            missing_text = "\n".join([f"- **`{m['column']}`**: {m['null_count']} missing rows ({m['null_percentage']}%)" for m in missing_list])
        else:
            missing_text = "- No missing values detected across attributes (100% complete)."

        outlier_text = ""
        if outliers_list:
            outlier_text = "\n".join([f"- **`{o['column']}`**: {o['outlier_count']} outliers ({o['outlier_percentage']}%) outside range {o['bounds']}" for o in outliers_list])
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
            ("⚠️ **Moderate Readiness**: Preprocessing required. Missing values and distribution skews must be addressed before model training to avoid bias."
             if overall >= 65 else
             "🚨 **Not Ready for ML**: Severe data hygiene issues detected (missing data, duplicates, and unvalidated types). Training models directly will lead to high variance, data leakage, and degraded generalization.")
        )

        report = f"""{note}### 1. Executive Summary & Health Grade
The dataset **'{summary.get('dataset_name')}'** comprises **{dims.get('rows', 0)} rows** and **{dims.get('columns', 0)} attributes** ({dims.get('total_cells', 0)} total data points).

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
        missing = summary.get("missing_values", [])
        outliers = summary.get("outliers", [])
        duplicates = summary.get("duplicates", {})
        overall = scores.get("overall_score", 0)

        if any(term in q for term in ["wrong", "issue", "problem", "what is wrong"]):
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
                "4. **Outlier Treatment**: Cap extreme outliers to upper/lower IQR bounds.\n"
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
