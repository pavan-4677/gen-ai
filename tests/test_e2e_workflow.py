"""
Comprehensive End-to-End Workflow Test Suite.
Verifies all 12 operational capabilities requested:
1. CSV and Excel uploads
2. Dataset preview
3. Missing value detection
4. Duplicate detection
5. Outlier detection
6. Quality score calculation
7. Interactive charts generation
8. Gen AI assistant modules
9. Data cleaning operations
10. Before-and-after comparison
11. PDF and Excel report downloads
12. Session state & navigation simulation
"""

import io
import os
import pytest
import pandas as pd
import numpy as np

from src.data_loader import DataLoader
from src.quality_analyzer import QualityAnalyzer
from src.outlier_detector import OutlierDetector
from src.quality_scoring import QualityScoringEngine
from src.data_cleaner import DataCleaner
from src.ai_service import AIService
from src.report_generator import ReportGenerator
from src.utils import apply_plotly_theme
import plotly.express as px
import plotly.graph_objects as go


class TestEndToEndWorkflow:
    """Rigorous end-to-end verification across the entire data diagnosis lifecycle."""

    @pytest.fixture(autouse=True)
    def setup_paths(self):
        self.sample_csv_path = "sample_data/sample_dataset.csv"
        self.sample_xlsx_path = "sample_data/sample_dataset.xlsx"
        self.churn_csv_path = "sample_data/customer_churn_dirty.csv"

    # 1. Check CSV and Excel Uploads
    def test_01_csv_and_excel_ingestion(self):
        loader = DataLoader()
        
        # Test real CSV
        df_csv, meta_csv = loader.load_dataset(self.sample_csv_path)
        assert not df_csv.empty
        assert meta_csv["total_rows"] == 30
        assert meta_csv["total_columns"] == 10
        assert meta_csv["filename"] == "sample_dataset.csv"
        assert len(meta_csv["column_names"]) == 10

        # Test real Excel
        df_xlsx, meta_xlsx = loader.load_dataset(self.sample_xlsx_path)
        assert not df_xlsx.empty
        assert meta_xlsx["total_rows"] == 30
        assert meta_xlsx["total_columns"] == 10
        assert meta_xlsx["filename"] == "sample_dataset.xlsx"

        # Test BytesIO simulation of file upload
        with open(self.sample_csv_path, "rb") as f:
            bio = io.BytesIO(f.read())
            bio.name = "uploaded_demo.csv"
        df_bio, meta_bio = loader.load_dataset(bio)
        assert len(df_bio) == 30

    # 2. Check Dataset Preview
    def test_02_dataset_preview(self):
        loader = DataLoader()
        df, meta = loader.load_dataset(self.sample_csv_path)

        preview_rows = df.head(10)
        assert len(preview_rows) == 10
        assert "CustomerID" in preview_rows.columns
        assert "Age" in preview_rows.columns
        assert "AnnualIncome" in preview_rows.columns

        # Verify dtypes extracted
        assert "Age" in meta["dtypes"]
        assert meta["memory_usage_mb"] > 0

    # 3. Check Missing Value Detection
    def test_03_missing_value_detection(self):
        loader = DataLoader()
        df, _ = loader.load_dataset(self.sample_csv_path)
        analyzer = QualityAnalyzer()
        diag = analyzer.analyze(df, "sample_dataset.csv")

        miss = diag["missing_analysis"]
        assert miss["total_missing_cells"] > 0
        assert miss["rows_with_missing"] > 0
        
        # Specifically Emily Davis row has null Age, David Wilson has null AnnualIncome, Ava Robinson has null SpendingScore
        null_cols = [c["column"] for c in miss["column_breakdown"]]
        assert "Age" in null_cols
        assert "AnnualIncome" in null_cols
        assert "SpendingScore" in null_cols

    # 4. Check Duplicate Detection
    def test_04_duplicate_detection(self):
        loader = DataLoader()
        df, _ = loader.load_dataset(self.sample_csv_path)
        analyzer = QualityAnalyzer()
        diag = analyzer.analyze(df, "sample_dataset.csv")

        dups = diag["duplicate_analysis"]
        assert dups["duplicate_rows_count"] >= 2  # Michael Brown and David Wilson rows are duplicated
        assert dups["duplicate_percentage"] > 0
        assert len(dups["duplicate_indices"]) >= 2

    # 5. Check Outlier Detection (IQR Method)
    def test_05_outlier_detection(self):
        loader = DataLoader()
        df, _ = loader.load_dataset(self.sample_csv_path)
        detector = OutlierDetector(iqr_multiplier=1.5)
        outlier_res = detector.detect_outliers_iqr(df)

        assert outlier_res["method"] == "IQR"
        assert outlier_res["outlier_rows_count"] > 0
        
        # Age has Daniel Martin (150) and William Lewis (-5)
        # AnnualIncome has Alexander Walker (980,000)
        assert "Age" in outlier_res["columns_with_outliers"] or "AnnualIncome" in outlier_res["columns_with_outliers"]
        
        # Check outlier row extraction
        outlier_rows = detector.get_outlier_rows(df, outlier_res["affected_row_indices"])
        assert len(outlier_rows) == outlier_res["outlier_rows_count"]

    # 6. Check Quality Score Calculation
    def test_06_quality_score_calculation(self):
        loader = DataLoader()
        df, _ = loader.load_dataset(self.sample_csv_path)
        analyzer = QualityAnalyzer()
        diag = analyzer.analyze(df, "sample_dataset.csv")
        scores = diag["quality_scores"]

        assert 0.0 <= scores["overall_score"] <= 100.0
        dims = scores["dimensions"]
        assert 0.0 <= dims["completeness"]["score"] <= 100.0
        assert 0.0 <= dims["uniqueness"]["score"] <= 100.0
        assert 0.0 <= dims["validity"]["score"] <= 100.0
        assert 0.0 <= dims["consistency"]["score"] <= 100.0
        assert scores["grade"] in ["A (Excellent)", "B (Good)", "C (Fair)", "D (Poor)"]

    # 7. Check Interactive Charts Generation
    def test_07_interactive_charts_generation(self):
        loader = DataLoader()
        df, _ = loader.load_dataset(self.sample_csv_path)
        analyzer = QualityAnalyzer()
        diag = analyzer.analyze(df, "sample_dataset.csv")
        scores = diag["quality_scores"]

        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=scores["overall_score"],
            gauge={'axis': {'range': [0, 100]}}
        ))
        assert fig_gauge is not None

        # Missing Values Bar Chart
        miss_data = diag["missing_analysis"]["column_breakdown"]
        if miss_data:
            fig_miss = px.bar(pd.DataFrame(miss_data), x="column", y="missing_percentage")
            apply_plotly_theme(fig_miss)
            assert fig_miss.layout.plot_bgcolor == "#0F172A"
            assert fig_miss.layout.paper_bgcolor == "rgba(0,0,0,0)"

        # Correlation Heatmap
        corr_matrix = diag["profile"]["correlation_matrix"]
        if corr_matrix:
            fig_corr = px.imshow(pd.DataFrame(corr_matrix))
            apply_plotly_theme(fig_corr)
            assert fig_corr is not None

    # 8. Check Gen AI Assistant Modules
    def test_08_gen_ai_assistant_modules(self):
        loader = DataLoader()
        df, _ = loader.load_dataset(self.sample_csv_path)
        analyzer = QualityAnalyzer()
        diag = analyzer.analyze(df, "sample_dataset.csv")
        ai = AIService(api_key="")  # Verify deterministic fallback mode

        # AI Summary
        summary = ai.generate_dataset_summary(diag)
        assert "sample_dataset.csv" in summary
        assert "records" in summary

        # AI Anomaly Explanations
        anomalies = ai.explain_anomalies(diag)
        assert "missing_values_explanation" in anomalies
        assert "duplicates_explanation" in anomalies
        assert "outliers_explanation" in anomalies

        # AI Cleaning Recommendations
        recs = ai.generate_cleaning_recommendations(diag)
        assert "Deduplication" in recs or "Imputation" in recs or "Winsorization" in recs

        # ML Readiness Assessment
        ml_eval = ai.assess_ml_readiness(diag)
        assert ml_eval["status"] in ["HIGH", "MODERATE", "LOW"]
        assert "tree_based" in ml_eval["model_suitability"]

        # Chatbot Q&A
        chat_q1 = ai.answer_user_query("What is wrong with my dataset?", [], diag)
        assert len(chat_q1) > 20
        chat_q2 = ai.answer_user_query("Is this dataset ready for machine learning?", [], diag)
        assert "score" in chat_q2 or "machine learning" in chat_q2

    # 9. Check Data Cleaning Operations
    def test_09_data_cleaning_operations(self):
        loader = DataLoader()
        df, _ = loader.load_dataset(self.sample_csv_path)
        cleaner = DataCleaner()

        # Deduplication
        df_dedup, rec_dup = cleaner.remove_duplicates(df)
        assert len(df_dedup) < len(df)
        assert rec_dup["rows_delta"] < 0

        # Imputation
        df_imp, rec_imp = cleaner.impute_missing(df_dedup, "Age", strategy="median")
        assert df_imp["Age"].isna().sum() == 0

        # Winsorization (Outlier Capping)
        df_cap, rec_cap = cleaner.cap_outliers_iqr(df_imp, "AnnualIncome", multiplier=1.5)
        assert df_cap["AnnualIncome"].max() < df["AnnualIncome"].max()

        # Text Standardization
        df_text, rec_text = cleaner.standardize_text(df_cap, "Gender", strip_whitespace=True, casing="title")
        assert df_text["Gender"].nunique() <= 2  # Standardized to 'Male' and 'Female'

        # Placeholder replacement
        df_place, rec_place = cleaner.replace_placeholders(df_text, placeholders=["?"])
        assert (df_place["Education"] == "?").sum() == 0

    # 10. Check Before-and-After Comparison
    def test_10_before_and_after_comparison(self):
        loader = DataLoader()
        df_raw, _ = loader.load_dataset(self.sample_csv_path)
        cleaner = DataCleaner()
        analyzer = QualityAnalyzer()

        # Raw diagnosis
        diag_raw = analyzer.analyze(df_raw, "sample_dataset.csv")
        raw_score = diag_raw["quality_scores"]["overall_score"]
        raw_nulls = diag_raw["profile"]["overview"]["total_missing_cells"]

        # Run cleaning sequence
        df_clean, _ = cleaner.remove_duplicates(df_raw)
        df_clean, _ = cleaner.replace_placeholders(df_clean, placeholders=["?"])
        df_clean, _ = cleaner.impute_missing(df_clean, "Age", strategy="median")
        df_clean, _ = cleaner.impute_missing(df_clean, "AnnualIncome", strategy="median")
        df_clean, _ = cleaner.impute_missing(df_clean, "SpendingScore", strategy="median")
        df_clean, _ = cleaner.impute_missing(df_clean, "Education", strategy="mode")
        df_clean, _ = cleaner.cap_outliers_iqr(df_clean, "AnnualIncome", multiplier=1.5)
        df_clean, _ = cleaner.standardize_text(df_clean, "Gender", strip_whitespace=True, casing="title")

        # Cleaned diagnosis
        diag_clean = analyzer.analyze(df_clean, "Cleaned_sample_dataset.csv")
        clean_score = diag_clean["quality_scores"]["overall_score"]
        clean_nulls = diag_clean["profile"]["overview"]["total_missing_cells"]

        # Compare Before vs After
        assert clean_score >= raw_score
        assert clean_nulls <= raw_nulls
        assert diag_clean["duplicate_analysis"]["duplicate_rows_count"] == 0

    # 11. Check PDF and Excel Report Downloads
    def test_11_report_generation_downloads(self):
        loader = DataLoader()
        df, _ = loader.load_dataset(self.sample_csv_path)
        analyzer = QualityAnalyzer()
        diag = analyzer.analyze(df, "sample_dataset.csv")
        rep_gen = ReportGenerator()

        # PDF Report
        pdf_bytes = rep_gen.generate_pdf_report(
            diagnosis=diag,
            cleaning_history=[{"action": "Deduplicate", "details": "Removed duplicates", "rows_before": 30, "rows_after": 28, "rows_delta": -2}],
            ai_explanation="Test AI Insights"
        )
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 2000
        assert pdf_bytes.startswith(b"%PDF")

        # Excel Export
        excel_bytes = rep_gen.export_excel(df)
        assert isinstance(excel_bytes, bytes)
        assert len(excel_bytes) > 1000

        # CSV Export
        csv_bytes = rep_gen.export_csv(df)
        assert isinstance(csv_bytes, bytes)
        assert b"CustomerID,Name,Age" in csv_bytes

        # JSON Export
        json_str = rep_gen.generate_json_report(diag)
        assert "sample_dataset.csv" in json_str
        assert "quality_scores" in json_str

    # 12. Check Navigation and Session State Simulation
    def test_12_session_state_preservation(self):
        loader = DataLoader()
        df, meta = loader.load_dataset(self.sample_csv_path)
        analyzer = QualityAnalyzer()
        diag = analyzer.analyze(df, "sample_dataset.csv")

        # Simulate Streamlit session state dictionary
        session = {
            "df_raw": df.copy(),
            "df_clean": df.copy(),
            "active_dataset_name": "sample_dataset.csv",
            "dataset_metadata": meta,
            "diagnosis_results": diag,
            "quality_scores": diag["quality_scores"],
            "cleaning_history": [],
            "ai_chat_history": [],
            "settings": {"openai_api_key": "", "iqr_threshold": 1.5}
        }

        # Simulate page transitions
        # Page 2 (Upload) -> sets state
        assert session["df_raw"] is not None
        # Page 3 (Diagnosis) -> accesses state
        assert session["diagnosis_results"]["profile"]["overview"]["total_rows"] == 30
        # Page 6 (Cleaning) -> mutates clean state with audit
        cleaner = DataCleaner()
        new_df, rec = cleaner.remove_duplicates(session["df_clean"])
        session["df_clean"] = new_df
        session["cleaning_history"].append(rec)
        # Verify raw state was NOT mutated
        assert len(session["df_raw"]) == 30
        assert len(session["df_clean"]) < 30
        assert len(session["cleaning_history"]) == 1
        # Page 7 (Report) -> reads both raw and clean
        assert session["df_clean"] is not session["df_raw"]
