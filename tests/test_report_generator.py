"""
Tests for Report Generator Module.
"""

import json
import pandas as pd
import pytest
from src.report_generator import ReportGenerator


@pytest.fixture
def mock_diagnosis():
    return {
        "dataset_name": "DemoReportDataset.csv",
        "profile": {
            "overview": {
                "total_rows": 50,
                "total_columns": 4,
                "total_cells": 200,
                "total_missing_cells": 5,
                "missing_percentage": 2.5,
                "duplicate_rows_count": 0,
                "duplicate_percentage": 0.0,
                "numerical_columns_count": 2,
                "categorical_columns_count": 2,
                "datetime_columns_count": 0,
                "boolean_columns_count": 0
            }
        },
        "quality_scores": {
            "overall_score": 92.5,
            "grade": "A (Excellent)",
            "status": "Ready for Modeling",
            "dimensions": {
                "completeness": {"score": 97.5, "missing_cells": 5},
                "uniqueness": {"score": 100.0, "duplicate_rows": 0},
                "validity": {"score": 95.0, "invalid_values_count": 0},
                "consistency": {"score": 92.0, "inconsistency_points": 0}
            }
        },
        "missing_analysis": {"column_breakdown": [{"column": "col1", "missing_count": 5, "missing_percentage": 10.0, "severity": "Low"}]},
        "duplicate_analysis": {"duplicate_rows_count": 0},
        "outlier_analysis": {"columns_with_outliers": []},
        "dtype_analysis": {"column_types": {"col1": "float64"}},
        "detected_issues": []
    }


def test_generate_json_report(mock_diagnosis):
    gen = ReportGenerator()
    json_str = gen.generate_json_report(mock_diagnosis)
    parsed = json.loads(json_str)

    assert parsed["metadata"]["dataset_name"] == "DemoReportDataset.csv"
    assert parsed["quality_scores"]["overall_score"] == 92.5
    assert "version" in parsed["metadata"]


def test_generate_pdf_report(mock_diagnosis):
    gen = ReportGenerator()
    pdf_bytes = gen.generate_pdf_report(
        diagnosis=mock_diagnosis,
        cleaning_history=[{"action": "Impute Missing", "details": "Imputed 5 values", "rows_before": 50, "rows_after": 50, "rows_delta": 0}],
        ai_explanation="Test AI summary and analysis insights."
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")


def test_export_csv():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    csv_bytes = ReportGenerator.export_csv(df)
    assert isinstance(csv_bytes, bytes)
    assert b"a,b" in csv_bytes


def test_export_excel():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    excel_bytes = ReportGenerator.export_excel(df)
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 100
