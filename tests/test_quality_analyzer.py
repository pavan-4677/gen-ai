"""
Tests for Quality Analyzer Module.
"""

import pandas as pd
import pytest
from src.quality_analyzer import QualityAnalyzer


@pytest.fixture
def dirty_sample_df():
    return pd.DataFrame({
        "id": [101, 102, 103, 103, 104],
        "name": ["John", " Jane ", "Bob", "Bob", "Alice"],
        "age": [25, 30, -5, -5, 150],  # Negative age and outlier
        "salary": ["$50,000", "$60,000", "$70,000", "$70,000", "None"],  # String currency & placeholder
        "constant_col": ["Fixed", "Fixed", "Fixed", "Fixed", "Fixed"]
    })


def test_full_analysis_pipeline(dirty_sample_df):
    analyzer = QualityAnalyzer()
    res = analyzer.analyze(dirty_sample_df, dataset_name="DirtyData")

    # Check that overview was compiled
    assert res["profile"]["overview"]["total_rows"] == 5
    assert res["profile"]["overview"]["total_columns"] == 5

    # Check duplicate detection (row 103 is duplicate)
    assert res["duplicate_analysis"]["duplicate_rows_count"] == 1

    # Check constant column
    constant_cols = [c["column"] for c in res["cardinality_analysis"]["constant_columns"]]
    assert "constant_col" in constant_cols

    # Check type mismatch on salary (object contains currency)
    warnings = res["dtype_analysis"]["type_mismatch_warnings"]
    assert any(w["column"] == "salary" for w in warnings)

    # Check negative value warning on age
    neg_issues = res["invalid_values_analysis"]["negative_value_issues"]
    assert any(n["column"] == "age" for n in neg_issues)

    # Check issues compiled
    assert res["total_issues_count"] > 0
    assert "quality_scores" in res

