"""
Tests for Outlier Detector Module.
"""

import numpy as np
import pandas as pd
import pytest
from src.outlier_detector import OutlierDetector


@pytest.fixture
def outlier_dataframe():
    """DataFrame with clear outliers in salary."""
    # Normal salaries around 50,000, plus one extreme outlier at 1,000,000
    data = {
        "id": range(1, 11),
        "salary": [48000, 52000, 50000, 51000, 49000, 53000, 49500, 51500, 50500, 1000000]
    }
    return pd.DataFrame(data)


def test_iqr_outlier_detection(outlier_dataframe):
    detector = OutlierDetector(iqr_multiplier=1.5)
    res = detector.detect_outliers_iqr(outlier_dataframe, columns=["salary"])

    assert "salary" in res["columns_with_outliers"]
    rep = res["column_reports"]["salary"]
    assert rep["outlier_count"] == 1
    assert 9 in rep["outlier_indices"]  # Index of 1,000,000
    assert rep["sample_outliers"] == [1000000]
    assert res["outlier_rows_count"] == 1


def test_iqr_bounds_math():
    # Symmetric data: 10, 20, 30, 40, 50
    # Q1=20, Q3=40, IQR=20. Lower = 20 - 1.5*20 = -10. Upper = 40 + 30 = 70.
    df = pd.DataFrame({"val": [10, 20, 30, 40, 50, 200]})
    detector = OutlierDetector(iqr_multiplier=1.5)
    res = detector.detect_outliers_iqr(df)
    rep = res["column_reports"]["val"]

    assert rep["outlier_count"] == 1
    assert rep["upper_bound"] < 200


def test_get_outlier_rows(outlier_dataframe):
    detector = OutlierDetector()
    res = detector.detect_outliers_iqr(outlier_dataframe)
    indices = res["affected_row_indices"]
    rows = detector.get_outlier_rows(outlier_dataframe, indices)

    assert len(rows) == 1
    assert rows.iloc[0]["salary"] == 1000000


def test_zscore_detection(outlier_dataframe):
    detector = OutlierDetector(z_threshold=2.0)
    res = detector.detect_outliers_zscore(outlier_dataframe, columns=["salary"])
    assert res["column_reports"]["salary"]["outlier_count"] >= 1

