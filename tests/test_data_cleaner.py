"""
Tests for Data Cleaner Module.
"""

import pandas as pd
import numpy as np
import pytest
from src.data_cleaner import DataCleaner


@pytest.fixture
def dirty_df():
    return pd.DataFrame({
        "id": [1, 2, 2, 4, 5],
        "score": [10.0, np.nan, np.nan, 40.0, 500.0],  # Missing + outlier
        "city": [" New York ", "chicago", "CHICAGO", "new york", "Boston"],
        "tag": ["?", "ok", "valid", "null", "pass"]
    })


def test_remove_duplicates():
    df = pd.DataFrame({
        "id": [1, 2, 2, 4],
        "name": ["A", "B", "B", "C"]
    })
    cleaner = DataCleaner()
    df_clean, rec = cleaner.remove_duplicates(df)

    assert len(df_clean) == 3
    assert rec["rows_delta"] == -1
    assert rec["action"] == "Remove Duplicates"


def test_remove_duplicates_subset(dirty_df):
    cleaner = DataCleaner()
    df_clean, rec = cleaner.remove_duplicates(dirty_df, subset=["id"])

    assert len(df_clean) == 4
    assert rec["rows_delta"] == -1


def test_impute_missing_median(dirty_df):
    cleaner = DataCleaner()
    # Scores without NaNs: [10.0, 40.0, 500.0] -> median is 40.0
    df_clean, rec = cleaner.impute_missing(dirty_df, "score", strategy="median")

    assert df_clean["score"].isna().sum() == 0
    assert rec["cells_modified"] == 2
    assert df_clean.loc[1, "score"] == 40.0


def test_impute_missing_mode():
    df = pd.DataFrame({"category": ["A", "B", "A", None]})
    cleaner = DataCleaner()
    df_clean, rec = cleaner.impute_missing(df, "category", strategy="mode")

    assert df_clean["category"].isna().sum() == 0
    assert df_clean.loc[3, "category"] == "A"


def test_drop_missing_rows(dirty_df):
    cleaner = DataCleaner()
    df_clean, rec = cleaner.drop_missing_rows(dirty_df, columns=["score"])

    assert len(df_clean) == 3
    assert rec["rows_delta"] == -2


def test_cap_outliers_iqr():
    df = pd.DataFrame({"val": [10.0, 12.0, 11.0, 13.0, 10.0, 100.0]})
    cleaner = DataCleaner()
    df_clean, rec = cleaner.cap_outliers_iqr(df, "val", multiplier=1.5)

    assert df_clean["val"].max() < 100.0
    assert len(df_clean) == 6  # Preserves row count


def test_standardize_text():
    df = pd.DataFrame({"city": [" new york ", "NEW YORK", "New York"]})
    cleaner = DataCleaner()
    df_clean, rec = cleaner.standardize_text(df, "city", strip_whitespace=True, casing="title")

    assert df_clean["city"].nunique() == 1
    assert df_clean["city"].iloc[0] == "New York"


def test_replace_placeholders(dirty_df):
    cleaner = DataCleaner()
    df_clean, rec = cleaner.replace_placeholders(dirty_df, columns=["tag"])

    # '?' and 'null' should become NaN
    assert df_clean["tag"].isna().sum() == 2


def test_convert_data_type():
    df = pd.DataFrame({"price": ["$10.50", "$20.00", "$30.25"]})
    cleaner = DataCleaner()
    df_clean, rec = cleaner.convert_data_type(df, "price", target_type="float")

    assert pd.api.types.is_float_dtype(df_clean["price"])
    assert df_clean["price"].iloc[0] == 10.50
