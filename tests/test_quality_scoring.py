"""
Tests for Quality Scoring Engine.
"""

import pandas as pd
import pytest
from src.quality_scoring import QualityScoringEngine


def test_perfect_dataset_score():
    """A clean, full dataset should score 100 on completeness, uniqueness, and near 100 overall."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Ethan"],
        "age": [25, 30, 35, 40, 45]
    })
    engine = QualityScoringEngine()
    result = engine.calculate_scores(df)

    assert result["overall_score"] >= 95.0
    assert result["dimensions"]["completeness"]["score"] == 100.0
    assert result["dimensions"]["uniqueness"]["score"] == 100.0
    assert result["grade"].startswith("A")


def test_missing_data_penalty():
    """Completeness score should decrease linearly with missing cells."""
    # 10 cells, 2 nulls -> 80% completeness
    df = pd.DataFrame({
        "a": [1, None, 3, 4, 5],
        "b": [10, 20, 30, None, 50]
    })
    engine = QualityScoringEngine()
    result = engine.calculate_scores(df)

    assert result["dimensions"]["completeness"]["score"] == 80.0
    assert result["dimensions"]["completeness"]["missing_cells"] == 2


def test_duplicate_data_penalty():
    """Uniqueness score should penalize duplicate rows."""
    # 4 rows, 2 exact duplicates -> 50% uniqueness
    df = pd.DataFrame({
        "a": [1, 1, 2, 2],
        "b": ["x", "x", "y", "y"]
    })
    engine = QualityScoringEngine()
    result = engine.calculate_scores(df)

    assert result["dimensions"]["uniqueness"]["score"] == 50.0
    assert result["dimensions"]["uniqueness"]["duplicate_rows"] == 2


def test_validity_placeholder_penalty():
    """Validity score should identify sentinel placeholder strings."""
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "?"],
        "notes": ["None", "ok", "NA"]
    })
    engine = QualityScoringEngine()
    result = engine.calculate_scores(df)

    assert result["dimensions"]["validity"]["invalid_values_count"] >= 3
    assert result["dimensions"]["validity"]["score"] < 100.0


def test_empty_dataset_handling():
    """Engine should handle empty datasets safely without zero-division exceptions."""
    df = pd.DataFrame()
    engine = QualityScoringEngine()
    result = engine.calculate_scores(df)

    assert result["overall_score"] == 0.0
    assert result["status"] == "Empty Dataset"


def test_custom_weights():
    """Custom weights should adjust the calculated overall score."""
    df = pd.DataFrame({
        "a": [1, None, 3],  # 66.6% complete
        "b": [10, 20, 30]   # 100% unique
    })
    # Heavily weight uniqueness (1.0 vs 0.0)
    engine = QualityScoringEngine(weights={"completeness": 0.0, "uniqueness": 1.0, "validity": 0.0, "consistency": 0.0})
    result = engine.calculate_scores(df)

    assert result["overall_score"] == 100.0
