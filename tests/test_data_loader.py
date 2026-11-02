"""
Tests for Data Loader Module.
"""

import io
import os
import pytest
import pandas as pd
from src.data_loader import DataLoader, DataLoaderError


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a temporary valid CSV file."""
    fpath = tmp_path / "valid.csv"
    fpath.write_text("id,name,val\n1,Alice,100\n2,Bob,200\n3,Charlie,300\n")
    return str(fpath)


@pytest.fixture
def sample_excel_file(tmp_path):
    """Create a temporary valid Excel file."""
    fpath = tmp_path / "valid.xlsx"
    df = pd.DataFrame({"id": [1, 2], "score": [95.5, 88.0]})
    df.to_excel(str(fpath), index=False)
    return str(fpath)


def test_load_valid_csv(sample_csv_file):
    loader = DataLoader()
    df, meta = loader.load_dataset(sample_csv_file)
    assert len(df) == 3
    assert list(df.columns) == ["id", "name", "val"]
    assert meta["total_rows"] == 3
    assert meta["total_columns"] == 3
    assert meta["filename"] == "valid.csv"


def test_load_valid_excel(sample_excel_file):
    loader = DataLoader()
    df, meta = loader.load_dataset(sample_excel_file)
    assert len(df) == 2
    assert list(df.columns) == ["id", "score"]
    assert meta["total_rows"] == 2


def test_load_empty_file_fails(tmp_path):
    loader = DataLoader()
    empty_f = tmp_path / "empty.csv"
    empty_f.write_text("")
    with pytest.raises(DataLoaderError, match="empty"):
        loader.load_dataset(str(empty_f))


def test_unsupported_format(tmp_path):
    loader = DataLoader()
    unsupported = tmp_path / "test.json"
    unsupported.write_text('{"a": 1}')
    with pytest.raises(DataLoaderError, match="Unsupported file format"):
        loader.load_dataset(str(unsupported))


def test_file_size_exceeded(tmp_path):
    # Set limit to 1 KB for testing
    loader = DataLoader(max_file_size_mb=0.001)
    large_f = tmp_path / "large.csv"
    large_f.write_text("x," * 1000 + "\n" + "1," * 1000)
    with pytest.raises(DataLoaderError, match="File size exceeds limit"):
        loader.load_dataset(str(large_f))


def test_load_bytes_io():
    loader = DataLoader()
    data = b"colA,colB\n10,20\n30,40\n"
    bio = io.BytesIO(data)
    setattr(bio, "name", "memory.csv")
    df, meta = loader.load_dataset(bio)
    assert len(df) == 2
    assert meta["total_columns"] == 2
