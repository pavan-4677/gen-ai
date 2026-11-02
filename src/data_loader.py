"""
Data Loader Module for Intelligent Dataset Diagnosis and Quality Analysis System.
Provides secure, robust ingestion for CSV and Excel files with encoding detection,
validation, and metadata extraction.
"""

import io
import os
from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd


class DataLoaderError(Exception):
    """Custom exception raised during dataset loading and validation."""
    pass


class DataLoader:
    """Robust dataset loader with format validation, encoding sniffing, and metadata calculation."""

    SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}

    def __init__(self, max_file_size_mb: int = 50):
        self.max_file_size_mb = max_file_size_mb
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def load_dataset(
        self,
        file_source: Union[str, io.BytesIO, Any],
        filename: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load dataset from a file path or a Streamlit UploadedFile object.
        
        Args:
            file_source: File path string or file-like object (e.g. UploadedFile).
            filename: Explicit file name if source is BytesIO.
            
        Returns:
            Tuple of (pd.DataFrame, metadata_dict)
            
        Raises:
            DataLoaderError if file is empty, invalid, exceeds size, or unsupported.
        """
        # Determine filename and size
        if isinstance(file_source, str):
            if not os.path.exists(file_source):
                raise DataLoaderError(f"File not found: {file_source}")
            fname = os.path.basename(file_source)
            file_size = os.path.getsize(file_source)
            file_bytes = None
        else:
            # File-like object (Streamlit UploadedFile or BytesIO)
            fname = getattr(file_source, "name", filename or "dataset")
            if hasattr(file_source, "size"):
                file_size = file_source.size
            else:
                # Seek to find size if possible
                try:
                    pos = file_source.tell()
                    file_source.seek(0, io.SEEK_END)
                    file_size = file_source.tell()
                    file_source.seek(pos)
                except Exception:
                    file_size = 0
            file_bytes = file_source

        # Validation 1: Empty file check
        if file_size == 0:
            raise DataLoaderError("The uploaded file is empty (0 bytes). Please upload a valid dataset.")

        # Validation 2: File size limit
        if file_size > self.max_file_size_bytes:
            raise DataLoaderError(
                f"File size exceeds limit of {self.max_file_size_mb} MB (Uploaded size: {file_size / (1024*1024):.2f} MB)."
            )

        # Validation 3: File extension
        ext = os.path.splitext(fname)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise DataLoaderError(
                f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}."
            )

        # Ingestion logic
        df = self._read_file(file_source, ext)

        if df is None or df.empty:
            raise DataLoaderError("The dataset contains 0 rows. Please provide a dataset with records.")

        if len(df.columns) == 0:
            raise DataLoaderError("The dataset does not contain any readable columns.")

        metadata = self._extract_metadata(df, fname, file_size)
        return df, metadata

    def _read_file(self, file_source: Union[str, Any], ext: str) -> pd.DataFrame:
        """Internal reader handling CSV encoding fallback and Excel parsing."""
        if ext in {".xlsx", ".xls"}:
            try:
                # Reset pointer if file-like
                if hasattr(file_source, "seek"):
                    file_source.seek(0)
                return pd.read_excel(file_source)
            except Exception as e:
                raise DataLoaderError(f"Failed to parse Excel file: {str(e)}")

        # CSV / TSV handling with encoding trials
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
        delimiter = "\t" if ext == ".tsv" else None  # Let pandas sniff if None or comma

        for enc in encodings:
            try:
                if hasattr(file_source, "seek"):
                    file_source.seek(0)
                
                # If delimiter is None, try sep=None with python engine for delimiter sniffing
                if delimiter is None:
                    try:
                        df = pd.read_csv(file_source, sep=None, engine="python", encoding=enc)
                    except Exception:
                        if hasattr(file_source, "seek"):
                            file_source.seek(0)
                        df = pd.read_csv(file_source, sep=",", encoding=enc)
                else:
                    df = pd.read_csv(file_source, sep=delimiter, encoding=enc)
                
                return df
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                # If it's a parse error, re-raise with friendly message
                raise DataLoaderError(f"Error parsing CSV file with encoding '{enc}': {str(e)}")

        raise DataLoaderError("Unable to decode the CSV file. Please ensure it is saved in UTF-8 or Latin-1 encoding.")

    def _extract_metadata(self, df: pd.DataFrame, filename: str, file_size: int) -> Dict[str, Any]:
        """Extract structured metadata from the DataFrame."""
        mem_bytes = df.memory_usage(deep=True).sum()
        mem_mb = mem_bytes / (1024 * 1024)

        return {
            "filename": filename,
            "file_size_bytes": file_size,
            "formatted_size": self._format_size(file_size),
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "column_names": [str(c) for c in df.columns],
            "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
            "memory_usage_mb": round(mem_mb, 4) if mem_mb > 0 else 0.0001,
            "numerical_columns": [str(c) for c in df.select_dtypes(include=["number"]).columns],
            "categorical_columns": [str(c) for c in df.select_dtypes(include=["object", "category", "string"]).columns],
            "datetime_columns": [str(c) for c in df.select_dtypes(include=["datetime"]).columns],
            "boolean_columns": [str(c) for c in df.select_dtypes(include=["bool"]).columns],
        }

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format byte size."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.2f} MB"

