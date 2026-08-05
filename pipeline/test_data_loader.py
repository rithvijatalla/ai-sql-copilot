"""
Tests for pipeline/data_loader.py.

No API key or network needed - load_uploaded_files() is pure local file/DB
handling, so these run as plain pytest against real temp files, same as
guardrails/test_checks.py's static (non-LLM) checks.
"""

import io
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline.data_loader as data_loader_module
from pipeline.data_loader import load_uploaded_files, sanitize_identifier
from pipeline.schema_utils import get_table_names


class _FakeUploadedFile(io.BytesIO):
    """Minimal stand-in for a Streamlit UploadedFile: file-like (for
    pandas' read_csv/read_excel) plus a .name attribute."""

    def __init__(self, name: str, content: bytes):
        super().__init__(content)
        self.name = name


def _spy_on_mkstemp(monkeypatch) -> list[str]:
    """Patches data_loader's tempfile.mkstemp to record every path it
    hands out, while still calling through to the real mkstemp - so a test
    can assert on the exact temp file load_uploaded_files() created, even
    when the function raises before returning it."""
    created_paths: list[str] = []
    real_mkstemp = tempfile.mkstemp

    def spy(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        created_paths.append(path)
        return fd, path

    monkeypatch.setattr(data_loader_module.tempfile, "mkstemp", spy)
    return created_paths


def test_load_uploaded_files_loads_a_valid_csv():
    csv_file = _FakeUploadedFile("customers.csv", b"id,name\n1,Alice\n2,Bob\n")
    db_path = load_uploaded_files([csv_file])
    try:
        assert Path(db_path).exists()
        assert get_table_names(db_path) == ["customers"]
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_load_uploaded_files_deletes_temp_file_on_partial_failure(monkeypatch):
    """A malformed/unsupported file later in the list should not leak the
    temp db this call already created - and the original exception should
    still propagate, not get swallowed."""
    created_paths = _spy_on_mkstemp(monkeypatch)

    good_file = _FakeUploadedFile("good.csv", b"a,b\n1,2\n")
    bad_file = _FakeUploadedFile("bad.txt", b"not a csv or excel file")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_uploaded_files([good_file, bad_file])

    assert len(created_paths) == 1, "expected exactly one temp db to have been created"
    assert not Path(created_paths[0]).exists(), "temp db file should have been deleted after the failure"


def test_load_uploaded_files_deletes_temp_file_on_malformed_csv(monkeypatch):
    """Same guarantee for a failure that comes from pandas itself (a
    genuinely malformed CSV), not just the unsupported-extension path."""
    created_paths = _spy_on_mkstemp(monkeypatch)

    good_file = _FakeUploadedFile("good.csv", b"a,b\n1,2\n")
    malformed_file = _FakeUploadedFile("malformed.csv", b'"unterminated quote,a,b\n1,2,3\n')

    with pytest.raises(Exception):
        load_uploaded_files([good_file, malformed_file])

    assert len(created_paths) == 1
    assert not Path(created_paths[0]).exists()


def test_load_uploaded_files_multiple_valid_files():
    csv_file = _FakeUploadedFile("orders.csv", b"order_id,amount\n1,10.5\n")
    another_csv = _FakeUploadedFile("customers.csv", b"id,name\n1,Alice\n")
    db_path = load_uploaded_files([csv_file, another_csv])
    try:
        assert set(get_table_names(db_path)) == {"orders", "customers"}
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_sanitize_identifier_handles_special_characters():
    assert sanitize_identifier("Sales Data!") == "sales_data"


def test_sanitize_identifier_handles_leading_digit():
    assert sanitize_identifier("123abc") == "t_123abc"


def test_sanitize_identifier_deduplicates_against_existing():
    existing = {"sales"}
    assert sanitize_identifier("Sales", existing) == "sales_2"
