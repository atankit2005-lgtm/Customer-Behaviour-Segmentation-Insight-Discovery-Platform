from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from cbsi.app.pipeline import AnalysisBundle

SourceFingerprint = tuple[str, int, int]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORKBOOK = PROJECT_ROOT / "data" / "raw" / "online_retail_II.xlsx"
EXPLORER_KEYS = (
    "explorer_segments",
    "explorer_countries",
    "explorer_risk_bands",
    "explorer_probability",
    "explorer_search",
    "selected_customer",
)


def initialize_state() -> None:
    st.session_state.setdefault("dataset_path", str(DEFAULT_WORKBOOK))
    st.session_state.setdefault("analysis_bundle", None)
    st.session_state.setdefault("analysis_fingerprint", None)
    st.session_state.setdefault("analysis_error", None)


def source_fingerprint(workbook_path: str) -> SourceFingerprint:
    if not workbook_path.strip():
        raise ValueError("Enter the path to an existing .xlsx transaction workbook.")
    path = Path(workbook_path.strip()).expanduser()
    if path.suffix.lower() != ".xlsx":
        raise ValueError("Only .xlsx transaction workbooks are supported.")
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve(strict=True)
    if path.suffix.lower() != ".xlsx":
        raise ValueError("The resolved workbook must also be an .xlsx file.")
    if not path.is_file():
        raise ValueError("The workbook path must identify a file, not a directory.")
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns


@st.cache_data(max_entries=2, show_spinner=False)
def cached_analysis(path: str, size: int, mtime_ns: int) -> AnalysisBundle:
    from cbsi.app.pipeline import run_analysis

    fingerprint = (path, size, mtime_ns)
    if source_fingerprint(path) != fingerprint:
        raise ValueError("The workbook changed before analysis. Submit it again.")
    bundle = run_analysis(path)
    if source_fingerprint(path) != fingerprint:
        raise ValueError("The workbook changed during analysis. Submit it again.")
    return bundle


def clear_analysis() -> None:
    st.session_state["analysis_bundle"] = None
    st.session_state["analysis_fingerprint"] = None
    st.session_state["analysis_error"] = None
    for key in EXPLORER_KEYS:
        st.session_state.pop(key, None)


def invalidate_stale_analysis() -> bool:
    previous = st.session_state["analysis_fingerprint"]
    if previous is None:
        return False
    try:
        current = source_fingerprint(st.session_state["dataset_path"])
    except OSError, ValueError:
        clear_analysis()
        return True
    if previous != current:
        clear_analysis()
        return True
    return False
