from __future__ import annotations

from typing import TYPE_CHECKING
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

import streamlit as st

if TYPE_CHECKING:
    from cbsi.app.pipeline import AnalysisBundle

PAGES = (
    "Executive Overview",
    "Customer Segmentation",
    "Business Insights",
    "Churn Analysis",
    "Customer Explorer",
)
EXPECTED_ERRORS = (OSError, ValueError, ImportError, BadZipFile, ParseError, EOFError)


def _run_analysis() -> None:
    from cbsi.app.state import cached_analysis, clear_analysis, source_fingerprint

    clear_analysis()
    try:
        fingerprint = source_fingerprint(st.session_state["dataset_path"])
        with st.spinner("Preparing transactions, comparing segments and validating churn models…"):
            bundle = cached_analysis(*fingerprint)
        if source_fingerprint(st.session_state["dataset_path"]) != fingerprint:
            raise ValueError("The workbook changed during analysis. Run analysis again.")
    except EXPECTED_ERRORS as error:
        st.session_state["analysis_error"] = error
        return
    st.session_state["analysis_bundle"] = bundle
    st.session_state["analysis_fingerprint"] = fingerprint


def _sidebar_status(bundle: AnalysisBundle | None) -> None:
    st.divider()
    st.subheader("Analysis status")
    if bundle is None:
        st.caption("Not ready · submit a workbook to begin.")
        return
    from cbsi.app.views.shared import model_label

    st.success("Ready · source verified")
    st.caption(f"Full-history segmentation: {model_label(bundle.current.selection.selected.model)}")
    st.caption(
        f"Historical segmentation: {model_label(bundle.historical.selection.selected.model)}"
    )
    st.caption(f"Churn model: {model_label(bundle.churn_selection.selected.model_name)}")
    st.metric("Full-history customers", f"{bundle.metadata.customer_count:,}")
    st.metric("Full-history segments", str(bundle.current.selection.selected.n_clusters))
    st.metric("Historical test customers", f"{bundle.churn_split.test.customer_count:,}")
    st.metric("Historical test inactivity", f"{bundle.churn_split.test.prevalence:.1%}")
    st.caption("Full-history and historical segments are benchmarked independently.")


def _readiness() -> None:
    st.title("Customer Behaviour Intelligence")
    st.write("From transaction history to explainable customer segments and retention priorities.")
    st.info("No analysis runs until you select Run analysis in the sidebar.")
    with st.container(border=True):
        st.subheader("Get your workspace ready")
        st.write("1. Enter an existing Online Retail II **.xlsx** workbook path.")
        st.write("2. Include both sheets: **Year 2009-2010** and **Year 2010-2011**.")
        st.write("3. Run analysis, then explore the five pages using the sidebar.")
        st.caption(
            "The first run compares three segmentation algorithms and three churn models. "
            "Enough transaction history and both active/inactive target classes are required. "
            "Results are cached by workbook path, size and modification time."
        )
    st.caption("Monetary values use workbook price units; no currency conversion is applied.")


def _render_page(page: str, bundle: AnalysisBundle) -> None:
    from cbsi.app.views import churn, explorer, insights, overview, segmentation

    renderers = {
        "Executive Overview": overview.render,
        "Customer Segmentation": segmentation.render,
        "Business Insights": insights.render,
        "Churn Analysis": churn.render,
        "Customer Explorer": explorer.render,
    }
    renderers[page](bundle)


def main() -> None:
    st.set_page_config(
        page_title="Customer Behaviour Intelligence",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    from cbsi.app.state import initialize_state, invalidate_stale_analysis

    initialize_state()
    with st.sidebar:
        st.title("Customer Intelligence")
        st.caption("Segmentation · strategy · retention")
        with st.form("dataset_form"):
            st.text_input("Workbook path (.xlsx)", key="dataset_path")
            submitted = st.form_submit_button("Run analysis", type="primary")
        page = st.radio("Workspace", PAGES, key="page")
    stale = invalidate_stale_analysis()
    if submitted:
        _run_analysis()
    bundle = st.session_state["analysis_bundle"]
    try:
        with st.sidebar:
            _sidebar_status(bundle)
        if stale and not submitted:
            st.warning("The workbook source changed. Previous results were cleared; run again.")
        error = st.session_state["analysis_error"]
        if error is not None:
            st.error("Analysis could not complete. Check the workbook, its data and dependencies.")
            st.exception(error)
        if bundle is None:
            _readiness()
        else:
            _render_page(page, bundle)
    except ImportError as error:
        st.error("A dashboard dependency is unavailable in this Python environment.")
        st.exception(error)


if __name__ == "__main__":
    main()
