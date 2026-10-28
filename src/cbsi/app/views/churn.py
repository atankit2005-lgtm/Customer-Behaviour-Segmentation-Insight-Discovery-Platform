from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from cbsi.app.views.shared import (
    bar_chart,
    histogram,
    historical_caption,
    metrics,
    model_label,
    selection_details,
    show_table,
)
from cbsi.churn.analysis import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD, RISK_BANDS
from cbsi.churn.dataset import TARGET_DEFINITION

if TYPE_CHECKING:
    from cbsi.app.pipeline import AnalysisBundle


def _temporal_design(bundle: AnalysisBundle) -> None:
    st.subheader("A chronological, historical validation design")
    historical_caption(bundle)
    st.write(
        f"Target: no qualifying purchase in the next {bundle.config.prediction_window_days} days. "
        "This is an inactivity proxy, not evidence that a customer has permanently left."
    )
    st.caption(TARGET_DEFINITION)
    show_table(bundle.temporal_summary, height=180)
    st.caption(
        "Features use purchases at or before each observation cutoff. Training labels finish "
        "before validation features; validation labels finish before test features. Models "
        "are selected on validation, then refitted on train + validation and evaluated on test. "
        "Customer identities can recur across snapshots."
    )
    with st.expander("Interpretation and limitations"):
        st.write(
            "This is a retrospective holdout, not live future scoring. New customers after the "
            "test cutoff are not in this cohort. Seasonality, incomplete purchase visibility "
            "and changing behavior can affect results. Risk bands are operational probability "
            "thresholds, not calibrated loss estimates or proof of responsiveness to an offer."
        )


def _model_validation(bundle: AnalysisBundle) -> None:
    selection = bundle.churn_selection
    st.subheader(f"Selected churn model · {model_label(selection.selected.model_name)}")
    st.caption(f"Classification threshold: {selection.selected.model.threshold:.2f}")
    for reason in selection.rationale:
        st.write(reason)
    st.markdown("**Validation comparison · all three models**")
    show_table(selection.comparison, height=200)
    test = bundle.test_metrics
    st.subheader("Final historical test performance")
    metrics(
        (
            ("ROC-AUC", f"{test.roc_auc:.3f}"),
            ("PR-AUC (average precision)", f"{test.pr_auc:.3f}"),
            ("Inactive prevalence", f"{test.positive_prevalence:.1%}"),
            ("Precision · inactive class", f"{test.precision:.3f}"),
            ("Recall · inactive class", f"{test.recall:.3f}"),
            ("F1 · inactive class", f"{test.f1:.3f}"),
        )
    )
    st.subheader("Confusion matrix · historical test")
    matrix = pd.DataFrame(
        test.confusion_matrix,
        index=["Actual 0 · active", "Actual 1 · inactive"],
        columns=["Predicted 0 · active", "Predicted 1 · inactive"],
    )
    st.dataframe(matrix, width="stretch")
    st.caption(
        "Rows = observed outcomes; columns = model predictions. Class 0 means a qualifying "
        "purchase in the prediction window; class 1 means no qualifying purchase."
    )


def _feature_importance(bundle: AnalysisBundle) -> None:
    st.subheader("Model feature importance")
    importance = bundle.feature_importance
    st.bar_chart(importance, x="feature", y="importance", color="direction", horizontal=True)
    show_table(importance, height=340)
    st.caption(
        "Logistic regression shows absolute standardized coefficients with increases_risk / "
        "decreases_risk direction. Tree models show native unsigned importance; not_available "
        "means direction cannot be inferred. These associations are model-specific, not causal "
        "drivers or estimates of the effect of changing customer behavior."
    )


def _risk_distribution(bundle: AnalysisBundle) -> None:
    st.subheader("Historical test · risk distribution")
    distribution = bundle.risk_distribution.set_index("risk_band")
    columns = st.columns(len(RISK_BANDS))
    ranges = {
        "High": f"{HIGH_RISK_THRESHOLD:.0%} ≤ probability ≤ 100%",
        "Medium": f"{MEDIUM_RISK_THRESHOLD:.0%} ≤ probability < {HIGH_RISK_THRESHOLD:.0%}",
        "Low": f"0% ≤ probability < {MEDIUM_RISK_THRESHOLD:.0%}",
    }
    for column, band in zip(columns, RISK_BANDS, strict=True):
        with column, st.container(border=True):
            st.metric(f"{band} risk", f"{int(distribution.loc[band, 'count']):,}")
            st.caption(f"{distribution.loc[band, 'share']:.1%} of test cohort · {ranges[band]}")
    histogram(
        bundle.predictions["churn_probability"],
        "Historical test · predicted inactivity probability",
        probability=True,
    )
    st.caption("Risk-band thresholds differ from the model's binary classification threshold.")


def _historical_segments(bundle: AnalysisBundle) -> None:
    historical_caption(bundle)
    bar_chart(
        bundle.segment_churn,
        "segment_label",
        "churn_rate",
        "Historical segments · observed inactivity rate",
    )
    st.caption("Rates use a 0–1 chart scale. Observed inactivity is distinct from predicted risk.")
    show_table(bundle.segment_churn, height=330)
    with st.expander("Historical segmentation model, benchmark and rationale"):
        st.write(
            "This segmentation is benchmarked independently using exactly the nine numerical "
            "features of the test snapshot. No post-cutoff purchasing informs membership. "
            "Country is joined from historical observations. Do not equate these labels or "
            "cluster identifiers with the full-history segmentation."
        )
        selection_details(bundle.historical.selection, key="historical_candidate_algorithms")


def render(bundle: AnalysisBundle) -> None:
    st.title("Churn Analysis")
    st.write(
        "Evaluate a historical inactivity signal, its validation quality and cohort-level risk."
    )
    _temporal_design(bundle)
    _model_validation(bundle)
    _feature_importance(bundle)
    _risk_distribution(bundle)
    _historical_segments(bundle)
