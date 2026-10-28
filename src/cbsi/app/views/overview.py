from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from cbsi.app.views.shared import (
    bar_chart,
    full_history_caption,
    historical_caption,
    metrics,
    priority_order,
    show_table,
)

if TYPE_CHECKING:
    from cbsi.app.pipeline import AnalysisBundle


def _headline_metrics(bundle: AnalysisBundle) -> None:
    metadata = bundle.metadata
    high_risk = bundle.risk_distribution.set_index("risk_band").loc["High", "count"]
    metrics(
        (
            ("Full history · customers", f"{metadata.customer_count:,}"),
            ("Full history · monetary value", f"{metadata.monetary_total:,.2f}"),
            ("Full history · orders", f"{metadata.invoice_count:,}"),
            (
                "Full history · non-noise segments",
                str(bundle.current.selection.selected.n_clusters),
            ),
            ("Historical test · inactivity rate", f"{bundle.churn_split.test.prevalence:.1%}"),
            ("Historical test · high-risk customers", f"{int(high_risk):,}"),
        )
    )
    st.caption(
        "Orders count distinct customer–invoice pairs. Inactivity is an observed historical "
        "target; high risk is a model probability band, not a confirmed future loss."
    )


def _eligibility(bundle: AnalysisBundle) -> None:
    metadata = bundle.metadata
    with st.expander("Purchase eligibility and order economics"):
        metrics(
            (
                ("Raw transaction rows", f"{metadata.raw_transaction_count:,}"),
                ("Eligible transaction rows", f"{metadata.eligible_transaction_count:,}"),
                ("Excluded transaction rows", f"{metadata.excluded_transaction_count:,}"),
                ("Raw identified customers", f"{metadata.raw_customer_count:,}"),
                ("Eligible customers", f"{metadata.customer_count:,}"),
                ("Average order value", f"{metadata.average_order_value:,.2f}"),
            )
        )
        st.caption(
            f"Total eligible quantity: {metadata.total_quantity:,.2f}. Eligibility removes "
            "duplicates, cancellations, invalid purchases and missing customer identifiers. "
            "The excluded count is the complete funnel difference, not a reason-level breakdown."
        )


def _segment_charts(bundle: AnalysisBundle) -> None:
    summary = bundle.current.profiles.summary
    left, right = st.columns(2)
    with left:
        bar_chart(summary, "segment_label", "customer_count", "Full history · segment size")
    with right:
        bar_chart(
            summary, "segment_label", "monetary_total", "Full history · monetary contribution"
        )
    st.subheader("Full history · customer share and monetary contribution")
    st.scatter_chart(
        summary,
        x="customer_share",
        y="monetary_total",
        size="customer_count",
        color="segment_label",
    )
    st.caption(
        "X: customer share (0–1) · Y: total monetary value · bubble size: customers. "
        "Labels are behavior-derived; similarly worded labels can describe separate clusters."
    )
    historical_caption(bundle)
    bar_chart(
        bundle.segment_churn,
        "segment_label",
        "average_churn_probability",
        "Historical test · predicted inactivity by historical segment",
    )
    st.caption(
        "Chart probabilities use a 0–1 scale and the historical cohort, "
        "not full-history membership."
    )


def _focus_panels(bundle: AnalysisBundle) -> None:
    summary = bundle.current.profiles.summary
    largest = summary.loc[summary["customer_count"].idxmax()]
    valuable = summary.loc[summary["monetary_mean"].idxmax()]
    risk = bundle.segment_churn.loc[bundle.segment_churn["average_churn_probability"].idxmax()]
    left, middle, right = st.columns(3)
    with left, st.container(border=True):
        st.subheader("Largest · full history")
        st.write(largest["segment_label"])
        st.caption(
            f"{int(largest['customer_count']):,} customers · {largest['customer_share']:.1%}"
        )
    with middle, st.container(border=True):
        st.subheader("Highest value · full history")
        st.write(valuable["segment_label"])
        st.caption(f"Mean monetary value per customer: {valuable['monetary_mean']:,.2f}")
    with right, st.container(border=True):
        st.subheader("Highest risk · historical test")
        st.write(risk["segment_label"])
        st.caption(
            f"Average predicted inactivity probability: {risk['average_churn_probability']:.1%}"
        )
    st.subheader("Full history · opportunities to prioritize")
    top = priority_order(bundle.current.insights).head(3)
    show_table(
        top.loc[
            :,
            [
                "segment_label",
                "priority",
                "primary_objective",
                "key_opportunities",
                "recommended_actions",
                "rationale",
            ],
        ],
        height=230,
    )
    st.caption(
        "Priority is supplied by the descriptive insight engine, not an estimated treatment "
        "effect or ROI. Validate actions with measured tests before scaling."
    )


def render(bundle: AnalysisBundle) -> None:
    st.title("Executive Overview")
    st.write("Understand the customer base, value concentration and historical retention signals.")
    full_history_caption(bundle)
    _headline_metrics(bundle)
    _eligibility(bundle)
    _segment_charts(bundle)
    _focus_panels(bundle)
