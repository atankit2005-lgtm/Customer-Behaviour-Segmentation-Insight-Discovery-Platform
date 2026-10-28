from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from cbsi.app.pipeline import filter_scored_customers
from cbsi.app.views.shared import (
    behavioral_metrics,
    historical_caption,
    insight_details,
    metrics,
    show_table,
)
from cbsi.churn.analysis import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD, RISK_BANDS
from cbsi.features.customer import NUMERIC_FEATURE_COLUMNS

if TYPE_CHECKING:
    from cbsi.app.pipeline import AnalysisBundle

DISPLAY_COLUMNS = (
    "customer_id",
    "segment_label",
    "country",
    "risk_band",
    "churn_probability",
    "actual_churn",
    "churn_prediction",
    *NUMERIC_FEATURE_COLUMNS,
)


def _filters(bundle: AnalysisBundle) -> pd.DataFrame:
    customers = bundle.scored_customers
    with st.container(border=True):
        st.subheader("Narrow the historical cohort")
        left, middle, right = st.columns(3)
        with left:
            segment_options = sorted(customers["segment_label"].dropna().unique().tolist())
            segments = st.multiselect(
                "Historical segments",
                segment_options,
                default=segment_options,
                key="explorer_segments",
            )
        with middle:
            country_options = sorted(customers["country"].dropna().unique().tolist())
            countries = st.multiselect(
                "Countries", country_options, default=country_options, key="explorer_countries"
            )
        with right:
            risks = st.multiselect(
                "Risk bands", RISK_BANDS, default=list(RISK_BANDS), key="explorer_risk_bands"
            )
        probability = st.slider(
            "Predicted inactivity probability range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.01,
            format="%.2f",
            key="explorer_probability",
        )
        search = st.text_input(
            "Customer ID contains", key="explorer_search", help="Literal substring, not a regex."
        )
        st.caption(
            "Empty segment, country or risk selections match no customers. "
            f"High ≥ {HIGH_RISK_THRESHOLD:.0%}; medium ≥ {MEDIUM_RISK_THRESHOLD:.0%} and "
            f"< {HIGH_RISK_THRESHOLD:.0%}; low < {MEDIUM_RISK_THRESHOLD:.0%}."
        )
    return filter_scored_customers(
        customers,
        segment_labels=tuple(segments),
        countries=tuple(countries),
        risk_bands=tuple(risks),
        probability_range=(float(probability[0]), float(probability[1])),
        customer_id=search,
    )


def _customer_detail(bundle: AnalysisBundle, customer: pd.Series) -> None:
    st.subheader(f"Customer {customer['customer_id']}")
    st.write(f"**Historical segment:** {customer['segment_label']}")
    st.caption(
        f"Country: {customer['country']} · technical historical cluster ID: {customer['cluster']}"
    )
    metrics(
        (
            ("Historical predicted inactivity", f"{customer['churn_probability']:.1%}"),
            ("Risk band", str(customer["risk_band"])),
            ("Observed test outcome", "Inactive" if customer["actual_churn"] == 1 else "Active"),
        )
    )
    st.caption(
        f"Model classification: {'Inactive' if customer['churn_prediction'] == 1 else 'Active'}. "
        "Outcomes describe the historical prediction window, not today's customer status."
    )
    st.markdown("**Behavior as of the test observation cutoff**")
    behavioral_metrics(customer)
    st.caption(
        "Recency is measured relative to the observation cutoff + one day. Order counts are "
        "distinct customer–invoice pairs. All numeric features are reused from the test snapshot."
    )
    profile = bundle.historical.profiles.summary.set_index("cluster").loc[customer["cluster"]]
    insight = bundle.historical.insights.set_index("cluster").loc[customer["cluster"]]
    with st.container(border=True):
        st.subheader("Historical segment · strategy and evidence")
        insight_details(insight, profile)
        st.caption(
            "This strategy is based on historical segment behavior, not an individualized "
            "treatment effect. Actions should be tested; no causal benefit is implied."
        )


def render(bundle: AnalysisBundle) -> None:
    st.title("Customer Explorer")
    st.write("Inspect scored customers without mixing historical features with later purchases.")
    historical_caption(bundle)
    st.info(
        "Historical test customers only. Segment membership, country and behavior use the "
        "test observation cutoff. Full-history profiles include later purchases and may assign "
        "the same customer differently; their labels are not interchangeable."
    )
    filtered = _filters(bundle)
    st.subheader(f"{len(filtered):,} matching customers")
    st.caption(f"From {bundle.churn_split.test.customer_count:,} historical test customers.")
    if filtered.empty:
        st.session_state.pop("selected_customer", None)
        st.info(
            "No customers match these filters. Widen the probability range or adjust selections."
        )
        return
    bounded = filtered.sort_values("churn_probability", ascending=False, kind="stable").head(100)
    show_table(bounded.loc[:, list(DISPLAY_COLUMNS)], height=420, limit=100)
    st.caption(
        f"Showing {len(bounded):,} of {len(filtered):,} matches, highest predicted risk first. "
        "Customer selection is limited to these 100 rows; narrow the filters or search to "
        "inspect another customer. Actual/predicted churn: 0 = active, 1 = inactive."
    )
    choices = bounded["customer_id"].astype(str).tolist()
    if st.session_state.get("selected_customer") not in choices:
        st.session_state.pop("selected_customer", None)
    selected = st.selectbox("Inspect customer", choices, key="selected_customer")
    customer = bounded.loc[bounded["customer_id"].astype(str).eq(selected)].iloc[0]
    _customer_detail(bundle, customer)
