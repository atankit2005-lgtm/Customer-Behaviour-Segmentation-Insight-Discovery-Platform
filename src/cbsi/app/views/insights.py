from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from cbsi.app.views.shared import (
    full_history_caption,
    insight_details,
    label,
    priority_order,
    show_table,
)

if TYPE_CHECKING:
    from cbsi.app.pipeline import AnalysisBundle


def render(bundle: AnalysisBundle) -> None:
    st.title("Business Insights")
    st.write("Translate measured customer behavior into testable, proportionate actions.")
    full_history_caption(bundle)
    st.info(
        "These full-history strategies are descriptive recommendations, not causal findings. "
        "They do not estimate uplift, profitability or guaranteed retention outcomes."
    )
    insights = priority_order(bundle.current.insights)
    show_table(
        insights.loc[
            :,
            [
                "segment_label",
                "customer_count",
                "strategy_category",
                "priority",
                "primary_objective",
            ],
        ],
        height=280,
    )
    profiles = bundle.current.profiles.summary.set_index("cluster")
    for _, insight in insights.iterrows():
        with st.container(border=True):
            st.subheader(insight["segment_label"])
            st.caption(
                f"Full-history segment · {label(insight['strategy_category'])} · "
                f"{insight['priority']} priority"
            )
            insight_details(insight, profiles.loc[insight["cluster"]])
    st.caption(
        "Recommendations, priorities, rationale and KPIs come from the existing insight engine. "
        "Historical test recommendations appear separately in Customer Explorer."
    )
