from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from cbsi.app.views.shared import (
    full_history_caption,
    histogram,
    label,
    selection_details,
    show_table,
)
from cbsi.features.customer import NUMERIC_FEATURE_COLUMNS

if TYPE_CHECKING:
    from cbsi.app.pipeline import AnalysisBundle


def _heatmap(values: pd.DataFrame, statistic: str) -> None:
    st.subheader(f"Behavioral heatmap · {statistic}")
    figure, axis = plt.subplots(
        figsize=(max(7, len(values.columns) * 1.8), max(3.5, len(values) * 0.7))
    )
    try:
        image = axis.imshow(values.to_numpy(dtype=float), cmap="Blues", aspect="auto")
        axis.set_xticks(range(len(values.columns)), values.columns, rotation=25, ha="right")
        axis.set_yticks(range(len(values)), values.index)
        midpoint = (values.to_numpy().max() + values.to_numpy().min()) / 2
        for row in range(len(values)):
            for column in range(len(values.columns)):
                value = values.iloc[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:,.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > midpoint else "#172554",
                    fontsize=9,
                )
        figure.colorbar(image, ax=axis, shrink=0.75, label=statistic)
        figure.tight_layout()
        st.pyplot(figure)
    finally:
        plt.close(figure)


def _behavior_comparison(bundle: AnalysisBundle) -> None:
    st.subheader("Compare measured segment behavior")
    statistic = st.radio(
        "Statistic", ("Index", "Mean", "Median"), horizontal=True, key="segment_statistic"
    )
    features = st.multiselect(
        "Features to compare",
        NUMERIC_FEATURE_COLUMNS,
        default=["recency_days", "frequency", "monetary"],
        format_func=label,
        key="segment_comparison_features",
    )
    if not features:
        st.info("Select at least one numeric feature to compare segments.")
        return
    suffix = statistic.lower()
    summary = bundle.current.profiles.summary
    comparison = summary.loc[:, ["segment_label", *[f"{item}_{suffix}" for item in features]]]
    values = comparison.set_index("segment_label").rename(
        columns={f"{feature}_{suffix}": label(feature) for feature in features}
    )
    show_table(values.reset_index(), height=300)
    chart = values.reset_index().melt(
        id_vars="segment_label", var_name="Feature", value_name="Value"
    )
    st.bar_chart(chart, x="segment_label", y="Value", color="Feature", horizontal=True, stack=False)
    _heatmap(values, statistic)
    if statistic == "Index":
        st.caption(
            "Index = segment mean / overall customer mean. 1.00 is the population baseline; "
            "recency below 1 means more recent purchasing, not weaker engagement."
        )
    else:
        st.caption(
            "Means and medians retain each feature's units. Color and bar magnitudes across "
            "different units are not directly comparable; use Index for relative comparisons."
        )


def render(bundle: AnalysisBundle) -> None:
    st.title("Customer Segmentation")
    st.write("Explainable full-history profiles, backed by multi-algorithm validation.")
    full_history_caption(bundle)
    selection_details(bundle.current.selection, key="current_candidate_algorithms")
    st.subheader("Full segment summary")
    st.caption(
        "All numeric means, medians and relative indices are included. Characteristics are "
        "rendered as readable text. Cluster IDs are technical identifiers, not business labels."
    )
    show_table(bundle.current.profiles.summary, height=450)
    _behavior_comparison(bundle)
    feature = st.selectbox(
        "Customer behavior distribution",
        NUMERIC_FEATURE_COLUMNS,
        format_func=label,
        key="segmentation_histogram_feature",
    )
    histogram(bundle.current.profiles.customers[feature], f"Full history · {label(feature)}")
    st.caption("All eligible full-history customers; raw units, including distribution tails.")
