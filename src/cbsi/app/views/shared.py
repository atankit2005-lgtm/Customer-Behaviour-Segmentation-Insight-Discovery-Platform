from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pandas.api.types import is_float_dtype, is_integer_dtype

from cbsi.features.customer import NUMERIC_FEATURE_COLUMNS
from cbsi.profiling.insights import PRIORITIES

if TYPE_CHECKING:
    from cbsi.app.pipeline import AnalysisBundle
    from cbsi.clustering.selection import ClusteringSelection

ACCENT = "#2563EB"
FEATURE_LABELS = {
    "recency_days": "Recency (days)",
    "frequency": "Orders",
    "monetary": "Monetary value",
    "total_quantity": "Items purchased",
    "active_days": "Active purchase days",
    "avg_order_value": "Average order value",
    "avg_items_per_invoice": "Items per order",
    "unique_products": "Unique products",
    "purchase_span_days": "Purchase span (days)",
}


def label(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("_", " ").capitalize())


def model_label(name: str) -> str:
    return {
        "kmeans": "K-Means",
        "agglomerative": "Agglomerative",
        "dbscan": "DBSCAN",
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
    }.get(name, label(name))


def date_label(value: pd.Timestamp) -> str:
    return value.strftime("%d %b %Y, %H:%M")


def full_history_caption(bundle: AnalysisBundle) -> None:
    metadata = bundle.metadata
    st.caption(
        f"Full-history eligible purchases · {date_label(metadata.first_transaction)} to "
        f"{date_label(metadata.last_transaction)} · {metadata.customer_count:,} customers. "
        "Monetary values are in workbook price units, not profit or lifetime value."
    )


def historical_caption(bundle: AnalysisBundle) -> None:
    test = bundle.churn_split.test
    st.caption(
        f"Historical test cohort · {test.customer_count:,} customers observed through "
        f"{date_label(test.observation_end)} · outcomes through {date_label(test.prediction_end)}. "
        "Historical membership is independent of the full-history segmentation."
    )


def metrics(items: Sequence[tuple[str, str]], columns: int = 3) -> None:
    for offset in range(0, len(items), columns):
        for column, (title, value) in zip(
            st.columns(columns), items[offset : offset + columns], strict=False
        ):
            with column:
                with st.container(border=True):
                    st.metric(title, value)


def readable_value(value: object) -> object:
    if isinstance(value, Mapping):
        return ", ".join(f"{key}={item}" for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return "; ".join(str(item).replace("_", " ") for item in value) or "None"
    return value


def show_table(frame: pd.DataFrame, *, height: int = 350, limit: int = 300) -> None:
    if frame.empty:
        st.info("No records available for this view.")
        return
    display = frame.head(limit).copy(deep=True)
    formats = {}
    percent_columns = {
        "positive_prevalence",
        "prevalence",
        "churn_rate",
        "churn_probability",
        "average_churn_probability",
        "share",
    }
    for name in display.columns:
        if name == "customer_id":
            display[name] = display[name].map(
                lambda value: (
                    str(int(value))
                    if isinstance(value, float) and value.is_integer()
                    else str(value)
                )
            )
        elif display[name].dtype == object:
            display[name] = display[name].map(readable_value)
        if name in percent_columns or name.endswith(("_share", "_fraction")):
            formats[name] = "{:.1%}"
        elif is_integer_dtype(display[name]):
            formats[name] = "{:,}"
        elif is_float_dtype(display[name]):
            formats[name] = "{:,.3f}" if name.endswith(("_score", "_index")) else "{:,.2f}"
    st.dataframe(
        display.style.format(formats, na_rep="Not available"),
        hide_index=True,
        height=height,
        width="stretch",
        column_config={name: label(name) for name in display.columns},
    )
    if len(frame) > limit:
        st.caption(f"Showing the first {limit:,} of {len(frame):,} records.")


def bar_chart(frame: pd.DataFrame, category: str, value: str, title: str) -> None:
    st.subheader(title)
    if frame.empty:
        st.info("No chart data available.")
        return
    chart = frame.loc[:, [category, value]].rename(columns={category: label(category)})
    chart = chart.rename(columns={value: label(value)})
    st.bar_chart(chart, x=label(category), y=label(value), color=ACCENT, horizontal=True)


def histogram(values: pd.Series, title: str, *, probability: bool = False) -> None:
    st.subheader(title)
    if values.empty:
        st.info("No observations available.")
        return
    figure, axis = plt.subplots(figsize=(9, 3.6))
    try:
        axis.hist(values, bins=25, color=ACCENT, edgecolor="white", linewidth=0.6)
        axis.set_xlabel(title)
        axis.set_ylabel("Customers")
        if probability:
            axis.set_xlim(0, 1)
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        st.pyplot(figure)
    finally:
        plt.close(figure)


def selection_details(selection: ClusteringSelection, *, key: str) -> None:
    selected = selection.selected
    st.subheader(f"Selected model · {model_label(selected.model)}")
    st.caption(f"Parameters: {readable_value(selected.parameters)}")
    metrics(
        (
            ("Non-noise segments", str(selected.n_clusters)),
            ("Coverage", f"{selected.coverage_fraction:.1%}"),
            ("Noise customers", f"{selected.noise_points:,} ({selected.noise_fraction:.1%})"),
            ("Silhouette ↑", f"{selected.silhouette_score:.3f}"),
            ("Calinski–Harabasz ↑", f"{selected.calinski_harabasz_score:,.2f}"),
            ("Davies–Bouldin ↓", f"{selected.davies_bouldin_score:.3f}"),
        )
    )
    st.caption(
        f"Balance (smallest/largest): {selected.balance_score:.3f} · "
        f"Cluster size ratio: {selected.cluster_size_ratio:.2f} · "
        f"Smallest/largest non-noise sizes: {selected.min_cluster_size:,}/"
        f"{selected.max_cluster_size:,}. Noise is retained in profiles, not counted as a segment."
    )
    for reason in selection.rationale:
        st.write(reason)
    with st.expander("Eligibility rules"):
        criteria = selection.criteria
        st.write(
            f"At least {criteria.min_clusters} non-noise clusters; noise ≤ "
            f"{criteria.max_noise_fraction:.0%}; minimum assigned-cluster share ≥ "
            f"{criteria.min_cluster_fraction:.0%}; size ratio ≤ "
            f"{criteria.max_cluster_size_ratio:.1f}. All three metrics must be valid."
        )
    with st.expander("All candidate configurations and rejection reasons"):
        st.caption("K-Means, Agglomerative and DBSCAN · eligible and rejected configurations.")
        algorithms = tuple(selection.comparison["model"].unique())
        chosen = st.multiselect(
            "Algorithms", algorithms, default=list(algorithms), format_func=model_label, key=key
        )
        show_table(selection.comparison.loc[selection.comparison["model"].isin(chosen)], height=450)


def priority_order(insights: pd.DataFrame) -> pd.DataFrame:
    return insights.sort_values(
        "priority",
        key=lambda values: values.map({priority: rank for rank, priority in enumerate(PRIORITIES)}),
        kind="stable",
    )


def bullet_list(title: str, values: Sequence[str]) -> None:
    st.markdown(f"**{title}**")
    for value in values:
        st.write(f"• {value}")


def insight_details(insight: pd.Series, profile: pd.Series) -> None:
    st.write(profile["description"])
    bullet_list("Measured characteristics", profile["characteristics"])
    st.markdown(f"**Objective:** {insight['primary_objective']}")
    st.caption(
        f"Strategy: {label(insight['strategy_category'])} · Priority: {insight['priority']} · "
        f"{int(insight['customer_count']):,} customers ({insight['customer_share']:.1%})"
    )
    bullet_list("Recommended actions", insight["recommended_actions"])
    left, right = st.columns(2)
    with left:
        bullet_list("Risks to monitor", insight["key_risks"])
    with right:
        bullet_list("Opportunities to test", insight["key_opportunities"])
    st.markdown("**Evidence and rationale**")
    st.write(insight["rationale"])
    bullet_list("Recommended KPIs", [label(value) for value in insight["recommended_kpis"]])


def behavioral_metrics(customer: pd.Series) -> None:
    count_features = {
        "recency_days",
        "frequency",
        "active_days",
        "unique_products",
        "purchase_span_days",
    }
    metrics(
        [
            (
                label(feature),
                f"{customer[feature]:,.0f}"
                if feature in count_features
                else f"{customer[feature]:,.2f}",
            )
            for feature in NUMERIC_FEATURE_COLUMNS
        ]
    )
