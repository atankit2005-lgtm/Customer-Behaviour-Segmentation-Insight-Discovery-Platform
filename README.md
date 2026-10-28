# Customer-Behaviour-Segmentation-Insight-Discovery-Platform
End-to-end customer segmentation &amp; insight platform. Engineers RFM &amp; behavioral features, compares K-Means, Hierarchical, and DBSCAN using multi-metric validation, profiles segments with actionable business strategies, and benchmarks unsupervised clustering against supervised churn prediction.

## Integrated analytics dashboard

Requires Python 3.14 and the existing `data`, `ml`, and `app` dependency groups.
From the repository root on Windows:

```bash
.venv/Scripts/python.exe -m streamlit run src/cbsi/app/main.py
```

The sidebar defaults to `data/raw/online_retail_II.xlsx`. Supply an existing Excel
workbook with both `Year 2009-2010` and `Year 2010-2011` sheets, then select
**Run analysis**. Initial ingestion and both clustering benchmarks can take several
minutes; subsequent navigation and filters reuse session-safe cached results.
Changing the workbook path or its file metadata invalidates the cached analysis.

Five pages cover executive KPIs, customer segmentation and behavior, rule-based
business insights, temporal churn benchmarking, and a filtered customer explorer.
`cbsi.app.pipeline` orchestrates the existing backend; the Streamlit views only
present its structured outputs.

Full-history profiles and recommendations are separate from historical churn
results. The explorer and segment-by-churn comparisons use observation-only
segments aligned to the final test cutoff. Churn means no qualifying purchase in
the following 90 days, not permanent departure; scores are historical holdout
predictions, not live production scoring. Recommendations are descriptive, and
native feature importance is not causal evidence.
