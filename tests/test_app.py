from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from pandas.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest

from cbsi.app import state
from cbsi.app.pipeline import AnalysisBundle

APP_FILE = Path(__file__).resolve().parents[1] / "src" / "cbsi" / "app" / "main.py"


@pytest.fixture
def dashboard(
    analysis_bundle: AnalysisBundle, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> AppTest:
    workbook = tmp_path / "retail.xlsx"
    workbook.write_bytes(b"Synthetic workbook path for mocked UI tests")
    monkeypatch.setattr(state, "cached_analysis", lambda *args, **kwargs: deepcopy(analysis_bundle))
    app = AppTest.from_file(str(APP_FILE)).run(timeout=30)
    app.text_input(key="dataset_path").set_value(str(workbook))
    next(button for button in app.button if button.label == "Run analysis").click()
    return app.run(timeout=30)


def test_application_import_is_side_effect_free() -> None:
    from cbsi.app.main import main

    assert callable(main)


def test_customer_ids_display_as_identifiers_without_mutating_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    from cbsi.app.views import shared

    frame = pd.DataFrame({"customer_id": [12347.0, "00123", "U-000", 12347.5]})
    before = frame.copy(deep=True)
    captured: list[pd.DataFrame] = []
    monkeypatch.setattr(
        shared.st, "dataframe", lambda styled, **kwargs: captured.append(styled.data)
    )
    shared.show_table(frame)
    assert captured[0]["customer_id"].tolist() == ["12347", "00123", "U-000", "12347.5"]
    assert_frame_equal(frame, before)


def test_initial_dashboard_explains_readiness_without_training() -> None:
    app = AppTest.from_file(str(APP_FILE)).run(timeout=30)
    assert not app.exception
    assert app.text_input(key="dataset_path").value.endswith("online_retail_II.xlsx")
    assert any(button.label == "Run analysis" for button in app.button)
    assert app.info


def test_all_dashboard_pages_render_without_exceptions(dashboard: AppTest) -> None:
    navigation = dashboard.radio(key="page")
    assert len(navigation.options) == 5
    for page in navigation.options:
        dashboard.radio(key="page").set_value(page).run(timeout=30)
        assert not dashboard.exception, page
        assert not dashboard.error, page
        assert dashboard.title or dashboard.header, page


def test_ui_navigation_does_not_recompute_pipeline(
    dashboard: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_recompute(*args: object) -> AnalysisBundle:
        raise AssertionError("Navigation must reuse the session bundle")

    monkeypatch.setattr(state, "cached_analysis", unexpected_recompute)
    for page in dashboard.radio(key="page").options:
        dashboard.radio(key="page").set_value(page).run(timeout=30)
        assert not dashboard.exception
        assert not dashboard.error


def test_explorer_filters_search_and_customer_selection(dashboard: AppTest) -> None:
    dashboard.radio(key="page").set_value("Customer Explorer").run(timeout=30)
    dashboard.text_input(key="explorer_search").set_value("U-000").run(timeout=30)
    assert dashboard.selectbox(key="selected_customer").options == ["U-000"]
    assert not dashboard.exception
    dashboard.text_input(key="explorer_search").set_value(".*").run(timeout=30)
    assert any("No customers match" in message.value for message in dashboard.info)
    assert not dashboard.exception
    dashboard.text_input(key="explorer_search").set_value("")
    dashboard.multiselect(key="explorer_risk_bands").set_value([]).run(timeout=30)
    assert any("No customers match" in message.value for message in dashboard.info)
    assert not dashboard.exception


def test_empty_feature_selection_is_an_informative_state(dashboard: AppTest) -> None:
    dashboard.radio(key="page").set_value("Customer Segmentation").run(timeout=30)
    dashboard.multiselect(key="segment_comparison_features").set_value([]).run(timeout=30)
    assert any("Select at least one" in message.value for message in dashboard.info)
    assert not dashboard.exception


def test_views_do_not_mutate_the_session_bundle(dashboard: AppTest) -> None:
    before = deepcopy(dashboard.session_state["analysis_bundle"])
    for page in dashboard.radio(key="page").options:
        dashboard.radio(key="page").set_value(page).run(timeout=30)
    after = dashboard.session_state["analysis_bundle"]
    assert_frame_equal(before.predictions, after.predictions)
    assert_frame_equal(before.current.profiles.summary, after.current.profiles.summary)
    assert_frame_equal(before.historical.profiles.customers, after.historical.profiles.customers)


def test_changed_source_clears_previous_results(dashboard: AppTest) -> None:
    workbook = Path(dashboard.text_input(key="dataset_path").value)
    workbook.write_bytes(b"Changed synthetic source fingerprint")
    dashboard.run(timeout=30)
    assert dashboard.warning
    assert not dashboard.metric
    assert dashboard.session_state["analysis_bundle"] is None


def test_missing_workbook_has_useful_error_without_results(tmp_path: Path) -> None:
    app = AppTest.from_file(str(APP_FILE)).run(timeout=30)
    app.text_input(key="dataset_path").set_value(str(tmp_path / "missing.xlsx"))
    next(button for button in app.button if button.label == "Run analysis").click()
    app.run(timeout=30)
    assert app.error
    assert not app.metric


def test_non_workbook_paths_are_rejected_without_reading(tmp_path: Path) -> None:
    app = AppTest.from_file(str(APP_FILE)).run(timeout=30)
    app.text_input(key="dataset_path").set_value(str(tmp_path / ".env"))
    next(button for button in app.button if button.label == "Run analysis").click()
    app.run(timeout=30)
    assert app.error
    assert not app.metric


def test_pipeline_error_is_visible_and_does_not_show_stale_results(
    dashboard: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed_pipeline(*args: object) -> AnalysisBundle:
        raise ValueError("Churn target must contain both binary classes")

    monkeypatch.setattr(state, "cached_analysis", failed_pipeline)
    next(button for button in dashboard.button if button.label == "Run analysis").click()
    dashboard.run(timeout=30)
    assert dashboard.error
    assert not dashboard.metric


def test_cache_isolation_and_file_fingerprint_invalidation(
    analysis_bundle: AnalysisBundle, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cbsi.app.pipeline as pipeline

    calls: list[str] = []
    workbook = tmp_path / "cache-test.xlsx"
    workbook.write_bytes(b"Synthetic cache fixture")
    fingerprint = state.source_fingerprint(str(workbook))

    def fake_run(path: str) -> AnalysisBundle:
        calls.append(path)
        return deepcopy(analysis_bundle)

    monkeypatch.setattr(pipeline, "run_analysis", fake_run)
    state.cached_analysis.clear()
    try:
        first = state.cached_analysis(*fingerprint)
        first.predictions.loc[:, "churn_probability"] = 0
        second = state.cached_analysis(*fingerprint)
        assert_frame_equal(second.predictions, analysis_bundle.predictions)
        assert calls == [str(workbook)]
        workbook.write_bytes(b"Changed synthetic cache fixture")
        state.cached_analysis(*state.source_fingerprint(str(workbook)))
        assert calls == [str(workbook), str(workbook)]
    finally:
        state.cached_analysis.clear()
