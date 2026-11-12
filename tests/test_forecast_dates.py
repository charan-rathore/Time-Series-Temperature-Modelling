"""Forecast dates must come from the data the model actually saw.

The served numbers are `horizon` steps after the last row of the processed
dataset (lgbm/tft) or after an unrecorded SARIMA fit end. Dating them
today+1..+3 presented weeks-old output as "tomorrow".
"""

from datetime import date, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, load_config


class FakeManager:
    def __init__(self, model, through):
        self.model, self.through = model, through
        self.artifact_hashes = {}

    def data_through(self):
        return self.through

    def forecast(self, days=3):
        return [
            {"predicted_temp_c": 24.0, "lower_bound_c": 23.0, "upper_bound_c": 25.0, "model_used": self.model}
            for _ in range(days)
        ]


@pytest.fixture
def get_forecast():
    app.state.config = load_config()
    saved = getattr(app.state, "model_manager", None)

    def run(model, through):
        app.state.model_manager = FakeManager(model, through)
        return TestClient(app, raise_server_exceptions=False).get("/forecast?days=3").json()

    yield run
    if saved is not None:
        app.state.model_manager = saved
    elif hasattr(app.state, "model_manager"):
        del app.state.model_manager


def test_stale_lgbm_is_dated_from_data_end_not_tomorrow(get_forecast):
    through = date.today() - timedelta(days=25)
    body = get_forecast("lgbm", through)
    assert body["is_current"] is False
    assert body["data_through"] == through.isoformat()
    assert [p["date"] for p in body["forecasts"]] == [
        (through + timedelta(days=d)).isoformat() for d in (1, 2, 3)
    ]
    assert not any(p["is_current"] for p in body["forecasts"])


def test_sarima_origin_is_unverified_so_it_gets_no_date(get_forecast):
    body = get_forecast("sarima", date.today() - timedelta(days=25))
    assert body["is_current"] is False
    assert all(p["date"] is None and p["is_current"] is False for p in body["forecasts"])


def test_fresh_lgbm_is_current_and_future_dated(get_forecast):
    through = date.today()
    body = get_forecast("lgbm", through)
    assert body["is_current"] is True
    assert body["forecasts"][0]["date"] == (through + timedelta(days=1)).isoformat()


def test_data_end_yesterday_makes_day_one_today_not_tomorrow(get_forecast):
    through = date.today() - timedelta(days=1)
    body = get_forecast("lgbm", through)
    assert body["forecasts"][0]["date"] == date.today().isoformat()
    assert body["forecasts"][0]["is_current"] is False
