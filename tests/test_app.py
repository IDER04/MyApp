from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_renders_without_exceptions() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not app.exception


def test_app_updates_when_discount_rate_changes() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.number_input(key="discount_rate").set_value(18.0).run()
    assert not app.exception
    assert app.session_state["discount_rate"] == 18.0
