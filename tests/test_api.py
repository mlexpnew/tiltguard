import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from tiltguard.api import app  # noqa: E402


def test_scan_flags_breach():
    c = TestClient(app)
    fills = [
        {
            "account": "A",
            "ts": "2025-01-02T10:00:00",
            "direction": "Close Long",
            "notional": 9000,
            "closed_pnl": -6000,
            "fee": 4,
        }
    ]
    r = c.post("/v1/scan", json={"fills": fills, "limits": {"A": {"daily_loss": 5000}}})
    assert r.status_code == 200
    assert any(a["kind"] == "daily_loss_limit" for a in r.json()["alerts"])
