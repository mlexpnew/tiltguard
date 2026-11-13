import numpy as np
import pandas as pd

from tiltguard.data import load_fills
from tiltguard.detect import Rules, run_all
from tiltguard.research import persistence, regime_effect


def stream(seed=0, days=45, acct="A"):
    """A steady trader: ~40 fills a day, small orders, small wins and losses."""
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp("2025-01-01")
    for d in range(days):
        for _ in range(40):
            t = t0 + pd.Timedelta(days=d, minutes=int(rng.integers(0, 1440)))
            opening = rng.random() < 0.5
            rows.append(
                {
                    "account": acct,
                    "ts": t,
                    "coin": "ETH",
                    "direction": "Open Long" if opening else "Close Long",
                    "notional": float(rng.uniform(500, 2000)),
                    "closed_pnl": 0.0 if opening else float(rng.normal(5, 60)),
                    "fee": 0.4,
                }
            )
    return rows


def kinds(rows, rules=None):
    return {a.kind for a in run_all(load_fills(pd.DataFrame(rows)), rules)}


def test_steady_trader_is_quiet():
    assert not kinds(stream()) & {"tilt", "size_spike", "overtrading"}


def test_size_spike_against_own_normal():
    rows = stream() + [
        {
            "account": "A",
            "ts": pd.Timestamp("2025-02-10 12:00"),
            "coin": "BTC",
            "direction": "Open Long",
            "notional": 40_000.0,
            "closed_pnl": 0.0,
            "fee": 18.0,
        }
    ]
    assert "size_spike" in kinds(rows)


def test_tilt_after_big_loss():
    t = pd.Timestamp("2025-02-10 12:00")
    rows = stream() + [
        {
            "account": "A",
            "ts": t + pd.Timedelta(minutes=5),
            "coin": "BTC",
            "direction": "Close Long",
            "notional": 9000.0,
            "closed_pnl": -3000.0,
            "fee": 4.0,
        }
    ]
    rows += [
        {
            "account": "A",
            "ts": t + pd.Timedelta(minutes=70 + 15 * i),
            "coin": "BTC",
            "direction": "Open Long",
            "notional": 6000.0,
            "closed_pnl": 0.0,
            "fee": 2.7,
        }
        for i in range(12)
    ]
    assert "tilt" in kinds(rows)


def test_overtrading_day():
    day = pd.Timestamp("2025-02-10")
    rows = stream() + [
        {
            "account": "A",
            "ts": day + pd.Timedelta(minutes=2 * i),
            "coin": "ETH",
            "direction": "Open Long" if i % 2 else "Close Long",
            "notional": 1000.0,
            "closed_pnl": 0.0,
            "fee": 0.9,
        }
        for i in range(400)
    ]
    assert "overtrading" in kinds(rows)


def test_daily_loss_and_drawdown_limits():
    rows = stream() + [
        {
            "account": "A",
            "ts": pd.Timestamp("2025-02-10 12:00"),
            "coin": "BTC",
            "direction": "Close Long",
            "notional": 9000.0,
            "closed_pnl": -6000.0,
            "fee": 4.0,
        }
    ]
    k = kinds(rows, Rules(limits={"A": {"daily_loss": 5000, "max_drawdown": 5000}}))
    assert {"daily_loss_limit", "max_drawdown"} <= k


def test_new_account_uses_firm_wide_limit():
    rows = [
        {
            "account": "N",
            "ts": pd.Timestamp("2025-01-01 10:00"),
            "coin": "BTC",
            "direction": "Open Long",
            "notional": 250_000.0,
            "closed_pnl": 0.0,
            "fee": 100.0,
        }
    ]
    alerts = run_all(load_fills(pd.DataFrame(rows)))
    assert alerts and "firm-wide" in alerts[0].reason


def test_research_on_synthetic():
    f = load_fills(pd.DataFrame(stream(1, acct="A") + stream(2, acct="B")))
    s = pd.DataFrame(
        {
            "day": pd.date_range("2025-01-01", periods=45),
            "value": 50,
            "classification": ["Fear", "Greed"] * 22 + ["Fear"],
        }
    )
    s["regime"] = s.classification.str.lower()
    r = regime_effect(f, s)
    assert r["accounts"] == 2 and "ci" in r
    assert "mean" in persistence(f)
