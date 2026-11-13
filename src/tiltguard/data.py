"""Load exchange fills (Hyperliquid export format, or any CSV with the same meaning) and the
Bitcoin Fear & Greed index."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FILLS = ROOT / "data" / "historical_data.csv"
SENTIMENT = ROOT / "data" / "fear_greed_index.csv"

RENAME = {
    "Account": "account",
    "Coin": "coin",
    "Execution Price": "price",
    "Size USD": "notional",
    "Side": "side",
    "Direction": "direction",
    "Closed PnL": "closed_pnl",
    "Fee": "fee",
    "Start Position": "start_position",
    "Trade ID": "trade_id",
}


def load_fills(path=FILLS) -> pd.DataFrame:
    """Columns needed: account, ts (or Timestamp IST), direction, notional (Size USD); closed_pnl and fee
    if the export has them."""
    raw = pd.read_csv(path) if not isinstance(path, pd.DataFrame) else path.copy()
    f = raw.rename(columns=RENAME)
    if "ts" not in f:
        f["ts"] = pd.to_datetime(raw["Timestamp IST"], format="%d-%m-%Y %H:%M")
    f["ts"] = pd.to_datetime(f["ts"])
    for col, default in (("coin", ""), ("side", ""), ("closed_pnl", 0.0), ("fee", 0.0)):
        if col not in f:
            f[col] = default
    f["opening"] = f["direction"].astype(str).str.startswith(("Open", "Buy"))
    f["net"] = f["closed_pnl"] - f["fee"]
    f["day"] = f["ts"].dt.normalize()
    keep = [
        "ts",
        "day",
        "account",
        "coin",
        "side",
        "direction",
        "opening",
        "notional",
        "closed_pnl",
        "fee",
        "net",
    ]
    return f[keep].sort_values(["account", "ts"]).reset_index(drop=True)


def load_sentiment(path=SENTIMENT) -> pd.DataFrame:
    s = pd.read_csv(path)
    s["day"] = pd.to_datetime(s["date"])
    s["regime"] = s["classification"].map(
        lambda c: "fear" if "Fear" in c else ("greed" if "Greed" in c else "neutral")
    )
    return s[["day", "value", "classification", "regime"]]
