"""Answers to the original questions, with uncertainty, on the real data.

1. Does market sentiment (Fear vs Greed) change how much traders make?
   Compared within each account (Greed days minus Fear days), with a bootstrap over accounts,
   so one large trader or one wild day cannot decide the answer.
2. Is last month's leaderboard a guide to next month?
   Rank correlation of each month's PnL with the next month's, across accounts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _boot(values: np.ndarray, rng, n: int = 4000) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def regime_effect(f: pd.DataFrame, s: pd.DataFrame, seed: int = 0) -> dict:
    daily = f.groupby(["account", "day"]).net.sum().reset_index().merge(s, on="day", how="left")
    pooled = daily.groupby("regime").net.mean().to_dict()
    per = daily[daily.regime.isin(["fear", "greed"])].groupby(["account", "regime"]).net.mean().unstack()
    per = per.dropna()
    diff = (per["greed"] - per["fear"]).to_numpy()
    rng = np.random.default_rng(seed)
    lo, hi = _boot(diff, rng)
    return {
        "pooled_mean_daily_pnl": {k: round(v, 2) for k, v in pooled.items()},
        "accounts": int(len(diff)),
        "per_account_diff": float(diff.mean()) if len(diff) else float("nan"),
        "median_diff": float(np.median(diff)) if len(diff) else float("nan"),
        "share_better_in_greed": float((diff > 0).mean()) if len(diff) else float("nan"),
        "ci": (lo, hi),
        "significant": bool(lo > 0 or hi < 0),
    }


def persistence(f: pd.DataFrame, seed: int = 0) -> dict:
    m = f.assign(month=f.ts.dt.to_period("M")).groupby(["account", "month"]).net.sum().unstack()
    months = [c for c in m.columns if m[c].notna().sum() >= 10]
    corrs = []
    for a, b in zip(months, months[1:], strict=False):
        both = m[[a, b]].dropna()
        if len(both) >= 10:
            corrs.append(
                {
                    "from": str(a),
                    "to": str(b),
                    "accounts": len(both),
                    "spearman": float(both[a].rank().corr(both[b].rank())),
                }
            )
    vals = np.array([c["spearman"] for c in corrs])
    lo, hi = _boot(vals, np.random.default_rng(seed)) if len(vals) else (float("nan"), float("nan"))
    return {"pairs": corrs, "mean": float(vals.mean()) if len(vals) else float("nan"), "ci": (lo, hi)}


def findings(f: pd.DataFrame, s: pd.DataFrame) -> dict:
    return {"regime": regime_effect(f, s), "persistence": persistence(f)}
