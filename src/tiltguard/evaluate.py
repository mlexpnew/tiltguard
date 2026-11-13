"""Benchmark TiltGuard on real Hyperliquid trades with injected, known episodes.

    python -m tiltguard.evaluate --gate

Real fills have no labels, so known episodes are injected into real accounts' streams:
tilt (a big loss, then a burst of new positions), size spikes and overtrading days. Both
TiltGuard and fixed global thresholds (the usual simple risk screen) are scored on recall
of those episodes, and on how many alerts they raise on the untouched real data.
Dev: December 2024 to February 2025. Held-out: March to April 2025 with subtler episodes,
run once.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_fills, load_sentiment
from .detect import orders, run_all
from .research import findings

# scored from, to, episode strength (held-out episodes are subtler). Detectors see all earlier history,
# as they would in production.
PERIODS = {"dev": ("2024-12-01", "2025-03-01", 5.0), "held_out": ("2025-03-01", "2025-05-02", 3.6)}


def inject(f: pd.DataFrame, start: str, end: str, strength: float, seed: int):
    """Returns fills with episodes added, and the truth list."""
    rng = np.random.default_rng(seed)
    rows, truth = [], []
    accts = [a for a, g in f[(f.ts >= start) & (f.ts < end)].groupby("account") if len(g) >= 300]
    s0, s1 = pd.Timestamp(start), pd.Timestamp(end)

    def when():
        return s0 + pd.Timedelta(hours=int(rng.integers(24, int((s1 - s0).total_seconds() // 3600) - 24)))

    def base(acct, **kw):
        return {
            "account": acct,
            "coin": "BTC",
            "side": "BUY",
            "direction": "Open Long",
            "opening": True,
            "notional": 0.0,
            "closed_pnl": 0.0,
            "fee": 0.0,
            **kw,
        }

    def window(g, t):
        return g[(g.ts < t) & (g.ts >= t - pd.Timedelta(days=30))]

    def active(w):
        """Only inject where the trader has a recent normal to compare with (see cold start)."""
        hours = w.set_index("ts").net.resample("1h").sum()
        return w.opening.sum() >= 40 and (hours != 0).sum() >= 30 and w.day.nunique() >= 8

    for k, acct in enumerate(rng.permutation(accts)):
        g = f[f.account == acct]
        kind = ("tilt", "size_spike", "overtrading")[k % 3]
        for _ in range(40):
            t = when().floor("h")
            w = window(g, t)
            if active(w):
                break
        else:
            continue
        # every episode is scaled from the 30 days before it: no look-ahead
        if kind == "tilt":
            hourly = w.set_index("ts").net.resample("1h").sum()
            loss = max(1_500.0, 1.5 * -hourly[hourly < 0].quantile(0.05))
            op = w[w.opening].set_index("ts").notional.resample("1h").sum().rolling(6).sum()
            typical = float(op[op > 0].median())
            rows.append(
                base(
                    acct,
                    ts=t + pd.Timedelta(minutes=10),
                    direction="Close Long",
                    side="SELL",
                    opening=False,
                    notional=loss * 3,
                    closed_pnl=-loss,
                    fee=loss * 3 * 0.00045,
                )
            )
            n = int(rng.integers(8, 20))
            for _i in range(n):
                rows.append(
                    base(
                        acct,
                        ts=t + pd.Timedelta(minutes=int(rng.integers(70, 330))),
                        notional=strength * typical / n,
                        fee=strength * typical / n * 0.00045,
                    )
                )
        elif kind == "size_spike":
            ow = orders(w)
            big = float(ow[ow.opening].notional.quantile(0.95))
            amt = max(15_000.0, strength * big)
            rows.append(base(acct, ts=t + pd.Timedelta(minutes=7), notional=amt, fee=amt * 0.00045))
        else:
            daily = w.groupby("day").agg(n=("net", "size"), fee=("fee", "sum"))
            n = int(strength * daily.n.median()) + 20
            med = float(w.notional.median())
            fee = max(med * 0.00045, 150.0 / n)
            for i in range(n):
                rows.append(
                    base(
                        acct,
                        ts=t.normalize() + pd.Timedelta(minutes=int(rng.integers(60, 1380))),
                        direction="Open Long" if i % 2 == 0 else "Close Long",
                        opening=i % 2 == 0,
                        notional=med,
                        closed_pnl=0.0,
                        fee=fee,
                    )
                )
        truth.append({"kind": kind, "account": acct, "at": t})
    add = pd.DataFrame(rows)
    add["net"] = add.closed_pnl - add.fee
    add["day"] = add.ts.dt.normalize()
    return pd.concat([f, add[f.columns]]).sort_values(["account", "ts"]).reset_index(drop=True), truth


def global_thresholds(f: pd.DataFrame) -> list[dict]:
    """The usual screen: one number for every trader."""
    out = []
    o = orders(f)
    for r in o[o.opening & (o.notional > 100_000)].itertuples():
        out.append({"kind": "size_spike", "account": r.account, "at": r.ts})
    for acct, g in f.groupby("account"):
        h = g.set_index("ts").net.resample("1h").sum()
        op = g[g.opening].set_index("ts").notional.resample("1h").sum().reindex(h.index, fill_value=0)
        nxt = op[::-1].rolling(6, min_periods=1).sum()[::-1].shift(-1).fillna(0)
        for t in h.index[(h <= -10_000) & (nxt >= 250_000)]:
            out.append({"kind": "tilt", "account": acct, "at": t})
        d = g.groupby("day").size()
        for day in d.index[d > 300]:
            out.append({"kind": "overtrading", "account": acct, "at": day})
    return out


def _hit(truth: dict, alerts: list[dict]) -> bool:
    for a in alerts:
        if a["kind"] != truth["kind"] or a["account"] != truth["account"]:
            continue
        at = pd.Timestamp(a["at"])
        if truth["kind"] == "overtrading" and at.normalize() == truth["at"].normalize():
            return True
        if truth["kind"] != "overtrading" and pd.Timedelta(0) <= at - truth["at"] <= pd.Timedelta(hours=2):
            return True
    return False


def run(which: str, f: pd.DataFrame) -> dict:
    start, end, strength = PERIODS[which]
    data = f[f.ts < end]
    in_window = lambda alerts: [a for a in alerts if pd.Timestamp(str(a["at"])[:19]) >= pd.Timestamp(start)]  # noqa: E731
    # noise on untouched real data
    clean_tg = in_window(
        [a.as_dict() for a in run_all(data) if a.kind in ("tilt", "size_spike", "overtrading")]
    )
    clean_gt = in_window(global_thresholds(data))
    months = (pd.Timestamp(end) - pd.Timestamp(start)).days / 30.4
    acct_months = data[data.ts >= start].account.nunique() * months
    res = {"accounts": int(data[data.ts >= start].account.nunique()), "account_months": round(acct_months, 1)}
    recall = {"tiltguard": {}, "global_thresholds": {}}
    for seed in range(3):
        inj, truth = inject(data, start, end, strength, seed=100 + seed + (0 if which == "dev" else 50))
        tg = [a.as_dict() for a in run_all(inj)]
        gt = global_thresholds(inj)
        for name, alerts in (("tiltguard", tg), ("global_thresholds", gt)):
            for t in truth:
                k = recall[name].setdefault(t["kind"], [0, 0])
                k[0] += _hit(t, alerts)
                k[1] += 1
    for name, clean in (("tiltguard", clean_tg), ("global_thresholds", clean_gt)):
        found = sum(v[0] for v in recall[name].values())
        total = sum(v[1] for v in recall[name].values())
        res[name] = {
            "recall": round(found / total, 4),
            "found": found,
            "injected": total,
            "by_kind": {k: f"{v[0]}/{v[1]}" for k, v in recall[name].items()},
            "alerts_on_real_data": len(clean),
            "alerts_per_account_month": round(len(clean) / acct_months, 2),
        }
    return res


def report(r: dict) -> str:
    d, h = r["dev"], r["held_out"]
    lines = [
        f"Real Hyperliquid fills, {d['accounts']} accounts. Dev: Dec 2024 to Feb 2025. "
        "Held-out: Mar to Apr 2025, subtler episodes, run once.",
        "",
        "| dev / held-out | Episodes caught | Alerts per account-month | Caught by type (held-out) |",
        "|---|---|---|---|",
    ]
    for k, label in (("global_thresholds", "Global thresholds"), ("tiltguard", "**TiltGuard**")):
        a, b = d[k], h[k]
        kinds = ", ".join(f"{x} {y}" for x, y in sorted(b["by_kind"].items()))
        lines.append(
            f"| {label} | {a['found']}/{a['injected']} / {b['found']}/{b['injected']} | "
            f"{a['alerts_per_account_month']} / {b['alerts_per_account_month']} | {kinds} |"
        )
    reg, per = r["findings"]["regime"], r["findings"]["persistence"]
    lines += [
        "",
        f"Fear vs Greed (Greed minus Fear, mean daily PnL per account): ${reg['per_account_diff']:,.0f}, "
        f"95% CI ${reg['ci'][0]:,.0f} to ${reg['ci'][1]:,.0f}, {reg['accounts']} accounts.",
        f"Leaderboard persistence (rank correlation, one month to the next): {per['mean']:.2f}, "
        f"95% CI {per['ci'][0]:.2f} to {per['ci'][1]:.2f}.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark TiltGuard")
    ap.add_argument("--out", default="eval_results")
    ap.add_argument("--gate", action="store_true")
    a = ap.parse_args(argv)
    f = load_fills()
    r = {"dev": run("dev", f), "held_out": run("held_out", f), "findings": findings(f, load_sentiment())}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(r, indent=2, default=str))
    md = report(r)
    (out / "report.md").write_text(md)
    print(md)
    if a.gate:
        d = r["dev"]
        if d["tiltguard"]["recall"] < 0.85 or d["tiltguard"]["alerts_per_account_month"] > 1.5:
            print("gate failed", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
