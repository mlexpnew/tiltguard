"""Behaviour and risk-rule detectors. Each one compares a trader with their own recent normal
(the last 30 days), not with a single number for everyone: $50,000 is routine for one
trader and reckless for another.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Rules:
    baseline_days: int = 30
    tilt_loss_q: float = 0.05  # an hour in the trader's worst 5% ...
    tilt_min_loss: float = 1_000  # ... and at least $1,000 lost
    tilt_hours: int = 6
    tilt_mult: float = 3.0  # then opening 3x their usual 6-hour volume
    size_mult: float = 3.0  # an order 3x the trader's own large orders (95th percentile)
    size_min: float = 10_000
    overtrade_count_mult: float = 3.0  # a day with 3x the usual number of fills ...
    overtrade_fee_share: float = 0.5  # ... where fees eat half the gross result or more,
    overtrade_fee_mult: float = 2.0  # ... or are twice the usual day's fees
    overtrade_min_fee: float = 100
    min_history: int = 20  # fewer prior orders / active hours than this: cold start
    cold_size: float = 100_000  # cold start: firm-wide limits instead of the trader's own normal
    cold_tilt_loss: float = 10_000
    cold_tilt_opened: float = 250_000
    concentration_share: float = 0.7  # top 5% of trades make 70%+ of a month's profit
    limits: dict = field(default_factory=dict)  # account -> {"daily_loss": usd, "max_drawdown": usd}


@dataclass
class Alert:
    kind: str
    account: str
    at: str
    value: float
    reason: str

    def as_dict(self) -> dict:
        return self.__dict__


def orders(f: pd.DataFrame) -> pd.DataFrame:
    """Fills of the same order arrive as many rows; group them per account, minute, coin, direction."""
    g = f.groupby(["account", "ts", "coin", "direction", "opening"], as_index=False).agg(
        notional=("notional", "sum"),
        closed_pnl=("closed_pnl", "sum"),
        fee=("fee", "sum"),
        fills=("net", "size"),
    )
    return g.sort_values(["account", "ts"]).reset_index(drop=True)


def size_spikes(o: pd.DataFrame, r: Rules) -> list[Alert]:
    out = []
    for acct, g in o[o.opening].groupby("account"):
        g = g.set_index("ts").sort_index()
        med = g.notional.rolling(f"{r.baseline_days}D", closed="left").quantile(0.95)
        n = g.notional.rolling(f"{r.baseline_days}D", closed="left").count()
        # a quiet month falls back to the trader's whole earlier history before firm-wide limits
        ever_q = g.notional.expanding().quantile(0.95).shift(1)
        ever_n = pd.Series(np.arange(len(g)), index=g.index)
        g = g.assign(med=np.where(n >= r.min_history, med, ever_q), n=np.where(n >= r.min_history, n, ever_n))
        warm = g.n >= r.min_history
        hit = g[
            (warm & (g.notional >= r.size_mult * g.med) & (g.notional >= r.size_min))
            | (~warm & (g.notional >= r.cold_size))
        ]
        last = None
        for ts, row in zip(hit.index, hit.itertuples(), strict=True):
            if last is not None and ts - last < pd.Timedelta(hours=1):
                continue  # one alert per burst: the desk needs the episode, not every fill
            last = ts
            why = (
                f"{row.notional / row.med:.1f}x this trader's large orders (95th percentile) of recent orders"
                if row.n >= r.min_history
                else f"new or quiet account, above the firm-wide ${r.cold_size:,.0f} limit"
            )
            out.append(
                Alert(
                    "size_spike",
                    acct,
                    str(ts),
                    float(row.notional),
                    f"order of ${row.notional:,.0f} on {row.coin}, {why}",
                )
            )
    return out


def tilt(f: pd.DataFrame, r: Rules) -> list[Alert]:
    """A big loss, then a burst of new positions far above the trader's usual pace."""
    out = []
    for acct, g in f.groupby("account"):
        h = g.set_index("ts").resample("1h").agg({"net": "sum", "notional": "sum"})
        opened = g[g.opening].set_index("ts").notional.resample("1h").sum().reindex(h.index, fill_value=0)
        nxt = opened[::-1].rolling(r.tilt_hours, min_periods=1).sum()[::-1].shift(-1).fillna(0)
        roll6 = opened.rolling(r.tilt_hours).sum()
        base_q = h.net.where(h.net != 0).rolling(f"{r.baseline_days}D", closed="left").quantile(r.tilt_loss_q)
        active = roll6.where(roll6 > 0)
        base_vol = active.rolling(f"{r.baseline_days}D", closed="left").median()
        cnt = h.net.where(h.net != 0).rolling(f"{r.baseline_days}D", closed="left").count()
        warm = cnt >= r.min_history
        cand = h.index[
            (
                warm
                & (h.net <= -r.tilt_min_loss)
                & (h.net <= base_q)
                & (nxt >= r.tilt_mult * base_vol)
                & (base_vol > 0)
            )
            | (~warm & (h.net <= -r.cold_tilt_loss) & (nxt >= r.cold_tilt_opened))
        ]
        last = None
        for t in cand:
            if last is not None and t - last < pd.Timedelta(hours=r.tilt_hours):
                continue
            last = t
            pace = (
                f"{nxt[t] / base_vol[t]:.1f}x their usual pace"
                if warm[t] and base_vol[t] > 0
                else "new or quiet account, above firm-wide limits"
            )
            out.append(
                Alert(
                    "tilt",
                    acct,
                    str(t),
                    float(nxt[t]),
                    f"lost ${-h.net[t]:,.0f} in an hour, then opened ${nxt[t]:,.0f} in the next "
                    f"{r.tilt_hours} hours, {pace}",
                )
            )
    return out


def overtrading(f: pd.DataFrame, r: Rules) -> list[Alert]:
    out = []
    d = f.groupby(["account", "day"]).agg(
        fills=("net", "size"), fee=("fee", "sum"), gross=("closed_pnl", "sum")
    )
    for acct, g in d.groupby(level=0):
        g = g.droplevel(0)
        full = g.reindex(pd.date_range(g.index.min(), g.index.max()), fill_value=0)
        med = full.fills.where(full.fills > 0).rolling(f"{r.baseline_days}D", closed="left").median()
        n = full.fills.where(full.fills > 0).rolling(f"{r.baseline_days}D", closed="left").count()
        fee_med = full.fee.where(full.fills > 0).rolling(f"{r.baseline_days}D", closed="left").median()
        costly = (full.fee >= r.overtrade_fee_share * full.gross.abs().clip(lower=1)) | (
            (full.fee >= r.overtrade_fee_mult * fee_med) & (full.gross - full.fee <= 0)
        )
        hit = full[
            (n >= 5) & (full.fills >= r.overtrade_count_mult * med) & costly & (full.fee >= r.overtrade_min_fee)
        ]
        for day, row in hit.iterrows():
            out.append(
                Alert(
                    "overtrading",
                    acct,
                    str(day.date()),
                    float(row.fee),
                    f"{int(row.fills)} fills ({row.fills / med[day]:.1f}x usual); fees ${row.fee:,.0f} "
                    f"against a gross result of ${row.gross:,.0f}",
                )
            )
    return out


def rule_breaches(f: pd.DataFrame, r: Rules) -> list[Alert]:
    """Prop-firm style limits: daily loss limit and maximum drawdown from the equity peak."""
    out = []
    for acct, lim in r.limits.items():
        g = f[f.account == acct]
        if g.empty:
            continue
        daily = g.groupby("day").net.sum()
        if "daily_loss" in lim:
            for day, v in daily[daily <= -lim["daily_loss"]].items():
                out.append(
                    Alert(
                        "daily_loss_limit",
                        acct,
                        str(day.date()),
                        float(v),
                        f"lost ${-v:,.0f} in a day, limit ${lim['daily_loss']:,.0f}",
                    )
                )
        if "max_drawdown" in lim:
            cum = daily.cumsum()
            dd = cum.cummax() - cum
            breach = dd[dd >= lim["max_drawdown"]]
            if len(breach):
                day = breach.index[0]
                out.append(
                    Alert(
                        "max_drawdown",
                        acct,
                        str(day.date()),
                        float(dd[day]),
                        f"down ${dd[day]:,.0f} from the peak, limit ${lim['max_drawdown']:,.0f}",
                    )
                )
    return out


def concentration(f: pd.DataFrame, r: Rules) -> list[Alert]:
    """A month whose profit came from a handful of trades: luck to watch, not skill to copy."""
    out = []
    closes = f[f.closed_pnl != 0].copy()
    closes["month"] = closes.ts.dt.to_period("M")
    for (acct, m), g in closes.groupby(["account", "month"]):
        net = g.net.sum()
        if net <= 0 or len(g) < 40:
            continue
        top = g.net.nlargest(max(1, len(g) // 20)).sum()
        if top / net >= r.concentration_share:
            out.append(
                Alert(
                    "concentration",
                    acct,
                    str(m),
                    float(top / net),
                    f"the top 5% of {len(g)} closing trades made {top / net:.0%} of {m}'s net profit",
                )
            )
    return out


def run_all(f: pd.DataFrame, r: Rules | None = None) -> list[Alert]:
    r = r or Rules()
    return (
        tilt(f, r) + size_spikes(orders(f), r) + overtrading(f, r) + rule_breaches(f, r) + concentration(f, r)
    )


def summary(alerts: list[Alert]) -> pd.DataFrame:
    if not alerts:
        return pd.DataFrame(columns=["account", "kind", "alerts"])
    a = pd.DataFrame([x.as_dict() for x in alerts])
    return a.groupby(["account", "kind"]).size().rename("alerts").reset_index()


__all__ = [
    "Alert",
    "Rules",
    "concentration",
    "np",
    "orders",
    "overtrading",
    "rule_breaches",
    "run_all",
    "size_spikes",
    "summary",
    "tilt",
]
