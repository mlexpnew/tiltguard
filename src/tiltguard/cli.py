"""Command line.

tiltguard scan fills.csv --out alerts.xlsx      # alerts for every account in an export
tiltguard research                              # Fear vs Greed and leaderboard persistence on the sample
tiltguard benchmark                             # the injected-episode benchmark
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from .data import FILLS, load_fills, load_sentiment
from .detect import run_all, summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tiltguard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("fills", nargs="?", default=str(FILLS))
    s.add_argument("--since", help="only report alerts from this date (detectors still use earlier history)")
    s.add_argument("--out", default="tiltguard_alerts.xlsx")
    sub.add_parser("research")
    sub.add_parser("benchmark")
    a = ap.parse_args(argv)
    if a.cmd == "benchmark":
        from .evaluate import main as ev

        return ev([])
    if a.cmd == "research":
        from .research import findings

        print(json.dumps(findings(load_fills(), load_sentiment()), indent=2, default=str))
        return 0
    alerts = [x.as_dict() for x in run_all(load_fills(a.fills)) if not a.since or x.at >= a.since]
    df = pd.DataFrame(alerts, columns=["kind", "account", "at", "value", "reason"])
    with pd.ExcelWriter(a.out) as w:
        df.to_excel(w, sheet_name="alerts", index=False)
        summary(run_all(load_fills(a.fills))).to_excel(w, sheet_name="by_account", index=False)
    print(f"{len(df)} alerts -> {a.out}")
    print(df.kind.value_counts().to_string() if len(df) else "no alerts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
