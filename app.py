"""TiltGuard risk desk. streamlit run app.py"""

import pandas as pd
import streamlit as st

from tiltguard.data import load_fills, load_sentiment
from tiltguard.detect import Rules, run_all
from tiltguard.research import findings

st.set_page_config(page_title="TiltGuard", layout="wide")
st.title("TiltGuard")
st.caption(
    "Tilt, size spikes, overtrading, rule breaches and lucky streaks, each trader judged against their "
    "own last 30 days. Sample: real Hyperliquid fills, 32 accounts."
)

up = st.sidebar.file_uploader("Fills export (CSV)", type="csv")
fills = load_fills(pd.read_csv(up)) if up else st.cache_data(load_fills)()
since = st.sidebar.date_input("Alerts from", value=pd.Timestamp("2025-03-01"))
st.sidebar.subheader("Prop-firm limits (all accounts)")
dl = st.sidebar.number_input("Daily loss limit ($, 0 = off)", value=0, step=1000)
dd = st.sidebar.number_input("Max drawdown ($, 0 = off)", value=0, step=5000)
lim = {k: v for k, v in (("daily_loss", dl), ("max_drawdown", dd)) if v}
rules = Rules(limits={a: lim for a in fills.account.unique()} if lim else {})


@st.cache_data
def scan(f, limits_key):
    return pd.DataFrame(
        [x.as_dict() for x in run_all(f, rules)], columns=["kind", "account", "at", "value", "reason"]
    )


a = scan(fills, str(lim))
a = a[a["at"] >= str(since)]
c1, c2, c3 = st.columns(3)
c1.metric("Alerts", len(a))
c2.metric("Accounts with alerts", a.account.nunique())
c3.metric("Accounts", fills.account.nunique())
kinds = st.multiselect("Show", sorted(a.kind.unique()), default=sorted(a.kind.unique()))
view = a[a.kind.isin(kinds)].sort_values("at", ascending=False)
view = view.assign(account=view.account.str[:10])
st.dataframe(view[["at", "kind", "account", "reason"]], use_container_width=True, hide_index=True)

st.subheader("One trader")
acct = st.selectbox("Account", sorted(fills.account.unique()), format_func=lambda x: x[:10])
g = fills[fills.account == acct].set_index("ts").net.resample("1D").sum().cumsum()
st.line_chart(g.rename("cumulative net PnL ($)"))
st.dataframe(a[a.account == acct][["at", "kind", "reason"]], hide_index=True, use_container_width=True)

with st.expander("Does Fear vs Greed matter? Does last month's leaderboard predict the next?"):
    r = findings(fills, load_sentiment())
    reg, per = r["regime"], r["persistence"]
    st.write(
        f"Greed minus Fear, mean daily PnL per account: ${reg['per_account_diff']:,.0f} "
        f"(95% CI ${reg['ci'][0]:,.0f} to ${reg['ci'][1]:,.0f}, {reg['accounts']} accounts). "
        + ("Not distinguishable from zero." if not reg["significant"] else "")
    )
    st.write(
        f"Month-to-month rank correlation of PnL: {per['mean']:.2f} "
        f"(95% CI {per['ci'][0]:.2f} to {per['ci'][1]:.2f})."
    )
    st.dataframe(pd.DataFrame(per["pairs"]), hide_index=True)
