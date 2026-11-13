"""HTTP API for a prop firm's or copy-trading platform's risk desk.

uvicorn tiltguard.api:app       # set TILTGUARD_API_KEYS to require X-API-Key
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .data import load_fills
from .detect import Rules, run_all

app = FastAPI(
    title="TiltGuard",
    version="1.0.0",
    description="Behaviour and risk-rule alerts for trading accounts, each judged against its own normal.",
)


def auth(x_api_key: str | None = Header(default=None)):
    keys = {k.strip() for k in os.getenv("TILTGUARD_API_KEYS", "").split(",") if k.strip()}
    if keys and x_api_key not in keys:
        raise HTTPException(401, "missing or invalid X-API-Key")


class Fill(BaseModel):
    account: str
    ts: datetime
    coin: str = ""
    direction: str = Field(description="Open Long, Close Short, Buy, Sell, ...")
    notional: float = Field(ge=0)
    closed_pnl: float = 0.0
    fee: float = 0.0


class Limit(BaseModel):
    daily_loss: float | None = None
    max_drawdown: float | None = None


class ScanIn(BaseModel):
    fills: list[Fill] = Field(min_length=1, description="send at least the last 30 days per account")
    limits: dict[str, Limit] = {}
    since: datetime | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/scan", dependencies=[Depends(auth)])
def scan(body: ScanIn):
    f = load_fills(pd.DataFrame([x.model_dump() for x in body.fills]))
    limits = {a: {k: v for k, v in lim.model_dump().items() if v is not None} for a, lim in body.limits.items()}
    alerts = [x.as_dict() for x in run_all(f, Rules(limits=limits))]
    if body.since:
        alerts = [x for x in alerts if x["at"] >= body.since.isoformat(sep=" ")[: len(x["at"])]]
    return {"accounts": int(f.account.nunique()), "fills": len(f), "alerts": alerts}
