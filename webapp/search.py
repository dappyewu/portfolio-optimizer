"""Ticker lookup helpers built on yfinance's search endpoint."""
from __future__ import annotations

from dataclasses import dataclass

import yfinance as yf


@dataclass
class TickerHit:
    symbol: str
    name: str
    exchange: str

    @property
    def label(self) -> str:
        return f"{self.symbol} — {self.name} ({self.exchange})"


def search_tickers(query: str, max_results: int = 8) -> list[TickerHit]:
    query = query.strip()
    if not query:
        return []
    try:
        results = yf.Search(query, max_results=max_results, news_count=0).quotes or []
    except Exception:
        return []

    hits: list[TickerHit] = []
    for r in results:
        symbol = r.get("symbol")
        if not symbol:
            continue
        name = r.get("shortname") or r.get("longname") or symbol
        exchange = r.get("exchDisp") or r.get("exchange") or ""
        hits.append(TickerHit(symbol=symbol, name=name, exchange=exchange))
    return hits
