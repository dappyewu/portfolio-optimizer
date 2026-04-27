"""Core analysis engine: fetch prices, 90/10 split, MSR fit, out-of-sample backtest.

Reuses risk_kit from ../src so portfolio math stays in one place.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Make ../src importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
import risk_kit as rk  # noqa: E402

PERIODS_PER_YEAR = 252
TRAIN_FRACTION = 0.75


# ---------- Risk-free rate ----------

def get_risk_free_rate(start: datetime, end: datetime) -> float:
    """Average 10y treasury rate over the lookback window, via FRED.

    Falls back to 0.04 if FRED is unavailable.
    """
    try:
        from fredapi import Fred
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
        load_dotenv()
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            return 0.04
        fred = Fred(api_key=api_key)
        series = fred.get_series_latest_release("GS10") / 100
        avg = series.loc[start:end].mean()
        if pd.isna(avg):
            return float(series.iloc[-1])
        return float(avg)
    except Exception:
        return 0.04


# ---------- Data fetch ----------

def fetch_prices(tickers: list[str], years: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Download adjusted-close prices. Returns (price_df, kept, dropped)."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)

    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    df = pd.DataFrame()
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                series = raw[t]["Adj Close"]
            else:
                series = raw["Adj Close"] if "Adj Close" in raw.columns else raw
            df[t] = series
        except Exception:
            continue

    df = df.dropna(how="all")
    min_obs = int(0.65 * years * PERIODS_PER_YEAR)
    kept = [c for c in df.columns if df[c].notnull().sum() >= min_obs]
    dropped = [c for c in tickers if c not in kept]
    df = df[kept].dropna()
    return df, kept, dropped


# ---------- Allocation styles ----------

@dataclass
class StrategyResult:
    name: str
    weights: np.ndarray  # aligned to tickers order
    test_returns: pd.Series
    wealth: pd.Series
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float


@dataclass
class AnalysisResult:
    tickers: list[str]
    train_prices: pd.DataFrame
    test_prices: pd.DataFrame
    train_returns: pd.DataFrame
    test_returns: pd.DataFrame
    er_train: pd.Series
    cov_train: pd.DataFrame
    risk_free_rate: float
    split_date: pd.Timestamp
    strategies: dict[str, StrategyResult] = field(default_factory=dict)
    dropped_tickers: list[str] = field(default_factory=list)
    initial_investment: float = 10_000.0

    def weights_table(self) -> pd.DataFrame:
        rows = {name: s.weights for name, s in self.strategies.items()}
        df = pd.DataFrame(rows, index=self.tickers).T
        return (df * 100).round(2)

    def stats_table(self) -> pd.DataFrame:
        rows = {
            name: {
                "Ann. Return %": round(s.ann_return * 100, 2),
                "Ann. Volatility %": round(s.ann_vol * 100, 2),
                "Sharpe": round(s.sharpe, 3),
                "Max Drawdown %": round(s.max_drawdown * 100, 2),
            }
            for name, s in self.strategies.items()
        }
        return pd.DataFrame(rows).T

    def wealth_table(self) -> pd.DataFrame:
        return pd.DataFrame({name: s.wealth for name, s in self.strategies.items()})


# ---------- Core run ----------

def _summarize(name: str, weights: np.ndarray, test_returns_df: pd.DataFrame,
               rf: float, initial: float, tickers: list[str]) -> StrategyResult:
    test_port = test_returns_df @ weights
    test_port.name = name

    wealth = initial * (1 + test_port).cumprod()
    # prepend the initial investment one bar before the first test return
    pre_index = test_port.index[:1] - pd.Timedelta(days=1)
    seed = pd.Series([initial], index=pre_index)
    wealth = pd.concat([seed, wealth])

    ann_ret = rk.annualize_rets(test_port, PERIODS_PER_YEAR)
    ann_vol = rk.annualize_vol(test_port, PERIODS_PER_YEAR)
    sharpe = rk.sharpe_ratio(test_port, rf, PERIODS_PER_YEAR)
    dd = rk.drawdown(test_port)["Drawdown"].min()

    return StrategyResult(
        name=name,
        weights=weights,
        test_returns=test_port,
        wealth=wealth,
        ann_return=float(ann_ret),
        ann_vol=float(ann_vol),
        sharpe=float(sharpe),
        max_drawdown=float(dd),
    )


def run_analysis(
    tickers: list[str],
    user_weights_pct: dict[str, float],
    years: int,
    initial_investment: float = 10_000.0,
) -> AnalysisResult:
    """Full pipeline: fetch -> split 90/10 -> fit MSR on train -> backtest on test."""
    if not tickers:
        raise ValueError("No tickers provided.")

    prices, kept, dropped = fetch_prices(tickers, years)
    if len(kept) < 2:
        raise ValueError(
            f"Need at least 2 tickers with sufficient history. Kept: {kept}, dropped: {dropped}."
        )

    # Normalise user weights to align with kept tickers and sum to 1
    raw_w = np.array([user_weights_pct.get(t, 0.0) for t in kept], dtype=float)
    if raw_w.sum() <= 0:
        # default to equal weight if user provided none for kept tickers
        raw_w = np.ones(len(kept))
    user_w = raw_w / raw_w.sum()

    daily_returns = prices.pct_change().dropna()
    n = len(daily_returns)
    split_idx = int(n * TRAIN_FRACTION)
    train_returns = daily_returns.iloc[:split_idx]
    test_returns = daily_returns.iloc[split_idx:]
    if len(test_returns) < 5:
        raise ValueError("Test window too small. Increase the lookback period.")

    split_date = test_returns.index[0]
    train_prices = prices.loc[:split_date].iloc[:-1]
    test_prices = prices.loc[split_date:]

    er_train = rk.annualize_rets(train_returns, PERIODS_PER_YEAR)
    cov_train = train_returns.cov() * PERIODS_PER_YEAR

    rf = get_risk_free_rate(train_returns.index[0].to_pydatetime(),
                            train_returns.index[-1].to_pydatetime())

    n_assets = len(kept)
    w_ew = np.repeat(1 / n_assets, n_assets)
    w_gmv = rk.gmv(cov_train.values)
    w_gmv = w_gmv / w_gmv.sum()
    w_msr = rk.msr(rf, er_train.values, cov_train.values)
    w_msr = w_msr / w_msr.sum()

    strategies = {
        "Original": _summarize("Original", user_w, test_returns, rf, initial_investment, kept),
        "Equal-Weight": _summarize("Equal-Weight", w_ew, test_returns, rf, initial_investment, kept),
        "GMV": _summarize("GMV", w_gmv, test_returns, rf, initial_investment, kept),
        "MSR": _summarize("MSR", w_msr, test_returns, rf, initial_investment, kept),
    }

    return AnalysisResult(
        tickers=kept,
        train_prices=train_prices,
        test_prices=test_prices,
        train_returns=train_returns,
        test_returns=test_returns,
        er_train=er_train,
        cov_train=cov_train,
        risk_free_rate=rf,
        split_date=split_date,
        strategies=strategies,
        dropped_tickers=dropped,
        initial_investment=initial_investment,
    )


# ---------- Efficient frontier (computed on training data) ----------

def efficient_frontier_points(er: pd.Series, cov: pd.DataFrame, n_points: int = 40) -> pd.DataFrame:
    target_rs = np.linspace(er.min(), er.max(), n_points)
    rows = []
    for tr in target_rs:
        try:
            w = rk.minimize_vol(tr, er.values, cov.values)
            rows.append({
                "Volatility": float(rk.portfolio_vol(w, cov.values)),
                "Return": float(rk.portfolio_return(w, er.values)),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)
