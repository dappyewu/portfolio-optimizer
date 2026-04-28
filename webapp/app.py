"""Streamlit UI for the portfolio optimiser.

Run from repo root:
    streamlit run webapp/app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from engine import efficient_frontier_points, run_analysis  # noqa: E402
from search import search_tickers  # noqa: E402

st.set_page_config(page_title="Portfolio Optimizer and Back Tester", layout="wide")

# ---------- Session state ----------
# Drain a "clear search" request before any widgets render — Streamlit
# disallows modifying a widget's state after it's instantiated in the
# current run, so we set a flag in the Add handler and clear here.
if st.session_state.pop("_clear_search", False):
    st.session_state["search_query"] = ""
    st.session_state["search_results"] = []

if "holdings" not in st.session_state:
    st.session_state.holdings = {}  # symbol -> {"name": str, "weight": float}
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "result" not in st.session_state:
    st.session_state.result = None
if "editor_version" not in st.session_state:
    st.session_state.editor_version = 0


def bump_editor():
    """Force the data_editor to re-initialise from session_state.holdings."""
    st.session_state.editor_version += 1


def add_ticker(symbol: str, name: str, weight: float = 0.0):
    st.session_state.holdings[symbol] = {"name": name, "weight": float(weight)}
    bump_editor()


def remove_ticker(symbol: str):
    st.session_state.holdings.pop(symbol, None)
    bump_editor()


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    st.caption("Adjust these **before** clicking Run analysis. They control the lookback window and starting capital used in the backtest.")
    years = st.slider("Lookback (years)", min_value=2, max_value=15, value=5, step=1,
                      help="How much history to use. First 75% trains the model, last 25% is the out-of-sample test.")
    initial_investment = st.number_input("Initial investment ($)", min_value=100.0,
                                         value=10_000.0, step=1000.0, format="%.0f")

    fred_ok = bool(os.getenv("FRED_API_KEY"))
    if not fred_ok:
        try:
            from dotenv import load_dotenv
            load_dotenv(_HERE.parent / ".env")
            fred_ok = bool(os.getenv("FRED_API_KEY"))
        except Exception:
            pass
    if fred_ok:
        st.caption("FRED API key detected — using live 10y Treasury for risk-free rate.")
    else:
        st.warning("No FRED_API_KEY found in .env — falling back to 4% risk-free rate.")

# ---------- Main ----------
st.title("Portfolio Optimizer and Back Tester")
st.caption(
    "Add holdings, set weights, then run the analysis. "
    "The first 75% of the price history fits the Maximum Sharpe Ratio (MSR) allocation; "
    "the last 25% is held out to compare your portfolio against MSR, GMV (Global Minimum Volatility), and Equal-Weight, "
    "rebalanced once at the split point. "
    "**Tip:** open the sidebar (top-left ☰ on mobile) to adjust the lookback window and starting capital before running the analysis."
)
st.warning("Refreshing or closing this page will clear your portfolio and results. Keep this tab open while you work.")

# --- Build portfolio ---
st.subheader("1. Build your portfolio")

tab_search, tab_upload = st.tabs(["Search & add", "Upload CSV"])

with tab_search:
    with st.form("search_form", clear_on_submit=False):
        col_search, col_action = st.columns([3, 1])
        with col_search:
            query = st.text_input(
                "Search by company or ticker (e.g. 'Apple', 'AAPL', 'Microsoft')",
                key="search_query",
            )
        with col_action:
            st.write("")
            st.write("")
            submitted = st.form_submit_button("Search", use_container_width=True)
        if submitted:
            st.session_state.search_results = search_tickers(query, max_results=8)

    if st.session_state.search_results:
        st.write("**Search results** — click *Add* to include in your portfolio:")
        for hit in st.session_state.search_results:
            c1, c2, c3 = st.columns([5, 2, 1])
            c1.write(f"**{hit.symbol}** — {hit.name}")
            c2.write(hit.exchange)
            if c3.button("Add", key=f"add_{hit.symbol}"):
                add_ticker(hit.symbol, hit.name)
                st.session_state._clear_search = True
                st.rerun()

with tab_upload:
    st.warning(
        "Tickers must match **Yahoo Finance** symbols exactly "
        "(e.g. `AAPL`, `MSFT`, `BRK-B`, `VOD.L` for London, `VOW3.DE` for Xetra). "
        "Anything Yahoo doesn't recognise will be silently dropped during analysis."
    )
    st.caption("CSV must have two columns: **Ticker** and **Weight** (weights as percentages or raw — they'll be normalised).")

    sample_csv = "Ticker,Weight\nAAPL,40\nMSFT,35\nGOOGL,25\n"
    st.download_button(
        "Download sample CSV",
        data=sample_csv,
        file_name="portfolio_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"], key="portfolio_upload")
    replace_existing = st.checkbox("Replace existing holdings (uncheck to merge)", value=True)

    if uploaded is not None:
        try:
            up_df = pd.read_csv(uploaded)
            cols_lower = {c.lower(): c for c in up_df.columns}
            if "ticker" not in cols_lower or "weight" not in cols_lower:
                st.error("CSV must contain 'Ticker' and 'Weight' columns.")
            else:
                t_col = cols_lower["ticker"]
                w_col = cols_lower["weight"]
                up_df = up_df[[t_col, w_col]].dropna()
                up_df[t_col] = up_df[t_col].astype(str).str.strip().str.upper()
                up_df[w_col] = pd.to_numeric(up_df[w_col], errors="coerce").fillna(0)
                up_df = up_df[up_df[t_col] != ""]

                if replace_existing:
                    st.session_state.holdings.clear()
                for _, row in up_df.iterrows():
                    sym = row[t_col]
                    weight = float(row[w_col])
                    existing_name = st.session_state.holdings.get(sym, {}).get("name", sym)
                    st.session_state.holdings[sym] = {"name": existing_name, "weight": weight}
                bump_editor()
                st.success(f"Loaded {len(up_df)} ticker(s) from CSV.")
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")

st.divider()

# --- Holdings table ---
st.subheader("2. Set weights")
if not st.session_state.holdings:
    st.info("No holdings yet. Search and add some companies above.")
else:
    holdings_df = pd.DataFrame(
        [
            {"Ticker": sym, "Company": data["name"], "Weight (%)": data["weight"]}
            for sym, data in st.session_state.holdings.items()
        ]
    )
    edited = st.data_editor(
        holdings_df,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn(disabled=True),
            "Company": st.column_config.TextColumn(disabled=True),
            "Weight (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0),
        },
        key=f"holdings_editor_{st.session_state.editor_version}",
    )
    # Sync edits back (user-typed values flow into session_state)
    for _, row in edited.iterrows():
        sym = row["Ticker"]
        if sym in st.session_state.holdings:
            st.session_state.holdings[sym]["weight"] = float(row["Weight (%)"])

    total = sum(h["weight"] for h in st.session_state.holdings.values())
    st.metric("Total weight", f"{total:.2f}%")
    btn_cols = st.columns(3)
    if btn_cols[0].button("Normalise to 100%", use_container_width=True):
        if total > 0:
            for h in st.session_state.holdings.values():
                h["weight"] = h["weight"] * 100.0 / total
            bump_editor()
            st.rerun()
    if btn_cols[1].button("Equal-weight", use_container_width=True):
        n = len(st.session_state.holdings)
        if n:
            for h in st.session_state.holdings.values():
                h["weight"] = 100.0 / n
            bump_editor()
            st.rerun()
    if btn_cols[2].button("Clear all", use_container_width=True):
        st.session_state.holdings.clear()
        st.session_state.result = None
        bump_editor()
        st.rerun()

    # Per-row remove buttons
    with st.expander("Remove individual tickers"):
        for sym in list(st.session_state.holdings.keys()):
            if st.button(f"Remove {sym}", key=f"rm_{sym}"):
                remove_ticker(sym)
                st.rerun()

st.divider()

# --- Run ---
st.subheader("3. Run the analysis")
run_disabled = len(st.session_state.holdings) < 2
if run_disabled:
    st.caption("Add at least 2 tickers to run.")

if st.button("Run analysis", type="primary", disabled=run_disabled):
    tickers = list(st.session_state.holdings.keys())
    user_weights = {t: st.session_state.holdings[t]["weight"] for t in tickers}
    with st.spinner("Downloading prices, fitting MSR, backtesting..."):
        try:
            st.session_state.result = run_analysis(
                tickers=tickers,
                user_weights_pct=user_weights,
                years=years,
                initial_investment=initial_investment,
            )
        except Exception as e:
            st.session_state.result = None
            st.error(f"Analysis failed: {e}")

# ---------- Results ----------
result = st.session_state.result
if result is not None:
    st.divider()
    st.header("Results")

    row1 = st.columns(2)
    row1[0].metric("Tickers used", len(result.tickers))
    row1[1].metric("Risk-free rate", f"{result.risk_free_rate * 100:.2f}%")
    row2 = st.columns(2)
    row2[0].metric("Train rows", len(result.train_returns))
    row2[1].metric("Test rows", len(result.test_returns))

    if result.dropped_tickers:
        st.warning(f"Dropped (insufficient history): {', '.join(result.dropped_tickers)}")
    st.caption(f"Train/test split at **{result.split_date.date()}**. "
               "MSR weights were computed on the training window only.")

    # Weights
    st.subheader("Allocation weights (%)")
    st.dataframe(result.weights_table(), use_container_width=True)

    # Expected (training-period) performance for each strategy
    import numpy as np
    er_vals = result.er_train.values
    cov_vals = result.cov_train.values
    rf = result.risk_free_rate
    train_rows = {}
    for name, s in result.strategies.items():
        w = s.weights
        ret = float(w @ er_vals)
        vol = float(np.sqrt(w @ cov_vals @ w))
        sharpe = (ret - rf) / vol if vol > 0 else float("nan")
        train_rows[name] = {
            "Ann. Return %": round(ret * 100, 2),
            "Ann. Volatility %": round(vol * 100, 2),
            "Sharpe": round(sharpe, 3),
        }
    train_stats = pd.DataFrame(train_rows).T

    train_span_days = (result.train_returns.index[-1] - result.train_returns.index[0]).days
    train_months = max(1, round(train_span_days / 30.44))
    st.subheader(f"Expected performance (training period — last {train_months} months excluding the held-out window)")
    st.caption("Annualised stats computed on the training window using each strategy's weights — "
               "this is what each strategy 'expected' to deliver based on the fitting data.")
    st.dataframe(train_stats, use_container_width=True)

    # Out-of-sample (held-out test window)
    test_span_days = (result.test_returns.index[-1] - result.test_returns.index[0]).days
    test_months = max(1, round(test_span_days / 30.44))
    st.subheader(f"Recent performance results if strategy implemented — last {test_months} months")
    st.dataframe(result.stats_table(), use_container_width=True)
    st.caption(
        f"⚠️ Test window is only ~{len(result.test_returns)} trading days. "
        "The Sharpe 95% CI column shows how uncertain each estimate is — wide intervals "
        "mean the realised difference between strategies is statistically weak. "
        "For more reliable conclusions, increase the lookback period in the sidebar."
    )

    # Wealth curve
    st.subheader("Wealth curve over the test window")
    wealth = result.wealth_table()
    fig_wealth = px.line(wealth, labels={"value": "Portfolio value ($)", "index": "Date",
                                         "variable": "Strategy"})
    fig_wealth.update_layout(legend_title_text="Strategy", height=450)
    st.plotly_chart(fig_wealth, use_container_width=True)

    # Drawdown
    st.subheader("Drawdown — Original vs MSR")
    dd_data = {}
    for name in ["Original", "MSR"]:
        s = result.strategies[name]
        wealth_s = s.wealth
        peaks = wealth_s.cummax()
        dd_data[name] = (wealth_s - peaks) / peaks * 100
    dd_df = pd.DataFrame(dd_data)
    fig_dd = px.area(dd_df, labels={"value": "Drawdown (%)", "index": "Date",
                                    "variable": "Strategy"})
    fig_dd.update_layout(height=350)
    st.plotly_chart(fig_dd, use_container_width=True)

    # Efficient frontier
    st.subheader("Efficient frontier (training data)")
    ef = efficient_frontier_points(result.er_train, result.cov_train, n_points=40)
    fig_ef = go.Figure()
    if not ef.empty:
        fig_ef.add_trace(go.Scatter(x=ef["Volatility"], y=ef["Return"], mode="lines",
                                    name="Frontier"))
    # Marker points for each strategy (based on training stats)
    import numpy as np
    er_vals = result.er_train.values
    cov_vals = result.cov_train.values
    for name, s in result.strategies.items():
        w = s.weights
        ret = float(w @ er_vals)
        vol = float(np.sqrt(w @ cov_vals @ w))
        fig_ef.add_trace(go.Scatter(x=[vol], y=[ret], mode="markers+text",
                                    name=name, text=[name], textposition="top center",
                                    marker=dict(size=11)))
    # Capital market line
    if not ef.empty:
        msr_w = result.strategies["MSR"].weights
        msr_ret = float(msr_w @ er_vals)
        msr_vol = float(np.sqrt(msr_w @ cov_vals @ msr_w))
        fig_ef.add_trace(go.Scatter(x=[0, msr_vol], y=[result.risk_free_rate, msr_ret],
                                    mode="lines", line=dict(dash="dash"), name="CML"))
    fig_ef.update_layout(xaxis_title="Volatility (annualised)",
                         yaxis_title="Return (annualised)", height=450)
    st.plotly_chart(fig_ef, use_container_width=True)

    st.caption(
        "Note: training-window MSR weights are not guaranteed to outperform out-of-sample. "
        "Past performance is not indicative of future returns."
    )
