#!/usr/bin/env python3
"""
01b_market_anomaly_reconciliation.py

Stage 1B:
Reconcile anomalies found by 01_market_data_feasibility_audit.py.

This script DOES NOT modify or clean the raw Yahoo data.
It diagnoses:
1. Stock dates present when ^NSEI is absent.
2. Zero-volume dates and whether they are systemic across the panel.
3. The single/rare large adjusted-return flags.
4. Corporate actions near large-return dates.
5. Cross-sectional agreement on the actual trading calendar.

Run from the same project directory used for Stage 1:
    python 01b_market_anomaly_reconciliation.py

Expected input:
    market_data_audit/raw_yahoo/
    market_data_audit/01_nifty50_reference_trading_calendar.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

AUDIT_DIR = Path("market_data_audit")
RAW_DIR = AUDIT_DIR / "raw_yahoo"
OUTPUT_DIR = AUDIT_DIR / "stage1b_reconciliation"

BENCHMARK = "^NSEI"

STOCKS = [
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJFINANCE.NS",
    "BHARTIARTL.NS",
    "HDFCBANK.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "ITC.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "MARUTI.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBIN.NS",
    "SUNPHARMA.NS",
    "TCS.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
]

LARGE_ABS_LOG_RETURN_THRESHOLD = 0.30

# A date is considered cross-sectionally systemic if at least this fraction
# of the 20 stocks shows the same condition.
SYSTEMIC_FRACTION = 0.80
SYSTEMIC_COUNT = int(np.ceil(len(STOCKS) * SYSTEMIC_FRACTION))


# ============================================================
# 2. HELPERS
# ============================================================

def raw_path(ticker: str) -> Path:
    name = ticker.replace("^", "INDEX_")
    return RAW_DIR / f"{name}.csv"


def load_raw(ticker: str) -> pd.DataFrame:
    path = raw_path(ticker)
    if not path.exists():
        raise FileNotFoundError(f"Missing raw file: {path}")

    df = pd.read_csv(path, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date").drop_duplicates("Date", keep="last")
    df = df.set_index("Date")

    for col in [
        "Open", "High", "Low", "Close", "Adj Close", "Volume",
        "Dividends", "Stock Splits"
    ]:
        if col not in df.columns:
            if col in ("Dividends", "Stock Splits"):
                df[col] = 0.0
            else:
                df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def fmt_date(x) -> str:
    return pd.Timestamp(x).strftime("%Y-%m-%d")


# ============================================================
# 3. MAIN
# ============================================================

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 86)
    print("STAGE 1B — MARKET ANOMALY RECONCILIATION")
    print("=" * 86)

    # Load all data
    data = {ticker: load_raw(ticker) for ticker in STOCKS}
    nifty = load_raw(BENCHMARK)

    nifty_dates = pd.DatetimeIndex(nifty.index.unique()).sort_values()

    # ------------------------------------------------------------------
    # A. Cross-sectional trading-calendar support
    # ------------------------------------------------------------------
    all_stock_dates = sorted(set().union(*[set(df.index) for df in data.values()]))

    calendar_rows = []
    for dt in all_stock_dates:
        present = []
        positive_volume = []
        zero_volume = []

        for ticker, df in data.items():
            if dt in df.index:
                present.append(ticker)
                vol = df.at[dt, "Volume"]
                if pd.notna(vol) and vol > 0:
                    positive_volume.append(ticker)
                elif pd.notna(vol) and vol == 0:
                    zero_volume.append(ticker)

        calendar_rows.append({
            "Date": fmt_date(dt),
            "Stock_Count_With_Row": len(present),
            "Stock_Count_Positive_Volume": len(positive_volume),
            "Stock_Count_Zero_Volume": len(zero_volume),
            "NIFTY50_Row_Present": bool(dt in nifty.index),
            "All_20_Stocks_Present": len(present) == len(STOCKS),
            "Systemic_Zero_Volume": len(zero_volume) >= SYSTEMIC_COUNT,
        })

    calendar_df = pd.DataFrame(calendar_rows)
    calendar_df.to_csv(
        OUTPUT_DIR / "01_cross_sectional_trading_calendar.csv",
        index=False
    )

    # Dates present in stocks but absent from NIFTY
    extra_dates_df = calendar_df[
        (calendar_df["Stock_Count_With_Row"] > 0)
        & (~calendar_df["NIFTY50_Row_Present"])
    ].copy()

    extra_dates_df.to_csv(
        OUTPUT_DIR / "02_stock_dates_absent_from_nifty50.csv",
        index=False
    )

    # Strong-panel dates: almost all stocks say the date exists.
    likely_real_market_dates = extra_dates_df[
        extra_dates_df["Stock_Count_With_Row"] >= SYSTEMIC_COUNT
    ].copy()

    likely_real_market_dates.to_csv(
        OUTPUT_DIR / "03_likely_real_market_dates_missing_from_nifty.csv",
        index=False
    )

    # ------------------------------------------------------------------
    # B. Zero-volume analysis
    # ------------------------------------------------------------------
    zero_rows = []
    for ticker, df in data.items():
        z = df[df["Volume"] == 0]
        for dt, row in z.iterrows():
            zero_rows.append({
                "Ticker": ticker,
                "Date": fmt_date(dt),
                "Open": row["Open"],
                "High": row["High"],
                "Low": row["Low"],
                "Close": row["Close"],
                "Adj_Close": row["Adj Close"],
                "Volume": row["Volume"],
                "NIFTY50_Row_Present": bool(dt in nifty.index),
            })

    zero_df = pd.DataFrame(zero_rows)
    zero_df.to_csv(
        OUTPUT_DIR / "04_zero_volume_rows_detail.csv",
        index=False
    )

    if not zero_df.empty:
        zero_by_date = (
            zero_df.groupby("Date")
            .agg(
                Stocks_With_Zero_Volume=("Ticker", "nunique"),
                Tickers=("Ticker", lambda s: ";".join(sorted(set(s)))),
                NIFTY50_Row_Present=("NIFTY50_Row_Present", "max"),
            )
            .reset_index()
            .sort_values(
                ["Stocks_With_Zero_Volume", "Date"],
                ascending=[False, True]
            )
        )
        zero_by_date["Systemic_Zero_Volume"] = (
            zero_by_date["Stocks_With_Zero_Volume"] >= SYSTEMIC_COUNT
        )
    else:
        zero_by_date = pd.DataFrame(columns=[
            "Date", "Stocks_With_Zero_Volume", "Tickers",
            "NIFTY50_Row_Present", "Systemic_Zero_Volume"
        ])

    zero_by_date.to_csv(
        OUTPUT_DIR / "05_zero_volume_dates_summary.csv",
        index=False
    )

    # ------------------------------------------------------------------
    # C. Large adjusted return flags + nearby corporate actions
    # ------------------------------------------------------------------
    large_rows = []

    for ticker, df in data.items():
        adj = df["Adj Close"]
        log_ret = np.log(adj / adj.shift(1))
        flagged = log_ret[log_ret.abs() > LARGE_ABS_LOG_RETURN_THRESHOLD]

        for dt, value in flagged.items():
            pos = df.index.get_loc(dt)

            prev_dt = df.index[pos - 1] if pos > 0 else pd.NaT
            next_dt = df.index[pos + 1] if pos < len(df) - 1 else pd.NaT

            current = df.loc[dt]
            prev = df.loc[prev_dt] if pd.notna(prev_dt) else None
            next_ = df.loc[next_dt] if pd.notna(next_dt) else None

            # Corporate-action window: previous, same, next trading row.
            window_start = max(0, pos - 2)
            window_end = min(len(df), pos + 3)
            action_window = df.iloc[window_start:window_end]
            nearby_dividend = float(action_window["Dividends"].fillna(0).sum())
            nearby_split_rows = action_window[
                action_window["Stock Splits"].fillna(0) != 0
            ]

            large_rows.append({
                "Ticker": ticker,
                "Date": fmt_date(dt),
                "Adjusted_Log_Return": float(value),
                "Adjusted_Simple_Return_Pct": float((np.exp(value) - 1.0) * 100.0),
                "Prev_Date": fmt_date(prev_dt) if pd.notna(prev_dt) else None,
                "Prev_Adj_Close": float(prev["Adj Close"]) if prev is not None else None,
                "Current_Adj_Close": float(current["Adj Close"]),
                "Next_Date": fmt_date(next_dt) if pd.notna(next_dt) else None,
                "Next_Adj_Close": float(next_["Adj Close"]) if next_ is not None else None,
                "Same_Day_Dividend": float(current["Dividends"]),
                "Same_Day_Stock_Split": float(current["Stock Splits"]),
                "Nearby_5Row_Dividend_Total": nearby_dividend,
                "Nearby_5Row_Split_Events": int(len(nearby_split_rows)),
                "Nearby_5Row_Split_Dates": ";".join(
                    fmt_date(x) for x in nearby_split_rows.index
                ),
            })

    large_df = pd.DataFrame(large_rows)
    large_df.to_csv(
        OUTPUT_DIR / "06_large_return_diagnostics.csv",
        index=False
    )

    # ------------------------------------------------------------------
    # D. Corporate actions summary
    # ------------------------------------------------------------------
    action_rows = []
    for ticker, df in data.items():
        actions = df[
            (df["Dividends"].fillna(0) != 0)
            | (df["Stock Splits"].fillna(0) != 0)
        ]

        for dt, row in actions.iterrows():
            action_rows.append({
                "Ticker": ticker,
                "Date": fmt_date(dt),
                "Dividend": float(row["Dividends"]),
                "Stock_Split": float(row["Stock Splits"]),
            })

    actions_df = pd.DataFrame(action_rows)
    actions_df.to_csv(
        OUTPUT_DIR / "07_corporate_actions_detail.csv",
        index=False
    )

    # ------------------------------------------------------------------
    # E. Candidate calendar interpretation (NO CLEANING)
    # ------------------------------------------------------------------
    systemic_zero_dates = []
    if not zero_by_date.empty:
        systemic_zero_dates = zero_by_date.loc[
            zero_by_date["Systemic_Zero_Volume"], "Date"
        ].tolist()

    likely_missing_index_dates = likely_real_market_dates["Date"].tolist()

    summary = {
        "panel": {
            "stocks": len(STOCKS),
            "stock_rows_each_expected_from_stage1": 2593,
            "nifty50_rows": int(len(nifty)),
        },
        "calendar_reconciliation": {
            "unique_dates_across_20_stocks": int(len(all_stock_dates)),
            "stock_dates_absent_from_nifty50": int(len(extra_dates_df)),
            "dates_absent_from_nifty50_supported_by_at_least_80pct_of_stocks":
                int(len(likely_real_market_dates)),
            "candidate_nifty50_data_gap_dates":
                likely_missing_index_dates,
        },
        "zero_volume": {
            "total_stock_zero_volume_rows": int(len(zero_df)),
            "unique_zero_volume_dates": int(zero_df["Date"].nunique())
                if not zero_df.empty else 0,
            "systemic_threshold_stock_count": SYSTEMIC_COUNT,
            "systemic_zero_volume_dates": systemic_zero_dates,
        },
        "large_returns": {
            "threshold_abs_log_return": LARGE_ABS_LOG_RETURN_THRESHOLD,
            "flagged_rows": int(len(large_df)),
            "flagged_tickers": sorted(large_df["Ticker"].unique().tolist())
                if not large_df.empty else [],
        },
        "corporate_actions": {
            "total_events": int(len(actions_df)),
            "dividend_events": int((actions_df["Dividend"] != 0).sum())
                if not actions_df.empty else 0,
            "split_events": int((actions_df["Stock_Split"] != 0).sum())
                if not actions_df.empty else 0,
        },
        "important_note": (
            "Diagnostic stage only. No dates or stocks have been removed. "
            "Cleaning rules will be decided after manual review of this output."
        ),
    }

    with open(
        OUTPUT_DIR / "00_stage1b_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Terminal report
    # ------------------------------------------------------------------
    print("\nA. Dates present in stocks but absent from ^NSEI")
    print("-" * 86)
    if extra_dates_df.empty:
        print("None.")
    else:
        print(extra_dates_df.to_string(index=False))

    print("\nB. Zero-volume dates")
    print("-" * 86)
    if zero_by_date.empty:
        print("None.")
    else:
        print(zero_by_date.to_string(index=False))

    print("\nC. Large adjusted-return diagnostics")
    print("-" * 86)
    if large_df.empty:
        print("None.")
    else:
        print(large_df.to_string(index=False))

    print("\n" + "=" * 86)
    print("COMPACT SUMMARY — COPY/PASTE THIS BACK INTO CHATGPT")
    print("=" * 86)
    print(json.dumps(summary, indent=2))
    print("=" * 86)

    print("\nFiles written:")
    for p in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {p}")

    print(
        "\nNEXT: Do not remove any date yet. Paste the terminal output above "
        "back into ChatGPT so the final market-calendar cleaning rule can be "
        "decided scientifically."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
