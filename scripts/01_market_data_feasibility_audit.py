#!/usr/bin/env python3
"""
01_market_data_feasibility_audit.py

Stage 1 of the frozen methodology:
Market-data feasibility and integrity audit for 20 NSE equities + NIFTY 50.

Frozen study design
-------------------
Market: India / NSE
Period: 2016-01-01 to 2026-06-30 inclusive
Source: Yahoo Finance via yfinance
Stocks: 20 large-cap NSE equities
Market context: NIFTY 50 (^NSEI)

What this script does
---------------------
1. Downloads daily OHLCV, adjusted close and corporate actions.
2. Saves one raw CSV per ticker.
3. Uses NIFTY 50 trading dates as the reference calendar.
4. Audits date coverage, missing dates, duplicates, missing values,
   invalid prices/volumes, OHLC consistency, zero-volume days,
   large adjusted-price returns, dividends and stock splits.
5. Produces a preliminary eligibility flag.
6. Writes CSV/JSON outputs and prints a compact terminal summary.

IMPORTANT
---------
This script does NOT calculate technical indicators and does NOT train a model.
We first verify the empirical market-data foundation.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# 1. FROZEN CONFIGURATION
# ============================================================

START_DATE = "2016-01-01"

# yfinance treats `end` as exclusive, so 2026-07-01 includes 2026-06-30.
END_DATE_EXCLUSIVE = "2026-07-01"
PLANNED_END_DATE = "2026-06-30"

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

TICKERS = STOCKS + [BENCHMARK]

OUTPUT_DIR = Path("market_data_audit")
RAW_DIR = OUTPUT_DIR / "raw_yahoo"

# Download retries
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 3

# Preliminary audit thresholds.
# These are NOT manuscript claims; we will inspect the actual audit output
# before freezing any exclusion rule.
MIN_BENCHMARK_COVERAGE_PCT = 98.0
MAX_END_LAG_TRADING_DAYS = 5

# Flag unusually large one-day adjusted-price movements for manual inspection.
# These are NOT automatically treated as errors.
LARGE_ABS_LOG_RETURN_THRESHOLD = 0.30  # ~35% simple move


# ============================================================
# 2. HELPERS
# ============================================================

def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance columns for both old and recent versions."""
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        # For a single ticker, yfinance may still return a MultiIndex.
        # Prefer the price-field level if present.
        level0 = list(out.columns.get_level_values(0))
        expected = {"Open", "High", "Low", "Close", "Adj Close", "Volume",
                    "Dividends", "Stock Splits"}
        if len(expected.intersection(set(level0))) >= 4:
            out.columns = out.columns.get_level_values(0)
        else:
            out.columns = [
                "_".join(str(x) for x in tup if str(x) not in ("", "None"))
                for tup in out.columns.to_flat_index()
            ]

    # Defensive renaming for occasional variants.
    rename_map = {
        "Adj_Close": "Adj Close",
        "AdjClose": "Adj Close",
        "Stock_Splits": "Stock Splits",
    }
    out = out.rename(columns=rename_map)

    # Remove duplicated column names if a provider/version created any.
    out = out.loc[:, ~out.columns.duplicated()].copy()

    return out


def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the index to timezone-naive normalized dates."""
    out = df.copy()

    idx = pd.to_datetime(out.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)

    out.index = idx.normalize()
    out.index.name = "Date"
    out = out[~out.index.isna()].sort_index()

    return out


def download_one(ticker: str) -> tuple[pd.DataFrame | None, str | None]:
    """Download one ticker with retry logic."""
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  Downloading {ticker} (attempt {attempt}/{MAX_RETRIES}) ...")

            df = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE_EXCLUSIVE,
                interval="1d",
                auto_adjust=False,
                actions=True,
                progress=False,
                threads=False,
            )

            if df is None or df.empty:
                last_error = "Empty dataframe returned by Yahoo Finance."
            else:
                df = flatten_yfinance_columns(df)
                df = normalize_index(df)

                if not df.empty:
                    return df, None

                last_error = "Dataframe became empty after date normalization."

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_WAIT_SECONDS)

    return None, last_error


def safe_pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return 100.0 * numerator / denominator


def as_date_string(x) -> str | None:
    if x is None or pd.isna(x):
        return None
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def audit_ticker(
    ticker: str,
    df: pd.DataFrame,
    benchmark_dates: pd.DatetimeIndex,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """
    Returns:
        summary_row
        missing_date_rows
        invalid_rows
        large_return_rows
    """
    required_price_cols = ["Open", "High", "Low", "Close", "Adj Close"]
    expected_cols = required_price_cols + ["Volume"]

    # Some Yahoo versions/symbols may omit corporate-action columns when empty.
    if "Dividends" not in df.columns:
        df["Dividends"] = 0.0
    if "Stock Splits" not in df.columns:
        df["Stock Splits"] = 0.0

    # Check expected columns.
    missing_columns = [c for c in expected_cols if c not in df.columns]

    # Duplicate dates.
    duplicate_date_mask = df.index.duplicated(keep=False)
    duplicate_date_count = int(duplicate_date_mask.sum())

    # Reference-calendar coverage.
    ticker_dates = pd.DatetimeIndex(df.index.unique()).sort_values()
    benchmark_dates = pd.DatetimeIndex(benchmark_dates.unique()).sort_values()

    missing_vs_benchmark = benchmark_dates.difference(ticker_dates)
    extra_vs_benchmark = ticker_dates.difference(benchmark_dates)
    common_dates = ticker_dates.intersection(benchmark_dates)

    benchmark_coverage_pct = safe_pct(len(common_dates), len(benchmark_dates))

    missing_date_rows = [
        {"Ticker": ticker, "Missing_Benchmark_Date": d.strftime("%Y-%m-%d")}
        for d in missing_vs_benchmark
    ]

    # Missing values in core fields.
    null_counts = {}
    for col in expected_cols:
        null_counts[col] = int(df[col].isna().sum()) if col in df.columns else len(df)

    core_null_total = sum(null_counts.values())

    # Invalid numeric values.
    invalid_rows = []
    invalid_price_count = 0
    negative_volume_count = 0
    zero_volume_count = 0
    ohlc_inconsistency_count = 0

    if not missing_columns:
        numeric = df[expected_cols].apply(pd.to_numeric, errors="coerce")

        invalid_price_mask = (numeric[required_price_cols] <= 0).any(axis=1)
        invalid_price_count = int(invalid_price_mask.sum())

        neg_volume_mask = numeric["Volume"] < 0
        negative_volume_count = int(neg_volume_mask.sum())

        zero_volume_mask = numeric["Volume"] == 0
        zero_volume_count = int(zero_volume_mask.sum())

        # OHLC consistency:
        # High should be >= Open/Low/Close and
        # Low should be <= Open/High/Close.
        max_olc = numeric[["Open", "Low", "Close"]].max(axis=1)
        min_ohc = numeric[["Open", "High", "Close"]].min(axis=1)
        ohlc_bad_mask = (numeric["High"] < max_olc) | (numeric["Low"] > min_ohc)
        ohlc_inconsistency_count = int(ohlc_bad_mask.sum())

        issue_mask = (
            invalid_price_mask
            | neg_volume_mask
            | ohlc_bad_mask
        )

        for dt in df.index[issue_mask]:
            reasons = []
            if bool(invalid_price_mask.loc[dt]):
                reasons.append("nonpositive_price")
            if bool(neg_volume_mask.loc[dt]):
                reasons.append("negative_volume")
            if bool(ohlc_bad_mask.loc[dt]):
                reasons.append("ohlc_inconsistency")

            invalid_rows.append({
                "Ticker": ticker,
                "Date": dt.strftime("%Y-%m-%d"),
                "Issue": ";".join(reasons),
            })

    # Large adjusted-close log returns for manual inspection.
    large_return_rows = []
    large_return_count = 0

    if "Adj Close" in df.columns:
        adj = pd.to_numeric(df["Adj Close"], errors="coerce")
        log_ret = np.log(adj / adj.shift(1))
        large_mask = log_ret.abs() > LARGE_ABS_LOG_RETURN_THRESHOLD
        large_return_count = int(large_mask.sum())

        for dt, val in log_ret[large_mask].items():
            large_return_rows.append({
                "Ticker": ticker,
                "Date": dt.strftime("%Y-%m-%d"),
                "Adjusted_Log_Return": float(val),
                "Absolute_Adjusted_Log_Return": float(abs(val)),
            })

    # Corporate actions.
    dividends = pd.to_numeric(df["Dividends"], errors="coerce").fillna(0.0)
    splits = pd.to_numeric(df["Stock Splits"], errors="coerce").fillna(0.0)

    dividend_event_count = int((dividends != 0).sum())
    split_event_count = int((splits != 0).sum())

    # Start/end lags relative to the benchmark.
    first_date = ticker_dates.min() if len(ticker_dates) else pd.NaT
    last_date = ticker_dates.max() if len(ticker_dates) else pd.NaT

    if len(common_dates):
        first_common = common_dates.min()
        last_common = common_dates.max()

        start_lag = int(np.searchsorted(benchmark_dates.values, first_common.to_datetime64()))
        # Number of benchmark trading sessions after the ticker's last common date.
        last_pos = int(np.searchsorted(benchmark_dates.values, last_common.to_datetime64()))
        end_lag = max(0, len(benchmark_dates) - 1 - last_pos)
    else:
        start_lag = len(benchmark_dates)
        end_lag = len(benchmark_dates)

    # Preliminary eligibility only.
    # We intentionally do NOT reject a ticker solely for zero-volume days or
    # large returns; these should first be investigated.
    prelim_eligible = (
        len(missing_columns) == 0
        and duplicate_date_count == 0
        and invalid_price_count == 0
        and negative_volume_count == 0
        and ohlc_inconsistency_count == 0
        and not math.isnan(benchmark_coverage_pct)
        and benchmark_coverage_pct >= MIN_BENCHMARK_COVERAGE_PCT
        and end_lag <= MAX_END_LAG_TRADING_DAYS
    )

    summary = {
        "Ticker": ticker,
        "Is_Benchmark": ticker == BENCHMARK,
        "Rows": int(len(df)),
        "First_Date": as_date_string(first_date),
        "Last_Date": as_date_string(last_date),
        "Benchmark_Trading_Days": int(len(benchmark_dates)),
        "Common_Benchmark_Days": int(len(common_dates)),
        "Missing_Benchmark_Days": int(len(missing_vs_benchmark)),
        "Extra_NonBenchmark_Days": int(len(extra_vs_benchmark)),
        "Benchmark_Coverage_Pct": round(float(benchmark_coverage_pct), 4)
        if not math.isnan(benchmark_coverage_pct) else None,
        "Start_Lag_Trading_Days": int(start_lag),
        "End_Lag_Trading_Days": int(end_lag),
        "Duplicate_Date_Count": duplicate_date_count,
        "Missing_Columns": ",".join(missing_columns),
        "Null_Open": null_counts["Open"],
        "Null_High": null_counts["High"],
        "Null_Low": null_counts["Low"],
        "Null_Close": null_counts["Close"],
        "Null_Adj_Close": null_counts["Adj Close"],
        "Null_Volume": null_counts["Volume"],
        "Core_Null_Total": int(core_null_total),
        "Invalid_Price_Row_Count": invalid_price_count,
        "Negative_Volume_Row_Count": negative_volume_count,
        "Zero_Volume_Row_Count": zero_volume_count,
        "OHLC_Inconsistency_Row_Count": ohlc_inconsistency_count,
        "Large_Adjusted_Log_Return_Count": large_return_count,
        "Dividend_Event_Count": dividend_event_count,
        "Split_Event_Count": split_event_count,
        "Prelim_Eligible": bool(prelim_eligible),
    }

    return summary, missing_date_rows, invalid_rows, large_return_rows


# ============================================================
# 3. MAIN
# ============================================================

def main() -> int:
    ensure_dirs()

    print("=" * 78)
    print("STAGE 1 — INDIAN MARKET DATA FEASIBILITY AUDIT")
    print("=" * 78)
    print(f"yfinance version : {yf.__version__}")
    print(f"Planned period   : {START_DATE} to {PLANNED_END_DATE} inclusive")
    print(f"Stocks           : {len(STOCKS)}")
    print(f"Benchmark        : {BENCHMARK} (NIFTY 50)")
    print(f"Output directory : {OUTPUT_DIR.resolve()}")
    print()

    # ----------------------------
    # Download
    # ----------------------------
    data: dict[str, pd.DataFrame] = {}
    download_log = []

    print("[1/4] Downloading Yahoo Finance data...")
    for ticker in TICKERS:
        df, error = download_one(ticker)

        if df is None:
            download_log.append({
                "Ticker": ticker,
                "Status": "FAILED",
                "Rows": 0,
                "Error": error or "Unknown error",
            })
            print(f"    FAILED: {ticker} -> {error}")
            continue

        data[ticker] = df
        raw_path = RAW_DIR / f"{ticker.replace('^', 'INDEX_')}.csv"
        df.to_csv(raw_path)

        download_log.append({
            "Ticker": ticker,
            "Status": "OK",
            "Rows": len(df),
            "Error": "",
        })
        print(
            f"    OK: {ticker:<16} "
            f"rows={len(df):>5} "
            f"{df.index.min().date()} -> {df.index.max().date()}"
        )

    pd.DataFrame(download_log).to_csv(
        OUTPUT_DIR / "00_download_log.csv",
        index=False,
    )

    if BENCHMARK not in data:
        print("\nERROR: NIFTY 50 benchmark download failed.")
        print("Cannot build the reference NSE trading calendar.")
        return 1

    failed = [x for x in download_log if x["Status"] != "OK"]
    if failed:
        print(f"\nWARNING: {len(failed)} ticker(s) failed to download.")
        print("The audit will continue for successful tickers.")

    # ----------------------------
    # Build benchmark calendar
    # ----------------------------
    print("\n[2/4] Building NIFTY 50 reference trading calendar...")
    benchmark_df = data[BENCHMARK]
    benchmark_dates = pd.DatetimeIndex(benchmark_df.index.unique()).sort_values()

    calendar_df = pd.DataFrame({
        "Date": benchmark_dates.strftime("%Y-%m-%d")
    })
    calendar_df.to_csv(
        OUTPUT_DIR / "01_nifty50_reference_trading_calendar.csv",
        index=False,
    )

    print(
        f"    NIFTY 50 reference sessions: {len(benchmark_dates)} "
        f"({benchmark_dates.min().date()} -> {benchmark_dates.max().date()})"
    )

    # ----------------------------
    # Audit
    # ----------------------------
    print("\n[3/4] Auditing coverage and integrity...")

    summary_rows = []
    missing_rows = []
    invalid_rows = []
    large_return_rows = []

    # Benchmark first, then stocks in frozen order.
    audit_order = [BENCHMARK] + STOCKS

    for ticker in audit_order:
        if ticker not in data:
            continue

        summary, miss, invalid, large = audit_ticker(
            ticker=ticker,
            df=data[ticker].copy(),
            benchmark_dates=benchmark_dates,
        )

        summary_rows.append(summary)
        missing_rows.extend(miss)
        invalid_rows.extend(invalid)
        large_return_rows.extend(large)

    summary_df = pd.DataFrame(summary_rows)

    # Save comprehensive summary.
    summary_df.to_csv(
        OUTPUT_DIR / "02_market_coverage_integrity_summary.csv",
        index=False,
    )

    pd.DataFrame(
        missing_rows,
        columns=["Ticker", "Missing_Benchmark_Date"],
    ).to_csv(
        OUTPUT_DIR / "03_missing_dates_vs_nifty50.csv",
        index=False,
    )

    pd.DataFrame(
        invalid_rows,
        columns=["Ticker", "Date", "Issue"],
    ).to_csv(
        OUTPUT_DIR / "04_invalid_ohlcv_rows.csv",
        index=False,
    )

    pd.DataFrame(
        large_return_rows,
        columns=[
            "Ticker",
            "Date",
            "Adjusted_Log_Return",
            "Absolute_Adjusted_Log_Return",
        ],
    ).to_csv(
        OUTPUT_DIR / "05_large_adjusted_return_flags.csv",
        index=False,
    )

    # Corporate-action event detail.
    corporate_rows = []
    for ticker, df in data.items():
        tmp = df.copy()

        if "Dividends" not in tmp.columns:
            tmp["Dividends"] = 0.0
        if "Stock Splits" not in tmp.columns:
            tmp["Stock Splits"] = 0.0

        div = pd.to_numeric(tmp["Dividends"], errors="coerce").fillna(0.0)
        spl = pd.to_numeric(tmp["Stock Splits"], errors="coerce").fillna(0.0)

        mask = (div != 0) | (spl != 0)

        for dt in tmp.index[mask]:
            corporate_rows.append({
                "Ticker": ticker,
                "Date": dt.strftime("%Y-%m-%d"),
                "Dividend": float(div.loc[dt]),
                "Stock_Split": float(spl.loc[dt]),
            })

    pd.DataFrame(
        corporate_rows,
        columns=["Ticker", "Date", "Dividend", "Stock_Split"],
    ).to_csv(
        OUTPUT_DIR / "06_corporate_actions.csv",
        index=False,
    )

    # Preliminary eligibility table for the 20 stock targets only.
    stock_summary = summary_df[
        summary_df["Ticker"].isin(STOCKS)
    ].copy()

    eligibility_cols = [
        "Ticker",
        "Rows",
        "First_Date",
        "Last_Date",
        "Benchmark_Coverage_Pct",
        "Missing_Benchmark_Days",
        "Core_Null_Total",
        "Duplicate_Date_Count",
        "Invalid_Price_Row_Count",
        "Negative_Volume_Row_Count",
        "Zero_Volume_Row_Count",
        "OHLC_Inconsistency_Row_Count",
        "Large_Adjusted_Log_Return_Count",
        "Dividend_Event_Count",
        "Split_Event_Count",
        "Prelim_Eligible",
    ]

    eligibility_df = stock_summary[eligibility_cols].copy()
    eligibility_df.to_csv(
        OUTPUT_DIR / "07_preliminary_stock_eligibility.csv",
        index=False,
    )

    # ----------------------------
    # Summary JSON
    # ----------------------------
    eligible_count = int(stock_summary["Prelim_Eligible"].sum()) if not stock_summary.empty else 0
    successful_stock_downloads = int(stock_summary.shape[0])

    audit_json = {
        "study_design": {
            "market": "India / NSE",
            "planned_start": START_DATE,
            "planned_end_inclusive": PLANNED_END_DATE,
            "stock_count_planned": len(STOCKS),
            "benchmark": BENCHMARK,
            "source": "Yahoo Finance via yfinance",
        },
        "download": {
            "successful_stock_downloads": successful_stock_downloads,
            "failed_stock_downloads": len(STOCKS) - successful_stock_downloads,
            "benchmark_downloaded": BENCHMARK in data,
        },
        "reference_calendar": {
            "nifty50_sessions": int(len(benchmark_dates)),
            "first_date": as_date_string(benchmark_dates.min()),
            "last_date": as_date_string(benchmark_dates.max()),
        },
        "preliminary_eligibility": {
            "threshold_benchmark_coverage_pct": MIN_BENCHMARK_COVERAGE_PCT,
            "max_end_lag_trading_days": MAX_END_LAG_TRADING_DAYS,
            "eligible_stocks": eligible_count,
            "ineligible_or_failed_stocks": len(STOCKS) - eligible_count,
            "note": (
                "Preliminary engineering audit only. "
                "No stock is removed until the output is reviewed."
            ),
        },
        "flags": {
            "total_missing_benchmark_date_records": len(missing_rows),
            "total_invalid_ohlcv_records": len(invalid_rows),
            "total_large_adjusted_return_flags": len(large_return_rows),
            "total_corporate_action_records": len(corporate_rows),
        },
    }

    with open(
        OUTPUT_DIR / "00_market_audit_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(audit_json, f, indent=2)

    # ----------------------------
    # Terminal output
    # ----------------------------
    print("\n[4/4] Audit complete.")
    print("-" * 78)

    terminal_cols = [
        "Ticker",
        "Rows",
        "Benchmark_Coverage_Pct",
        "Missing_Benchmark_Days",
        "Core_Null_Total",
        "Zero_Volume_Row_Count",
        "Large_Adjusted_Log_Return_Count",
        "Prelim_Eligible",
    ]

    if not stock_summary.empty:
        display_df = stock_summary[terminal_cols].copy()
        print(display_df.to_string(index=False))

    print("\n" + "=" * 78)
    print("COMPACT SUMMARY — COPY/PASTE THIS BACK INTO CHATGPT")
    print("=" * 78)
    print(json.dumps(audit_json, indent=2))
    print("=" * 78)

    print("\nFiles written:")
    for p in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  - {p}")
    print(f"  - {OUTPUT_DIR / '00_market_audit_summary.json'}")
    print(f"  - {RAW_DIR}/  (raw per-ticker Yahoo files)")

    print(
        "\nNEXT: Do not clean, drop, interpolate, or calculate technical "
        "indicators yet. Paste the compact summary and any warnings/errors "
        "back into ChatGPT first."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
