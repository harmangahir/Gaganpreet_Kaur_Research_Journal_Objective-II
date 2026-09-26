#!/usr/bin/env python3
"""
01d_finalize_market_dataset.py

Stage 1D — Finalize the market-data layer.

This script applies the conservative, leakage-safe market cleaning rule after
Stages 1, 1B and 1C:

1. Use Yahoo Finance ^NSEI dates as the common market-context calendar.
2. Require every retained date to be present for all 20 stocks and ^NSEI.
3. Do NOT interpolate missing NIFTY 50 observations.
4. Remove 2025-03-18 from the entire panel because Yahoo reports a systemic
   stock-volume anomaly on that otherwise valid benchmark trading day.
5. Consequently, stock-only dates absent from ^NSEI are excluded uniformly.
6. Raw Stage-1 files remain untouched.

This produces the frozen CLEAN MARKET DATASET only.
It does NOT calculate technical indicators or create modelling sequences yet.
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

OUTPUT_DIR = Path("data") / "stage1_market_final"
STOCK_DIR = OUTPUT_DIR / "stocks"

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

# Systemic Yahoo volume anomaly found in Stage 1B.
SYSTEMIC_BAD_DATES = {
    pd.Timestamp("2025-03-18"): (
        "Systemic Yahoo stock-volume anomaly: 19/20 stocks reported zero volume"
    )
}

CORE_STOCK_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
]

CORE_INDEX_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
]


# ============================================================
# 2. HELPERS
# ============================================================

def raw_path(ticker: str) -> Path:
    return RAW_DIR / f"{ticker.replace('^', 'INDEX_')}.csv"


def load_yahoo_raw(ticker: str) -> pd.DataFrame:
    path = raw_path(ticker)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing Stage-1 raw file for {ticker}: {path}"
        )

    df = pd.read_csv(path, parse_dates=["Date"])

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")

    if df["Date"].duplicated().any():
        duplicates = df.loc[df["Date"].duplicated(keep=False), "Date"]
        raise ValueError(
            f"{ticker}: duplicate dates remain in raw input: "
            f"{duplicates.dt.strftime('%Y-%m-%d').tolist()}"
        )

    df = df.set_index("Date")

    # Ensure corporate-action columns exist for stocks.
    if "Dividends" not in df.columns:
        df["Dividends"] = 0.0
    if "Stock Splits" not in df.columns:
        df["Stock Splits"] = 0.0

    for col in CORE_STOCK_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def fmt_date(x) -> str:
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def check_stock_integrity(ticker: str, df: pd.DataFrame) -> dict:
    missing_cols = [c for c in CORE_STOCK_COLUMNS if c not in df.columns]

    if missing_cols:
        return {
            "Ticker": ticker,
            "Rows": len(df),
            "Missing_Columns": ";".join(missing_cols),
            "Core_Null_Count": None,
            "Duplicate_Dates": int(df.index.duplicated().sum()),
            "Nonpositive_Price_Rows": None,
            "Negative_Volume_Rows": None,
            "Zero_Volume_Rows": None,
            "OHLC_Inconsistency_Rows": None,
            "Passed": False,
        }

    x = df[CORE_STOCK_COLUMNS].copy()

    null_count = int(
        x[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        .isna()
        .sum()
        .sum()
    )

    nonpositive_price = (
        x[["Open", "High", "Low", "Close", "Adj Close"]] <= 0
    ).any(axis=1)

    negative_volume = x["Volume"] < 0
    zero_volume = x["Volume"] == 0

    max_olc = x[["Open", "Low", "Close"]].max(axis=1)
    min_ohc = x[["Open", "High", "Close"]].min(axis=1)

    ohlc_bad = (
        (x["High"] < max_olc)
        | (x["Low"] > min_ohc)
    )

    passed = (
        null_count == 0
        and int(df.index.duplicated().sum()) == 0
        and int(nonpositive_price.sum()) == 0
        and int(negative_volume.sum()) == 0
        and int(zero_volume.sum()) == 0
        and int(ohlc_bad.sum()) == 0
    )

    return {
        "Ticker": ticker,
        "Rows": int(len(df)),
        "Missing_Columns": "",
        "Core_Null_Count": null_count,
        "Duplicate_Dates": int(df.index.duplicated().sum()),
        "Nonpositive_Price_Rows": int(nonpositive_price.sum()),
        "Negative_Volume_Rows": int(negative_volume.sum()),
        "Zero_Volume_Rows": int(zero_volume.sum()),
        "OHLC_Inconsistency_Rows": int(ohlc_bad.sum()),
        "Passed": bool(passed),
    }


# ============================================================
# 3. MAIN
# ============================================================

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STOCK_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("STAGE 1D — FINALIZE CLEAN MARKET DATASET")
    print("=" * 88)

    # --------------------------------------------------------
    # Load untouched Stage-1 Yahoo files
    # --------------------------------------------------------
    print("\n[1/6] Loading Stage-1 raw Yahoo files...")

    stock_raw = {ticker: load_yahoo_raw(ticker) for ticker in STOCKS}
    nifty_raw = load_yahoo_raw(BENCHMARK)

    print(f"  Loaded {len(stock_raw)} stocks")
    print(f"  Loaded benchmark {BENCHMARK}: {len(nifty_raw)} rows")

    # --------------------------------------------------------
    # Construct conservative common calendar
    # --------------------------------------------------------
    print("\n[2/6] Constructing common trading calendar...")

    nifty_dates = set(nifty_raw.index)

    common_dates = set(nifty_dates)
    for ticker in STOCKS:
        common_dates &= set(stock_raw[ticker].index)

    common_dates = {
        pd.Timestamp(d).normalize()
        for d in common_dates
        if pd.Timestamp(d).normalize() not in SYSTEMIC_BAD_DATES
    }

    final_calendar = pd.DatetimeIndex(sorted(common_dates))

    if len(final_calendar) == 0:
        raise RuntimeError("Final common calendar is empty.")

    print(
        f"  Final calendar: {len(final_calendar)} sessions "
        f"({final_calendar.min().date()} -> {final_calendar.max().date()})"
    )

    # --------------------------------------------------------
    # Build complete exclusion audit
    # --------------------------------------------------------
    print("\n[3/6] Building exclusion audit trail...")

    union_stock_dates = set().union(
        *[set(df.index) for df in stock_raw.values()]
    )

    all_observed_dates = sorted(union_stock_dates | set(nifty_raw.index))
    final_date_set = set(final_calendar)

    exclusion_rows = []

    for dt in all_observed_dates:
        dt = pd.Timestamp(dt).normalize()

        if dt in final_date_set:
            continue

        stock_count = sum(
            dt in stock_raw[ticker].index
            for ticker in STOCKS
        )
        nifty_present = dt in nifty_raw.index

        if dt in SYSTEMIC_BAD_DATES:
            reason = SYSTEMIC_BAD_DATES[dt]
        elif stock_count == len(STOCKS) and not nifty_present:
            reason = (
                "Excluded uniformly: all 20 stocks present but Yahoo ^NSEI "
                "market-context row unavailable; no interpolation used"
            )
        elif stock_count > 0 and not nifty_present:
            reason = (
                "Excluded: stock observations present but Yahoo ^NSEI "
                "market-context row unavailable"
            )
        elif nifty_present and stock_count < len(STOCKS):
            reason = (
                "Excluded: not all 20 stocks available on benchmark date"
            )
        else:
            reason = "Excluded by common-calendar intersection"

        exclusion_rows.append({
            "Date": fmt_date(dt),
            "Stocks_With_Row": int(stock_count),
            "NIFTY50_Row_Present": bool(nifty_present),
            "Reason": reason,
        })

    exclusion_df = pd.DataFrame(exclusion_rows)
    exclusion_df.to_csv(
        OUTPUT_DIR / "01_excluded_dates_audit.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Create cleaned per-stock files
    # --------------------------------------------------------
    print("\n[4/6] Creating final per-stock market files...")

    integrity_rows = []
    panel_parts = []

    for ticker in STOCKS:
        df = stock_raw[ticker].loc[final_calendar].copy()

        # Hard assertion: exact final calendar for every stock.
        if not df.index.equals(final_calendar):
            raise RuntimeError(
                f"{ticker}: final date index differs from frozen calendar."
            )

        integrity = check_stock_integrity(ticker, df)
        integrity_rows.append(integrity)

        out = df[CORE_STOCK_COLUMNS].copy()
        out.insert(0, "Ticker", ticker)

        stock_path = STOCK_DIR / f"{ticker}.csv"
        out.reset_index().to_csv(stock_path, index=False)

        panel_parts.append(out.reset_index())

        print(
            f"  {ticker:<16} "
            f"rows={len(out):>5} "
            f"zero_volume={integrity['Zero_Volume_Rows']} "
            f"passed={integrity['Passed']}"
        )

    integrity_df = pd.DataFrame(integrity_rows)
    integrity_df.to_csv(
        OUTPUT_DIR / "02_final_stock_integrity.csv",
        index=False,
    )

    if not integrity_df["Passed"].all():
        failed = integrity_df.loc[
            ~integrity_df["Passed"], "Ticker"
        ].tolist()
        raise RuntimeError(
            "Final stock integrity check failed for: "
            + ", ".join(failed)
        )

    # --------------------------------------------------------
    # Create cleaned NIFTY market-context file
    # --------------------------------------------------------
    print("\n[5/6] Creating final NIFTY 50 context file...")

    missing_index_cols = [
        c for c in CORE_INDEX_COLUMNS
        if c not in nifty_raw.columns
    ]
    if missing_index_cols:
        raise RuntimeError(
            "NIFTY raw file missing columns: "
            + ", ".join(missing_index_cols)
        )

    nifty_final = nifty_raw.loc[final_calendar, CORE_INDEX_COLUMNS].copy()

    if nifty_final[CORE_INDEX_COLUMNS].isna().any().any():
        raise RuntimeError(
            "Null values remain in final NIFTY 50 context data."
        )

    if (
        nifty_final[["Open", "High", "Low", "Close", "Adj Close"]] <= 0
    ).any().any():
        raise RuntimeError(
            "Non-positive price found in final NIFTY 50 context data."
        )

    nifty_final.reset_index().to_csv(
        OUTPUT_DIR / "03_nifty50_market_context.csv",
        index=False,
    )

    # Final calendar
    pd.DataFrame({
        "Date": final_calendar.strftime("%Y-%m-%d")
    }).to_csv(
        OUTPUT_DIR / "04_final_common_trading_calendar.csv",
        index=False,
    )

    # Long-format stock panel
    panel = pd.concat(panel_parts, ignore_index=True)
    panel = panel.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    panel.to_csv(
        OUTPUT_DIR / "05_market_panel_long.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------
    print("\n[6/6] Final verification...")

    expected_panel_rows = len(final_calendar) * len(STOCKS)

    if len(panel) != expected_panel_rows:
        raise RuntimeError(
            f"Panel row count mismatch: {len(panel)} != "
            f"{expected_panel_rows}"
        )

    # Every date must have exactly 20 stocks.
    count_by_date = panel.groupby("Date")["Ticker"].nunique()

    if not (count_by_date == len(STOCKS)).all():
        bad_dates = count_by_date[count_by_date != len(STOCKS)]
        raise RuntimeError(
            "Some final dates do not contain all 20 stocks:\n"
            + bad_dates.to_string()
        )

    # Every stock must have exact same number of dates.
    count_by_ticker = panel.groupby("Ticker")["Date"].nunique()

    if not (count_by_ticker == len(final_calendar)).all():
        raise RuntimeError(
            "Stocks do not share an identical final calendar."
        )

    # Additional panel manifest
    manifest = {
        "market": "India / NSE",
        "market_source": "Yahoo Finance via yfinance",
        "stocks": STOCKS,
        "benchmark": BENCHMARK,
        "planned_window": {
            "start": "2016-01-01",
            "end": "2026-06-30",
        },
        "effective_final_window": {
            "start": fmt_date(final_calendar.min()),
            "end": fmt_date(final_calendar.max()),
        },
        "final_trading_sessions_per_stock": int(len(final_calendar)),
        "final_stock_count": len(STOCKS),
        "final_stock_panel_rows": int(len(panel)),
        "nifty50_context_rows": int(len(nifty_final)),
        "excluded_observed_dates": int(len(exclusion_df)),
        "cleaning_policy": {
            "calendar": (
                "Intersection of all 20 stock dates and Yahoo ^NSEI dates"
            ),
            "nifty_missing_dates": (
                "Excluded uniformly; no index interpolation or back/forward "
                "filling"
            ),
            "systemic_bad_dates": {
                fmt_date(k): v
                for k, v in SYSTEMIC_BAD_DATES.items()
            },
            "raw_data_modified": False,
        },
        "integrity": {
            "all_20_stocks_passed": bool(
                integrity_df["Passed"].all()
            ),
            "final_zero_volume_stock_rows": int(
                integrity_df["Zero_Volume_Rows"].sum()
            ),
            "final_core_null_count": int(
                integrity_df["Core_Null_Count"].sum()
            ),
            "final_duplicate_dates": int(
                integrity_df["Duplicate_Dates"].sum()
            ),
            "final_invalid_price_rows": int(
                integrity_df["Nonpositive_Price_Rows"].sum()
            ),
            "final_ohlc_inconsistency_rows": int(
                integrity_df["OHLC_Inconsistency_Rows"].sum()
            ),
        },
    }

    with open(
        OUTPUT_DIR / "00_stage1_market_final_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 88)
    print("FINAL MARKET DATASET SUMMARY — COPY/PASTE THIS BACK INTO CHATGPT")
    print("=" * 88)
    print(json.dumps(manifest, indent=2))
    print("=" * 88)

    print("\nExcluded observed dates:")
    if exclusion_df.empty:
        print("  None")
    else:
        print(exclusion_df.to_string(index=False))

    print("\nFiles written:")
    for p in sorted(OUTPUT_DIR.glob("*")):
        if p.is_file():
            print(f"  - {p}")
    print(f"  - {STOCK_DIR}/  (20 cleaned per-stock files)")

    print(
        "\nNEXT: Stop here and paste the FINAL MARKET DATASET SUMMARY "
        "and excluded-date table into ChatGPT. Do not calculate technical "
        "indicators yet."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
