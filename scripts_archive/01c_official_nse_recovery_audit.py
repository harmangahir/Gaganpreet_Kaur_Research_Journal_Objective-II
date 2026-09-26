#!/usr/bin/env python3
"""
01c_official_nse_recovery_audit.py

Stage 1C — Official NSE recovery audit.

Purpose
-------
Before cleaning Yahoo Finance data, recover/verify two source-level anomalies
using official NSE endpoints:

A) NIFTY 50 rows missing from Yahoo ^NSEI on six stock-trading dates:
   2016-01-01
   2016-08-12
   2018-01-01
   2019-01-01
   2019-10-27
   2020-11-14

B) NSE equity traded quantity on 2025-03-18 for the 20-stock panel,
   because Yahoo reports zero volume for 19 of the 20 stocks on that date.

IMPORTANT
---------
This script is diagnostic only. It does NOT alter the Stage-1 Yahoo files.

Run:
    pip install -U requests pandas
    python 01c_official_nse_recovery_audit.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# ============================================================
# 1. CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("market_data_audit") / "stage1c_official_nse_recovery"
RAW_JSON_DIR = OUTPUT_DIR / "raw_json"

BASE_URL = "https://www.nseindia.com/"
INDEX_API = "https://www.nseindia.com/api/historical/indicesHistory"
SECURITY_API = "https://www.nseindia.com/api/historical/securityArchives"

MISSING_NIFTY_DATES = [
    "2016-01-01",
    "2016-08-12",
    "2018-01-01",
    "2019-01-01",
    "2019-10-27",
    "2020-11-14",
]

CHECK_DATE = "2025-03-18"

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

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/reports-indices-historical-index-data",
    "Connection": "keep-alive",
}

TIMEOUT = 20
REQUEST_PAUSE = 1.0


# ============================================================
# 2. HELPERS
# ============================================================

def ddmmyyyy(iso_date: str) -> str:
    return pd.Timestamp(iso_date).strftime("%d-%m-%Y")


def safe_float(x: Any):
    if x is None:
        return None
    if isinstance(x, str):
        x = x.replace(",", "").strip()
        if x in {"", "-", "--", "NA", "N/A", "null", "None"}:
            return None
    try:
        return float(x)
    except Exception:
        return None


def safe_int(x: Any):
    val = safe_float(x)
    if val is None:
        return None
    try:
        return int(round(val))
    except Exception:
        return None


def date_from_any(x: Any):
    if x is None:
        return None
    try:
        # dayfirst=True handles NSE's common DD-Mon-YYYY / DD-MM-YYYY formats.
        ts = pd.to_datetime(x, dayfirst=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return None


def recursive_dicts(obj: Any):
    """Yield every dictionary contained anywhere inside a JSON object."""
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from recursive_dicts(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from recursive_dicts(item)


def first_present(d: dict, keys: list[str]):
    for key in keys:
        if key in d and d[key] not in (None, "", "-", "--"):
            return d[key]
    return None


def warm_session() -> tuple[requests.Session, int | None, str | None]:
    s = requests.Session()
    try:
        r = s.get(BASE_URL, headers=HEADERS, timeout=TIMEOUT)
        return s, r.status_code, None
    except Exception as exc:
        return s, None, f"{type(exc).__name__}: {exc}"


def get_json(
    session: requests.Session,
    url: str,
    params: dict,
    raw_name: str,
) -> tuple[Any | None, int | None, str | None]:
    try:
        response = session.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=TIMEOUT,
        )
        status = response.status_code

        # Save exact response text for debugging if NSE changes schema.
        text_path = RAW_JSON_DIR / f"{raw_name}.txt"
        text_path.write_text(response.text, encoding="utf-8")

        if status != 200:
            return None, status, f"HTTP {status}"

        try:
            data = response.json()
        except Exception as exc:
            return None, status, f"JSON decode error: {exc}"

        json_path = RAW_JSON_DIR / f"{raw_name}.json"
        json_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return data, status, None

    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def parse_index_record(payload: Any, wanted_date: str) -> dict | None:
    """
    NSE has changed the index-history JSON schema over time.
    Search recursively for a dictionary containing the target date and OHLC keys.
    """
    date_keys = [
        "EOD_TIMESTAMP",
        "TIMESTAMP",
        "HistoricalDate",
        "Date",
        "date",
        "CH_TIMESTAMP",
    ]
    open_keys = [
        "EOD_OPEN_INDEX_VAL",
        "OPEN_INDEX_VAL",
        "OPEN",
        "Open",
        "open",
    ]
    high_keys = [
        "EOD_HIGH_INDEX_VAL",
        "HIGH_INDEX_VAL",
        "HIGH",
        "High",
        "high",
    ]
    low_keys = [
        "EOD_LOW_INDEX_VAL",
        "LOW_INDEX_VAL",
        "LOW",
        "Low",
        "low",
    ]
    close_keys = [
        "EOD_CLOSE_INDEX_VAL",
        "CLOSE_INDEX_VAL",
        "CLOSE",
        "Close",
        "close",
    ]
    shares_keys = [
        "EOD_TRADED_QTY",
        "EOD_SHARES_TRADED",
        "SHARES_TRADED",
        "Shares Traded",
    ]
    turnover_keys = [
        "EOD_TURN_OVER",
        "EOD_TURNOVER",
        "TURNOVER",
        "Turnover",
    ]

    candidates = []

    for d in recursive_dicts(payload):
        raw_date = first_present(d, date_keys)
        parsed_date = date_from_any(raw_date)

        if parsed_date == wanted_date:
            candidates.append(d)

    if not candidates:
        return None

    # Prefer a record that contains the most OHLC fields.
    def score(d):
        fields = [
            first_present(d, open_keys),
            first_present(d, high_keys),
            first_present(d, low_keys),
            first_present(d, close_keys),
        ]
        return sum(x is not None for x in fields)

    best = sorted(candidates, key=score, reverse=True)[0]

    return {
        "Date": wanted_date,
        "Open": safe_float(first_present(best, open_keys)),
        "High": safe_float(first_present(best, high_keys)),
        "Low": safe_float(first_present(best, low_keys)),
        "Close": safe_float(first_present(best, close_keys)),
        "Shares_Traded": safe_int(first_present(best, shares_keys)),
        "Turnover": safe_float(first_present(best, turnover_keys)),
        "Raw_Record_Keys": ";".join(sorted(best.keys())),
    }


def parse_security_record(payload: Any, wanted_date: str, symbol: str) -> dict | None:
    """
    Parse NSE historical securityArchives record flexibly.
    """
    date_keys = [
        "CH_TIMESTAMP",
        "TIMESTAMP",
        "Date",
        "date",
    ]
    open_keys = ["CH_OPENING_PRICE", "OPEN", "Open"]
    high_keys = ["CH_TRADE_HIGH_PRICE", "HIGH", "High"]
    low_keys = ["CH_TRADE_LOW_PRICE", "LOW", "Low"]
    close_keys = ["CH_CLOSING_PRICE", "CLOSE", "Close"]
    qty_keys = [
        "CH_TOT_TRADED_QTY",
        "TOT_TRADED_QTY",
        "Total Traded Quantity",
        "Volume",
        "volume",
    ]
    deliverable_keys = [
        "COP_DELIV_QTY",
        "CH_DELIV_QTY",
        "DELIV_QTY",
        "Deliverable Quantity",
    ]

    candidates = []

    for d in recursive_dicts(payload):
        parsed_date = date_from_any(first_present(d, date_keys))
        if parsed_date == wanted_date:
            candidates.append(d)

    if not candidates:
        return None

    def score(d):
        fields = [
            first_present(d, open_keys),
            first_present(d, high_keys),
            first_present(d, low_keys),
            first_present(d, close_keys),
            first_present(d, qty_keys),
        ]
        return sum(x is not None for x in fields)

    best = sorted(candidates, key=score, reverse=True)[0]

    return {
        "Ticker": f"{symbol}.NS",
        "NSE_Symbol": symbol,
        "Date": wanted_date,
        "Open_NSE": safe_float(first_present(best, open_keys)),
        "High_NSE": safe_float(first_present(best, high_keys)),
        "Low_NSE": safe_float(first_present(best, low_keys)),
        "Close_NSE": safe_float(first_present(best, close_keys)),
        "Total_Traded_Qty_NSE": safe_int(first_present(best, qty_keys)),
        "Deliverable_Qty_NSE": safe_int(first_present(best, deliverable_keys)),
        "Raw_Record_Keys": ";".join(sorted(best.keys())),
    }


# ============================================================
# 3. MAIN
# ============================================================

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print("STAGE 1C — OFFICIAL NSE RECOVERY AUDIT")
    print("=" * 92)

    session, warm_status, warm_error = warm_session()

    print(f"NSE homepage warm-up status: {warm_status}")
    if warm_error:
        print(f"NSE homepage warm-up error : {warm_error}")

    # --------------------------------------------------------
    # A. Recover missing NIFTY 50 rows
    # --------------------------------------------------------
    print("\n[A] Official NIFTY 50 historical rows")
    print("-" * 92)

    index_rows = []

    for iso_date in MISSING_NIFTY_DATES:
        params = {
            "indexType": "NIFTY 50",
            "from": ddmmyyyy(iso_date),
            "to": ddmmyyyy(iso_date),
        }

        raw_name = f"index_NIFTY50_{iso_date.replace('-', '')}"
        payload, status, error = get_json(
            session,
            INDEX_API,
            params,
            raw_name,
        )

        parsed = parse_index_record(payload, iso_date) if payload is not None else None

        row = {
            "Date": iso_date,
            "HTTP_Status": status,
            "Request_Error": error,
            "Recovered": parsed is not None,
            "Open": None,
            "High": None,
            "Low": None,
            "Close": None,
            "Shares_Traded": None,
            "Turnover": None,
            "Raw_Record_Keys": None,
        }

        if parsed:
            row.update(parsed)

        index_rows.append(row)

        print(
            f"  {iso_date}: "
            f"HTTP={status}  recovered={row['Recovered']}  "
            f"close={row['Close']}"
        )

        time.sleep(REQUEST_PAUSE)

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(
        OUTPUT_DIR / "01_official_nifty50_recovery.csv",
        index=False,
    )

    # --------------------------------------------------------
    # B. Recover official stock volume for 2025-03-18
    # --------------------------------------------------------
    print("\n[B] Official NSE stock rows for 2025-03-18")
    print("-" * 92)

    security_rows = []

    for ticker in STOCKS:
        symbol = ticker.replace(".NS", "")

        params = {
            "from": ddmmyyyy(CHECK_DATE),
            "to": ddmmyyyy(CHECK_DATE),
            "symbol": symbol,
            "dataType": "priceVolumeDeliverable",
            "series": "EQ",
        }

        raw_name = f"security_{symbol}_{CHECK_DATE.replace('-', '')}"
        payload, status, error = get_json(
            session,
            SECURITY_API,
            params,
            raw_name,
        )

        parsed = (
            parse_security_record(payload, CHECK_DATE, symbol)
            if payload is not None
            else None
        )

        row = {
            "Ticker": ticker,
            "NSE_Symbol": symbol,
            "Date": CHECK_DATE,
            "HTTP_Status": status,
            "Request_Error": error,
            "Recovered": parsed is not None,
            "Open_NSE": None,
            "High_NSE": None,
            "Low_NSE": None,
            "Close_NSE": None,
            "Total_Traded_Qty_NSE": None,
            "Deliverable_Qty_NSE": None,
            "Raw_Record_Keys": None,
        }

        if parsed:
            row.update(parsed)

        security_rows.append(row)

        print(
            f"  {symbol:<12} "
            f"HTTP={status}  recovered={row['Recovered']}  "
            f"qty={row['Total_Traded_Qty_NSE']}"
        )

        time.sleep(REQUEST_PAUSE)

    security_df = pd.DataFrame(security_rows)
    security_df.to_csv(
        OUTPUT_DIR / "02_official_nse_stock_recovery_2025-03-18.csv",
        index=False,
    )

    # --------------------------------------------------------
    # C. Summary
    # --------------------------------------------------------
    recovered_index = int(index_df["Recovered"].sum())
    recovered_security = int(security_df["Recovered"].sum())

    summary = {
        "nifty50_recovery": {
            "requested_dates": len(MISSING_NIFTY_DATES),
            "recovered_dates": recovered_index,
            "unrecovered_dates": len(MISSING_NIFTY_DATES) - recovered_index,
            "dates": MISSING_NIFTY_DATES,
        },
        "stock_volume_recovery_2025_03_18": {
            "requested_stocks": len(STOCKS),
            "recovered_stocks": recovered_security,
            "unrecovered_stocks": len(STOCKS) - recovered_security,
            "positive_official_volume_rows": int(
                (
                    pd.to_numeric(
                        security_df["Total_Traded_Qty_NSE"],
                        errors="coerce",
                    ) > 0
                ).sum()
            ),
        },
        "note": (
            "Diagnostic only. No Yahoo row has been overwritten or removed. "
            "If official NSE recovery succeeds, Stage 1D will create the "
            "reconciled market dataset with a full audit trail."
        ),
    }

    with open(
        OUTPUT_DIR / "00_stage1c_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 92)
    print("COMPACT SUMMARY — COPY/PASTE THIS BACK INTO CHATGPT")
    print("=" * 92)
    print(json.dumps(summary, indent=2))
    print("=" * 92)

    if recovered_index < len(MISSING_NIFTY_DATES):
        print(
            "\nNOTE: Some NIFTY rows were not recovered. "
            "This can happen if NSE changes the API schema or blocks automated access. "
            "Do not manually interpolate them yet."
        )

    if recovered_security < len(STOCKS):
        print(
            "\nNOTE: Some security rows were not recovered. "
            "Do not impute the 2025-03-18 volume anomaly yet."
        )

    print("\nFiles written:")
    for p in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  - {p}")
    print(f"  - {OUTPUT_DIR / '00_stage1c_summary.json'}")
    print(f"  - {RAW_JSON_DIR}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
