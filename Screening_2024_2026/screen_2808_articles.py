#!/usr/bin/env python3
"""
screen_2808_articles.py

Transparent, deterministic, Python-first title/abstract screening pipeline
for the 2024–2026 stock-forecasting discovery corpus assembled from:

- Scopus CSV exports
- ScienceDirect BibTeX exports
- Elicit BibTeX exports

The script:
1. reads all source exports from one folder;
2. normalizes DOI/title metadata;
3. deduplicates across databases;
4. applies transparent inclusion/exclusion rules;
5. assigns Include / Maybe / Exclude / Background;
6. adds explicit exclusion codes and evidence terms;
7. adds thematic tags for technical, fundamental, sentiment, multimodal,
   optimization, XAI, temporal validation, etc.;
8. produces a manual-QC queue;
9. writes summary files for manuscript reporting.

IMPORTANT:
- "Unclear" means "not evidenced in title/abstract/keywords"; it does NOT mean absent from the full paper.
- All Maybe records should be manually reviewed.
- A QC sample of Include/Exclude records should be manually checked before final counts are frozen.
- Automated counts are provisional until manual QC is complete.

Usage:
    python screen_2808_articles.py \
        --input-dir /path/to/all_exports \
        --output-dir screening_results \
        --expect-count 2808

Standard-library only. No pandas / sklearn required.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# 1. FROZEN SCREENING CONFIGURATION
# ============================================================

YEAR_WINDOW = {2024, 2025, 2026}

QC_SEED = 20260925
QC_INCLUDE_FRACTION = 0.05
QC_EXCLUDE_FRACTION = 0.10

EXCLUSION_CODES = {
    "E01": "Cryptocurrency/digital-asset forecasting without stock/equity forecasting",
    "E02": "Forex/currency forecasting without stock/equity forecasting",
    "E03": "Commodity/energy forecasting without stock/equity forecasting",
    "E04": "Portfolio optimization/asset allocation without an explicit eligible stock forecasting task",
    "E05": "Trading/RL strategy without an explicit eligible stock forecasting task",
    "E06": "Non-financial time-series forecasting / no stock-equity context",
    "E07": "Sentiment analysis related to stocks but no explicit eligible stock forecasting task",
    "E08": "Pure econometric/statistical forecasting without AI/ML/DL",
    "E09": "Review/survey/bibliometric paper retained as Background",
    "E10": "Editorial/commentary/non-primary research item",
    "E11": "Outside frozen 2024–2026 publication window",
    "E12": "Stock-market topic but no eligible price/return/direction/trend forecasting task",
    "E13": "AI/ML forecasting present but direct stock/equity relevance not established",
    "E14": "Duplicate",
    "E15": "Insufficient metadata",
}

# Core stock/equity domain
STOCK_TERMS = [
    "stock price", "stock market", "stock movement", "stock direction",
    "stock return", "stock trend", "share price", "share market",
    "equity price", "equity market", "equity return", "stock index",
    "market index", "stock forecasting", "stock prediction", "stock-market",
    "s&p 500", "s&p500", "nasdaq", "dow jones", "djia",
    "nifty", "sensex", "shanghai stock", "sse composite", "csi 300",
    "hang seng", "nikkei", "kospi", "ftse", "stoxx", "bovespa",
    "nepse", "bist", "bombay stock exchange", "national stock exchange",
    "a-share",
]

PREDICTION_TERMS = [
    "predict", "prediction", "forecast", "forecasting", "future price",
    "price movement", "directional", "movement prediction",
    "trend prediction", "return prediction", "return forecasting",
    "price estimation", "next-day", "next day", "multi-step",
    "multistep", "multi-horizon", "multihorizon",
]

AI_TERMS = [
    "machine learning", "deep learning", "artificial intelligence",
    "neural network", "lstm", "gru", "cnn", "rnn", "transformer",
    "attention", "xgboost", "lightgbm", "catboost", "random forest",
    "support vector", "svm", "gradient boosting", "mlp",
    "graph neural", "gnn", "finbert", "bert", "roberta", "fingpt",
    "large language model", "llm", "temporal convolution", "tcn",
    "ensemble learning", "hybrid model", "deep neural",
    "convolutional", "recurrent neural",
]

FINANCIAL_TERMS = [
    "financial market", "finance", "financial time series", "market data",
    "asset price", "securities", "investment", "trading", "portfolio",
    "capital market",
]

# Exclusion-domain terms
CRYPTO_TERMS = [
    "bitcoin", "ethereum", "cryptocurrency", "crypto currency",
    "cryptoasset", "crypto asset", "digital asset price", "digital currency",
]

FOREX_TERMS = [
    "forex", "foreign exchange", "exchange rate", "currency exchange",
    "currency market", "fx market",
]

COMMODITY_TERMS = [
    "crude oil", "oil price", "gold price", "commodity price",
    "commodity market", "electricity price", "energy price",
    "natural gas price", "carbon price", "metal price",
]

NONFINANCIAL_TERMS = [
    "traffic flow", "weather forecast", "weather prediction",
    "load forecasting", "electric load", "energy consumption",
    "wind power", "solar power", "water demand", "disease forecasting",
    "covid-19 cases", "air quality", "sales forecasting",
    "tourism demand", "crop yield",
]

REVIEW_TERMS = [
    "systematic review", "systematic literature review", "literature review",
    "bibliometric", "comprehensive survey", "a survey of", "survey on",
    "review of", "scoping review", "meta-analysis",
]

NONRESEARCH_TERMS = [
    "editorial", "commentary", "preface", "perspective article",
    "corrigendum", "erratum",
]

PORTFOLIO_TERMS = [
    "portfolio optimization", "portfolio optimisation", "asset allocation",
    "mean-variance", "portfolio selection",
]

TRADING_TERMS = [
    "algorithmic trading", "trading strategy", "trading strategies",
    "market making", "trade execution", "reinforcement learning trading",
]

ECONOMETRIC_TERMS = [
    "arima", "sarima", "garch", "vector autoregression", "econometric",
    "ets model", "prophet",
]

SENTIMENT_TERMS = [
    "sentiment", "polarity", "emotion analysis", "opinion mining",
]

VOLATILITY_TERMS = [
    "volatility prediction", "volatility forecasting", "forecast volatility",
    "predict volatility", "value at risk", "option pricing",
]

# Thematic tagging terms
HISTORICAL_TERMS = [
    "historical price", "price history", "ohlcv", "open high low close",
    "closing price", "close price", "market data", "time series", "volume",
]

TECHNICAL_TERMS = [
    "technical indicator", "technical analysis", "rsi",
    "relative strength index", "macd", "bollinger", "sma", "ema",
    "moving average", "stochastic oscillator", "atr",
    "average true range", "adx", "aroon", "momentum indicator", "cci",
]

FUNDAMENTAL_TERMS = [
    "fundamental analysis", "fundamental indicator", "financial ratio",
    "financial ratios", "financial statement", "balance sheet",
    "income statement", "cash flow", "earnings per share", "eps",
    "return on equity", "roe", "return on assets", "roa",
    "price-to-earnings", "p/e", "price to earnings", "book value",
    "debt-to-equity", "debt to equity", "revenue growth",
    "profit margin", "earnings",
]

NEWS_TERMS = [
    "financial news", "news headline", "news headlines", "news article",
    "news articles", "market news", "business news", "earnings call",
    "corporate disclosure", "corporate disclosures",
]

SOCIAL_TERMS = [
    "twitter", "tweet", "tweets", "reddit", "social media",
    "stocktwits", "weibo",
]

BERT_TERMS = ["finbert", "bert", "roberta"]

LLM_TERMS = [
    "large language model", "llm", "fingpt", "chatgpt", "gpt-4", "gpt4",
]

MULTIMODAL_TERMS = [
    "multimodal", "multi-modal", "multi source", "multi-source",
    "multisource", "multiple data sources", "heterogeneous data",
    "cross-modal", "cross modal",
]

FUSION_TERMS = [
    "fusion", "feature fusion", "data fusion", "decision-level",
    "decision level", "feature-level", "feature level", "early fusion",
    "late fusion", "intermediate fusion", "cross-attention",
    "cross attention", "gated fusion",
]

TRANSFORMER_TERMS = [
    "transformer", "attention mechanism", "self-attention",
    "self attention", "informer", "itransformer",
    "temporal fusion transformer",
]

LSTM_GRU_TERMS = [
    "lstm", "gru", "long short-term memory", "long short term memory",
    "gated recurrent", "recurrent neural", "rnn", "bilstm", "bi-lstm",
    "bigru", "bi-gru",
]

OPTIMIZATION_TERMS = [
    "hyperparameter optimization", "hyperparameter optimisation",
    "hyperparameter tuning", "bayesian optimization",
    "bayesian optimisation", "grid search", "random search", "optuna",
    "metaheuristic", "evolutionary algorithm", "swarm intelligence",
    "optimized", "optimised",
]

METAHEURISTIC_TERMS = [
    "particle swarm", "pso", "grey wolf", "gray wolf", "gwo",
    "artificial rabbits", "genetic algorithm", "differential evolution",
    "whale optimization", "sparrow search", "firefly", "ant colony",
    "bee colony", "cuckoo search", "beluga whale", "moth flame",
    "bat algorithm",
]

XAI_TERMS = [
    "explainable", "explainability", "interpretable", "interpretability",
    "xai", "shap", "lime", "feature importance",
    "attention visualization", "attention visualisation",
]

TEMPORAL_TERMS = [
    "walk-forward", "walk forward", "rolling window", "rolling-window",
    "expanding window", "expanding-window", "chronological split",
    "temporal split", "time-series cross-validation",
    "time series cross validation", "out-of-sample", "out of sample",
    "forward validation", "purged",
]

INDIA_TERMS = [
    "india", "indian stock", "nifty", "nse", "sensex", "bse",
    "bombay stock exchange", "national stock exchange",
]

MULTISTOCK_TERMS = [
    "multi-stock", "multiple stocks", "multiple companies",
    "several stocks", "cross-market", "cross market", "cross-country",
    "cross country", "global indices", "multiple indices",
    "five stocks", "six stocks", "ten stocks", "20 stocks",
]


# ============================================================
# 2. BASIC TEXT / FILE UTILITIES
# ============================================================

def clean_text(s: str | None) -> str:
    s = s or ""
    s = s.replace("\\&", "&").replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def normalize_doi(s: str | None) -> str:
    s = clean_text(s).lower()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s.rstrip(" .;,}")


def normalize_title(s: str | None) -> str:
    s = unicodedata.normalize("NFKD", clean_text(s))
    s = s.encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def make_uid(title: str, doi: str) -> str:
    doi = normalize_doi(doi)
    return "doi:" + doi if doi else "title:" + normalize_title(title)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def phrase_hits(text: str, phrases: list[str]) -> list[str]:
    """
    Token-aware phrase matching to avoid false positives for short terms
    such as NSE, BSE, RSI, SVM, GRU, etc.
    """
    t = (text or "").lower()
    hits = set()

    for phrase in phrases:
        p = phrase.lower()
        pattern = re.escape(p)

        left = r"(?<![a-z0-9])" if p and p[0].isalnum() else ""
        right = r"(?![a-z0-9])" if p and p[-1].isalnum() else ""

        if re.search(left + pattern + right, t):
            hits.add(phrase)

    return sorted(hits)


# ============================================================
# 3. BIBTEX PARSER
# ============================================================

def bib_entry_chunks(text: str) -> list[str]:
    starts = [m.start() for m in re.finditer(r"(?m)^@\w+\s*[\{\(]", text)]
    starts.append(len(text))
    return [text[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]


def parse_bib_entry(chunk: str) -> dict | None:
    m0 = re.match(r"@(\w+)\s*[\{\(]\s*([^,]+),", chunk, re.S)
    if not m0:
        return None

    entry_type, key = m0.group(1), m0.group(2).strip()
    body = chunk[m0.end():]

    fields = {}
    pos = 0

    while pos < len(body):
        m = re.search(
            r'(?m)^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*([\{"])',
            body[pos:]
        )
        if not m:
            break

        name = m.group(1).lower()
        opener = m.group(2)
        value_start = pos + m.end()

        if opener == "{":
            depth = 1
            j = value_start
            while j < len(body) and depth:
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                j += 1

            value = body[value_start:j - 1] if depth == 0 else body[value_start:]
            pos = j

        else:
            j = value_start
            escaped = False
            while j < len(body):
                ch = body[j]
                if ch == '"' and not escaped:
                    break
                escaped = (ch == "\\" and not escaped)
                if ch != "\\":
                    escaped = False
                j += 1

            value = body[value_start:j]
            pos = j + 1

        fields[name] = re.sub(r"\s+", " ", value).strip()

    fields["_type"] = entry_type
    fields["_key"] = key
    return fields


def parse_bib(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", errors="ignore") as f:
        text = f.read()

    out = []
    for chunk in bib_entry_chunks(text):
        entry = parse_bib_entry(chunk)
        if entry:
            out.append(entry)

    return out


def source_query_id(filename: str) -> str:
    m = re.search(r"Scopus_Query-(\d+)_", filename)
    if m:
        return "SC" + m.group(1).zfill(2)

    m = re.search(r"ScienceDirect_Query-(\d+)_", filename)
    if m:
        return "SD" + m.group(1).zfill(2)

    m = re.match(r"EL(\d+)_", filename)
    if m:
        return "EL" + m.group(1).zfill(2)

    return ""


# ============================================================
# 4. MERGE + DEDUPLICATE ALL SOURCES
# ============================================================

def merge_sources(input_dir: Path, output_dir: Path, expect_count: int | None):
    merged = {}
    source_manifest = []
    out_of_window = []

    # ---------------- Scopus CSV ----------------
    scopus_files = sorted(input_dir.glob("Scopus_Query-*_Records.csv"))

    for path in scopus_files:
        qid = source_query_id(path.name)
        rows = read_csv(path)

        source_manifest.append({
            "Source_File": path.name,
            "Database": "Scopus",
            "Query_ID": qid,
            "Parsed_Records": len(rows),
            "SHA256": sha256_file(path),
        })

        for row in rows:
            title = row.get("Title", "")
            doi = normalize_doi(row.get("DOI", ""))

            try:
                year = int(row.get("Year", ""))
            except Exception:
                year = row.get("Year", "")

            if year not in YEAR_WINDOW:
                out_of_window.append({
                    "Database": "Scopus",
                    "Query_ID": qid,
                    "Title": title,
                    "Year": year,
                    "DOI": doi,
                    "Source_File": path.name,
                })
                continue

            uid = make_uid(title, doi)

            r = merged.get(uid, {
                "UID": uid,
                "Title": title,
                "Authors": row.get("Authors", ""),
                "Year": year,
                "Venue": row.get("Source title", ""),
                "DOI": doi,
                "Abstract": row.get("Abstract", "") or "",
                "Keywords": (
                    (row.get("Author Keywords", "") or "")
                    + ("; " + row.get("Index Keywords", "")
                       if row.get("Index Keywords", "") else "")
                ),
                "Document_Type": row.get("Document Type", "") or "",
                "Scopus": "No",
                "ScienceDirect": "No",
                "Elicit": "No",
                "Query_Hits": set(),
            })

            r["Scopus"] = "Yes"
            r["Query_Hits"].add(qid)

            merged[uid] = r

    # ---------------- ScienceDirect BibTeX ----------------
    sd_files = sorted(input_dir.glob("ScienceDirect_Query-*.bib"))

    for path in sd_files:
        qid = source_query_id(path.name)
        entries = parse_bib(path)

        source_manifest.append({
            "Source_File": path.name,
            "Database": "ScienceDirect",
            "Query_ID": qid,
            "Parsed_Records": len(entries),
            "SHA256": sha256_file(path),
        })

        for e in entries:
            title = clean_text(e.get("title", ""))
            doi = normalize_doi(e.get("doi", ""))

            try:
                year = int(e.get("year", ""))
            except Exception:
                year = e.get("year", "")

            if year not in YEAR_WINDOW:
                out_of_window.append({
                    "Database": "ScienceDirect",
                    "Query_ID": qid,
                    "Title": title,
                    "Year": year,
                    "DOI": doi,
                    "Source_File": path.name,
                })
                continue

            uid = make_uid(title, doi)

            r = merged.get(uid, {
                "UID": uid,
                "Title": title,
                "Authors": clean_text(e.get("author", "")),
                "Year": year,
                "Venue": clean_text(e.get("journal", "")),
                "DOI": doi,
                "Abstract": clean_text(e.get("abstract", "")),
                "Keywords": clean_text(e.get("keywords", "")),
                "Document_Type": e.get("_type", "").title(),
                "Scopus": "No",
                "ScienceDirect": "No",
                "Elicit": "No",
                "Query_Hits": set(),
            })

            r["ScienceDirect"] = "Yes"
            r["Query_Hits"].add(qid)

            if not r["Abstract"] and clean_text(e.get("abstract", "")):
                r["Abstract"] = clean_text(e.get("abstract", ""))

            if not r["Keywords"] and clean_text(e.get("keywords", "")):
                r["Keywords"] = clean_text(e.get("keywords", ""))

            if not r["Venue"]:
                r["Venue"] = clean_text(e.get("journal", ""))

            merged[uid] = r

    # ---------------- Elicit BibTeX ----------------
    elicit_files = sorted(input_dir.glob("EL0*.bib"))

    for path in elicit_files:
        qid = source_query_id(path.name)
        entries = parse_bib(path)

        source_manifest.append({
            "Source_File": path.name,
            "Database": "Elicit",
            "Query_ID": qid,
            "Parsed_Records": len(entries),
            "SHA256": sha256_file(path),
        })

        for e in entries:
            title = clean_text(e.get("title", ""))
            doi = normalize_doi(e.get("doi", ""))

            try:
                year = int(e.get("year", ""))
            except Exception:
                year = e.get("year", "")

            if year not in YEAR_WINDOW:
                out_of_window.append({
                    "Database": "Elicit",
                    "Query_ID": qid,
                    "Title": title,
                    "Year": year,
                    "DOI": doi,
                    "Source_File": path.name,
                })
                continue

            uid = make_uid(title, doi)

            r = merged.get(uid, {
                "UID": uid,
                "Title": title,
                "Authors": clean_text(e.get("author", "")),
                "Year": year,
                "Venue": clean_text(e.get("journal", "")),
                "DOI": doi,
                "Abstract": clean_text(e.get("abstract", "")),
                "Keywords": clean_text(e.get("keywords", "")),
                "Document_Type": e.get("_type", "").title(),
                "Scopus": "No",
                "ScienceDirect": "No",
                "Elicit": "No",
                "Query_Hits": set(),
            })

            r["Elicit"] = "Yes"
            r["Query_Hits"].add(qid)

            if not r["Abstract"] and clean_text(e.get("abstract", "")):
                r["Abstract"] = clean_text(e.get("abstract", ""))

            if not r["Keywords"] and clean_text(e.get("keywords", "")):
                r["Keywords"] = clean_text(e.get("keywords", ""))

            merged[uid] = r

    master = []

    for i, uid in enumerate(sorted(merged), 1):
        r = merged[uid].copy()
        r["Master_ID"] = f"M-{i:04d}"
        r["Query_Hits"] = "; ".join(sorted(r["Query_Hits"]))
        master.append(r)

    if expect_count is not None and len(master) != expect_count:
        raise RuntimeError(
            f"Corpus integrity check failed: expected {expect_count} unique records "
            f"but obtained {len(master)}."
        )

    master_fields = [
        "Master_ID", "UID", "Title", "Authors", "Year", "Venue",
        "DOI", "Abstract", "Keywords", "Document_Type",
        "Scopus", "ScienceDirect", "Elicit", "Query_Hits",
    ]

    write_csv(
        output_dir / "stage1_master_deduplicated.csv",
        master,
        master_fields,
    )

    write_csv(
        output_dir / "stage0_source_manifest.csv",
        source_manifest,
        ["Source_File", "Database", "Query_ID", "Parsed_Records", "SHA256"],
    )

    write_csv(
        output_dir / "stage0_out_of_window.csv",
        out_of_window,
        ["Database", "Query_ID", "Title", "Year", "DOI", "Source_File"],
    )

    return master, source_manifest, out_of_window


# ============================================================
# 5. AUTOMATED SCREENING
# ============================================================

def evidence_level(record: dict) -> str:
    abstract = (record.get("Abstract") or "").strip()
    keywords = (record.get("Keywords") or "").strip()

    if len(abstract) >= 80:
        return "Abstract available"
    if abstract:
        return "Short abstract"
    if keywords:
        return "Title + keywords only"
    return "Title only"


def screen_record(record: dict) -> dict:
    title = record.get("Title", "") or ""
    abstract = record.get("Abstract", "") or ""
    keywords = record.get("Keywords", "") or ""

    text = " ".join([title, abstract, keywords]).lower()

    stock_hits = phrase_hits(text, STOCK_TERMS)
    prediction_hits = phrase_hits(text, PREDICTION_TERMS)
    ai_hits = phrase_hits(text, AI_TERMS)
    financial_hits = phrase_hits(text, FINANCIAL_TERMS)

    crypto_hits = phrase_hits(text, CRYPTO_TERMS)
    forex_hits = phrase_hits(text, FOREX_TERMS)
    commodity_hits = phrase_hits(text, COMMODITY_TERMS)
    nonfinancial_hits = phrase_hits(text, NONFINANCIAL_TERMS)

    review_hits = phrase_hits(title.lower(), REVIEW_TERMS)
    nonresearch_hits = phrase_hits(title.lower(), NONRESEARCH_TERMS)

    portfolio_hits = phrase_hits(text, PORTFOLIO_TERMS)
    trading_hits = phrase_hits(text, TRADING_TERMS)
    econometric_hits = phrase_hits(text, ECONOMETRIC_TERMS)
    sentiment_hits = phrase_hits(text, SENTIMENT_TERMS)
    volatility_hits = phrase_hits(text, VOLATILITY_TERMS)

    core_target_hits = phrase_hits(text, [
        "stock price", "share price", "equity price",
        "price prediction", "price forecasting",
        "stock movement", "movement prediction",
        "stock direction", "direction prediction", "directional",
        "stock return", "equity return",
        "return prediction", "return forecasting",
        "stock trend", "trend prediction",
    ])

    stock = bool(stock_hits)
    prediction = bool(prediction_hits)
    ai = bool(ai_hits)
    financial = bool(financial_hits)

    decision = "Maybe"
    exclusion_code = ""
    reason = "Available metadata are insufficient for a confident eligibility decision."

    if review_hits:
        decision = "Background"
        exclusion_code = "E09"
        reason = EXCLUSION_CODES["E09"]

    elif nonresearch_hits:
        decision = "Exclude"
        exclusion_code = "E10"
        reason = EXCLUSION_CODES["E10"]

    elif crypto_hits and not stock:
        decision = "Exclude"
        exclusion_code = "E01"
        reason = EXCLUSION_CODES["E01"]

    elif forex_hits and not stock:
        decision = "Exclude"
        exclusion_code = "E02"
        reason = EXCLUSION_CODES["E02"]

    elif commodity_hits and not stock:
        decision = "Exclude"
        exclusion_code = "E03"
        reason = EXCLUSION_CODES["E03"]

    elif nonfinancial_hits and not stock and not financial:
        decision = "Exclude"
        exclusion_code = "E06"
        reason = EXCLUSION_CODES["E06"]

    elif stock and portfolio_hits and not core_target_hits:
        decision = "Exclude"
        exclusion_code = "E04"
        reason = EXCLUSION_CODES["E04"]

    elif stock and trading_hits and not core_target_hits:
        decision = "Exclude"
        exclusion_code = "E05"
        reason = EXCLUSION_CODES["E05"]

    elif stock and sentiment_hits and not prediction:
        decision = "Exclude"
        exclusion_code = "E07"
        reason = EXCLUSION_CODES["E07"]

    elif stock and volatility_hits and not core_target_hits:
        decision = "Exclude"
        exclusion_code = "E12"
        reason = (
            "Stock-market study focuses on volatility/risk rather than an eligible "
            "price/return/direction/trend forecasting target."
        )

    elif stock and not prediction:
        decision = "Exclude"
        exclusion_code = "E12"
        reason = EXCLUSION_CODES["E12"]

    elif stock and prediction and not ai:
        if econometric_hits:
            decision = "Exclude"
            exclusion_code = "E08"
            reason = EXCLUSION_CODES["E08"]
        else:
            decision = "Maybe"
            reason = (
                "Stock forecasting is explicit, but AI/ML/DL use is unclear "
                "from available metadata."
            )

    elif stock and prediction and ai:
        decision = "Include"
        reason = "Explicit stock/equity forecasting task using AI/ML/DL."

    elif stock and ai and not prediction:
        decision = "Maybe"
        reason = (
            "Stock/equity context and AI/ML are present, "
            "but the exact forecasting task is unclear."
        )

    elif financial and prediction and ai and not stock:
        decision = "Maybe"
        reason = (
            "AI-based financial forecasting is present, "
            "but a stock/equity target is not explicit."
        )

    elif prediction and ai and not stock:
        decision = "Exclude"
        exclusion_code = "E13"
        reason = EXCLUSION_CODES["E13"]

    elif financial and ai and not prediction:
        decision = "Exclude"
        exclusion_code = "E13"
        reason = (
            "Financial AI topic is present, but an eligible "
            "stock forecasting task is not established."
        )

    elif not stock and not financial:
        decision = "Exclude"
        exclusion_code = "E06"
        reason = EXCLUSION_CODES["E06"]

    out = dict(record)

    out.update({
        "Evidence_Level": evidence_level(record),
        "Screening_Decision": decision,
        "Exclusion_Code": exclusion_code,
        "Exclusion_Reason": reason,
        "Stock_Evidence": "; ".join(stock_hits[:15]),
        "Prediction_Evidence": "; ".join(prediction_hits[:15]),
        "AI_Evidence": "; ".join(ai_hits[:15]),
        "NonTarget_Evidence": "; ".join(
            (crypto_hits + forex_hits + commodity_hits
             + nonfinancial_hits + volatility_hits)[:15]
        ),
    })

    return out


# ============================================================
# 6. THEMATIC TAGGING
# ============================================================

TAG_GROUPS = {
    "Historical_Data": HISTORICAL_TERMS,
    "Technical_Indicators": TECHNICAL_TERMS,
    "Fundamental_Data": FUNDAMENTAL_TERMS,
    "Financial_News": NEWS_TERMS,
    "Social_Media": SOCIAL_TERMS,
    "Sentiment": SENTIMENT_TERMS,
    "Multimodal": MULTIMODAL_TERMS,
    "Fusion": FUSION_TERMS,
    "Transformer_Attention": TRANSFORMER_TERMS,
    "LSTM_GRU": LSTM_GRU_TERMS,
    "Optimization": OPTIMIZATION_TERMS,
    "Metaheuristic": METAHEURISTIC_TERMS,
    "XAI": XAI_TERMS,
    "Temporal_Validation": TEMPORAL_TERMS,
    "Indian_Market": INDIA_TERMS,
    "Multi_Stock": MULTISTOCK_TERMS,
}


def detect_tag(text: str, terms: list[str]) -> tuple[str, str]:
    hits = phrase_hits(text, terms)
    return ("Yes" if hits else "Unclear"), "; ".join(hits[:15])


def detect_nlp_model(text: str) -> str:
    bert = bool(phrase_hits(text, BERT_TERMS))
    llm = bool(phrase_hits(text, LLM_TERMS))

    if bert and llm:
        return "Both"
    if bert:
        return "FinBERT/BERT/RoBERTa"
    if llm:
        return "LLM/FinGPT"
    return "Unclear"


def detect_prediction_target(text: str) -> str:
    hits = []

    if phrase_hits(text, [
        "stock price", "share price", "equity price",
        "closing price", "price prediction", "price forecasting"
    ]):
        hits.append("Price")

    if phrase_hits(text, [
        "stock movement", "movement prediction",
        "stock direction", "direction prediction", "directional"
    ]):
        hits.append("Direction/Movement")

    if phrase_hits(text, [
        "stock return", "equity return",
        "return prediction", "return forecasting"
    ]):
        hits.append("Return")

    if phrase_hits(text, [
        "stock trend", "trend prediction", "market trend"
    ]):
        hits.append("Trend")

    if len(hits) > 1:
        return "Multiple"
    if hits:
        return hits[0]
    return "Unclear"


def assign_primary_theme(title: str, tags: dict) -> str:
    title = title.lower()

    title_rules = [
        ("T7 Explainability/XAI", XAI_TERMS),
        ("T8 Temporal validation/leakage/robustness", TEMPORAL_TERMS),
        ("T6 Optimization/metaheuristics/HPO",
         OPTIMIZATION_TERMS + METAHEURISTIC_TERMS),
        ("T4 Multimodal/data fusion", MULTIMODAL_TERMS + FUSION_TERMS),
        ("T3 Financial news/sentiment",
         NEWS_TERMS + SENTIMENT_TERMS + BERT_TERMS + LLM_TERMS),
        ("T2 Fundamental-data forecasting", FUNDAMENTAL_TERMS),
        ("T5 Transformer/attention", TRANSFORMER_TERMS),
        ("T1 Historical/technical forecasting", TECHNICAL_TERMS),
    ]

    for label, terms in title_rules:
        if phrase_hits(title, terms):
            return label

    fallback = [
        ("Historical_Data", "T1 Historical/technical forecasting"),
        ("Fundamental_Data", "T2 Fundamental-data forecasting"),
        ("Sentiment", "T3 Financial news/sentiment"),
        ("Multimodal", "T4 Multimodal/data fusion"),
        ("Transformer_Attention", "T5 Transformer/attention"),
        ("Optimization", "T6 Optimization/metaheuristics/HPO"),
        ("XAI", "T7 Explainability/XAI"),
        ("Temporal_Validation", "T8 Temporal validation/leakage/robustness"),
    ]

    for key, label in fallback:
        if tags.get(key) == "Yes":
            return label

    return "T9 Other relevant AI stock forecasting"


def add_methodological_tags(record: dict) -> dict:
    text = " ".join([
        record.get("Title", ""),
        record.get("Abstract", ""),
        record.get("Keywords", ""),
    ]).lower()

    out = dict(record)
    tags = {}

    for tag_name, terms in TAG_GROUPS.items():
        value, evidence = detect_tag(text, terms)
        tags[tag_name] = value
        out[tag_name] = value
        out[tag_name + "_Evidence"] = evidence

    out["NLP_Model"] = detect_nlp_model(text)
    out["Prediction_Target"] = detect_prediction_target(text)
    out["Primary_Theme"] = assign_primary_theme(
        record.get("Title", ""),
        tags
    )

    theme_code_by_tag = {
        "Historical_Data": "T1",
        "Fundamental_Data": "T2",
        "Sentiment": "T3",
        "Multimodal": "T4",
        "Transformer_Attention": "T5",
        "Optimization": "T6",
        "XAI": "T7",
        "Temporal_Validation": "T8",
    }

    secondary = []
    for tag_name, theme_code in theme_code_by_tag.items():
        if tags.get(tag_name) == "Yes" and not out["Primary_Theme"].startswith(theme_code):
            secondary.append(theme_code)

    out["Secondary_Themes"] = "; ".join(secondary)

    decision = out["Screening_Decision"]

    if decision == "Exclude":
        relevance = 0

    elif decision == "Background":
        relevance = 1

    elif decision == "Maybe":
        relevance = 2

    else:
        focused = sum(
            tags.get(k) == "Yes"
            for k in [
                "Technical_Indicators",
                "Fundamental_Data",
                "Sentiment",
                "Multimodal",
                "Fusion",
                "Optimization",
                "XAI",
                "Temporal_Validation",
            ]
        )

        if (
            tags.get("Fundamental_Data") == "Yes"
            and tags.get("Technical_Indicators") == "Yes"
            and (
                tags.get("Sentiment") == "Yes"
                or tags.get("Financial_News") == "Yes"
            )
        ):
            relevance = 5

        elif focused >= 3:
            relevance = 5

        elif focused >= 1:
            relevance = 4

        else:
            relevance = 3

    out["Relevance_Score"] = relevance

    priority = 0

    if decision in {"Include", "Maybe"}:
        priority = 2
        priority += 2 if tags["Fundamental_Data"] == "Yes" else 0
        priority += 1 if tags["Technical_Indicators"] == "Yes" else 0
        priority += 1 if tags["Sentiment"] == "Yes" else 0
        priority += 2 if tags["Multimodal"] == "Yes" else 0
        priority += 1 if tags["Fusion"] == "Yes" else 0
        priority += 1 if tags["Optimization"] == "Yes" else 0
        priority += 1 if tags["XAI"] == "Yes" else 0
        priority += 2 if tags["Temporal_Validation"] == "Yes" else 0
        priority += 1 if tags["Indian_Market"] == "Yes" else 0
        priority += 1 if tags["Multi_Stock"] == "Yes" else 0

    out["Priority_Score"] = priority
    out["Needs_Full_Text"] = (
        "Yes" if decision in {"Include", "Maybe"}
        else "Optional" if decision == "Background"
        else "No"
    )

    return out


# ============================================================
# 7. QC SAMPLE
# ============================================================

def deterministic_sample(pool: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if n >= len(pool):
        return list(pool)

    idx = list(range(len(pool)))
    rng.shuffle(idx)

    chosen = sorted(idx[:n])
    return [pool[i] for i in chosen]


def build_manual_review_queue(tagged: list[dict]) -> list[dict]:
    included = [r for r in tagged if r["Screening_Decision"] == "Include"]
    excluded = [r for r in tagged if r["Screening_Decision"] == "Exclude"]
    maybe = [r for r in tagged if r["Screening_Decision"] == "Maybe"]
    background = [r for r in tagged if r["Screening_Decision"] == "Background"]

    include_n = math.ceil(len(included) * QC_INCLUDE_FRACTION)
    exclude_n = math.ceil(len(excluded) * QC_EXCLUDE_FRACTION)

    include_qc = deterministic_sample(included, include_n, QC_SEED)
    exclude_qc = deterministic_sample(excluded, exclude_n, QC_SEED + 1)

    queue = []

    def add_rows(rows, queue_reason):
        for r in rows:
            queue.append({
                "Master_ID": r["Master_ID"],
                "Queue_Reason": queue_reason,
                "AI_Decision": r["Screening_Decision"],
                "Title": r["Title"],
                "Year": r["Year"],
                "Venue": r["Venue"],
                "DOI": r["DOI"],
                "Abstract": r["Abstract"],
                "Keywords": r["Keywords"],
                "Exclusion_Code": r["Exclusion_Code"],
                "AI_Reason": r["Exclusion_Reason"],
                "Primary_Theme": r.get("Primary_Theme", ""),
                "Priority_Score": r.get("Priority_Score", ""),
                "Human_Decision": "",
                "Human_Exclusion_Code": "",
                "Reviewer_Notes": "",
            })

    add_rows(maybe, "All Maybe — mandatory manual review")
    add_rows(background, "All Background — verify classification")
    add_rows(include_qc, "QC sample — Include")
    add_rows(exclude_qc, "QC sample — Exclude")

    return queue


# ============================================================
# 8. SUMMARY STATISTICS
# ============================================================

def create_summary(tagged: list[dict]) -> dict:
    decisions = Counter(r["Screening_Decision"] for r in tagged)

    exclusions = Counter(
        r["Exclusion_Code"]
        for r in tagged
        if r["Screening_Decision"] == "Exclude"
    )

    years = defaultdict(Counter)
    for r in tagged:
        years[str(r["Year"])][r["Screening_Decision"]] += 1

    themes = Counter(
        r.get("Primary_Theme", "")
        for r in tagged
        if r["Screening_Decision"] in {"Include", "Maybe"}
    )

    tag_names = [
        "Historical_Data", "Technical_Indicators", "Fundamental_Data",
        "Financial_News", "Social_Media", "Sentiment", "Multimodal",
        "Fusion", "Transformer_Attention", "LSTM_GRU", "Optimization",
        "Metaheuristic", "XAI", "Temporal_Validation", "Indian_Market",
        "Multi_Stock",
    ]

    tag_counts = {
        tag: sum(
            r.get(tag) == "Yes"
            for r in tagged
            if r["Screening_Decision"] in {"Include", "Maybe"}
        )
        for tag in tag_names
    }

    source_counts = {
        "Scopus": sum(r["Scopus"] == "Yes" for r in tagged),
        "ScienceDirect": sum(r["ScienceDirect"] == "Yes" for r in tagged),
        "Elicit": sum(r["Elicit"] == "Yes" for r in tagged),
    }

    return {
        "screening_version": "1.0",
        "corpus_size": len(tagged),
        "decisions": dict(decisions),
        "full_text_queue_include_plus_maybe": (
            decisions.get("Include", 0) + decisions.get("Maybe", 0)
        ),
        "exclusion_codes": dict(exclusions),
        "decision_by_year": {
            y: dict(counter) for y, counter in years.items()
        },
        "primary_themes_include_plus_maybe": dict(themes),
        "tag_counts_include_plus_maybe": tag_counts,
        "source_membership_counts": source_counts,
        "note": (
            "Provisional automated title/abstract screening. "
            "Final eligibility requires manual resolution of all Maybe records "
            "and QC audit of sampled Include/Exclude records."
        ),
    }


# ============================================================
# 9. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Transparent Python-first screening of the 2024–2026 stock-forecasting literature corpus."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing all Scopus CSV, ScienceDirect BibTeX, and Elicit BibTeX exports.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder where screening outputs will be written.",
    )

    parser.add_argument(
        "--expect-count",
        type=int,
        default=2808,
        help="Expected unique deduplicated corpus size. Default: 2808.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== STAGE 1: Merge + deduplicate ===")
    master, source_manifest, out_of_window = merge_sources(
        input_dir,
        output_dir,
        args.expect_count,
    )

    print(f"Unique deduplicated records: {len(master)}")
    print(f"Out-of-window records logged: {len(out_of_window)}")

    print("\n=== STAGE 2: Automated title/abstract screening ===")
    screened = [screen_record(r) for r in master]

    screen_fields = list(screened[0].keys())

    write_csv(
        output_dir / "stage2_auto_screened.csv",
        screened,
        screen_fields,
    )

    for label, filename in [
        ("Include", "stage2_included.csv"),
        ("Maybe", "stage2_maybe.csv"),
        ("Exclude", "stage2_excluded.csv"),
        ("Background", "stage2_background.csv"),
    ]:
        subset = [r for r in screened if r["Screening_Decision"] == label]
        write_csv(output_dir / filename, subset, screen_fields)

    decision_counts = Counter(r["Screening_Decision"] for r in screened)

    for label in ["Include", "Maybe", "Exclude", "Background"]:
        print(f"{label}: {decision_counts.get(label, 0)}")

    print("\n=== STAGE 3: Methodological tagging ===")
    tagged = [add_methodological_tags(r) for r in screened]
    tagged_fields = list(tagged[0].keys())

    write_csv(
        output_dir / "stage3_tagged.csv",
        tagged,
        tagged_fields,
    )

    core_tags = [
        "Technical_Indicators", "Fundamental_Data", "Financial_News",
        "Sentiment", "Multimodal", "Fusion", "Transformer_Attention",
        "Optimization", "Metaheuristic", "XAI", "Temporal_Validation",
    ]

    core_priority = [
        r for r in tagged
        if r["Screening_Decision"] in {"Include", "Maybe"}
        and any(r.get(tag) == "Yes" for tag in core_tags)
    ]

    core_priority = sorted(
        core_priority,
        key=lambda r: (
            -int(r.get("Priority_Score") or 0),
            int(r["Year"]) if str(r["Year"]).isdigit() else 9999,
            r["Title"].lower(),
        )
    )

    write_csv(
        output_dir / "stage3_core_priority_shortlist.csv",
        core_priority,
        tagged_fields,
    )

    print(f"Core-theme Include/Maybe records: {len(core_priority)}")

    print("\n=== STAGE 4: Manual QC queue ===")
    manual_queue = build_manual_review_queue(tagged)

    manual_fields = [
        "Master_ID", "Queue_Reason", "AI_Decision", "Title", "Year",
        "Venue", "DOI", "Abstract", "Keywords", "Exclusion_Code",
        "AI_Reason", "Primary_Theme", "Priority_Score",
        "Human_Decision", "Human_Exclusion_Code", "Reviewer_Notes",
    ]

    write_csv(
        output_dir / "stage4_manual_review_queue.csv",
        manual_queue,
        manual_fields,
    )

    print(f"Manual review queue: {len(manual_queue)}")

    print("\n=== STAGE 5: Summary ===")
    summary = create_summary(tagged)

    with open(
        output_dir / "stage5_screening_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    exclusion_summary = [
        {
            "Exclusion_Code": code,
            "Reason": EXCLUSION_CODES.get(code, ""),
            "Count": count,
        }
        for code, count in sorted(
            Counter(
                r["Exclusion_Code"]
                for r in tagged
                if r["Screening_Decision"] == "Exclude"
            ).items()
        )
    ]

    write_csv(
        output_dir / "stage5_exclusion_summary.csv",
        exclusion_summary,
        ["Exclusion_Code", "Reason", "Count"],
    )

    print(json.dumps(summary, indent=2))

    print("\n============================================================")
    print("SCREENING COMPLETE")
    print("============================================================")
    print(f"Outputs written to: {output_dir}")
    print()
    print("IMPORTANT:")
    print("- Automated counts are provisional.")
    print("- Manually review ALL Maybe records.")
    print("- Manually review the QC samples.")
    print("- Only after QC should final manuscript / PRISMA counts be frozen.")
    print("============================================================")


if __name__ == "__main__":
    main()
