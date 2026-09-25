#!/usr/bin/env python3
"""
literature_relevance_mining_v2.py

Purpose
-------
This script is NOT a systematic-review eligibility engine.

It mines the frozen 2024–2026 literature corpus for a RESEARCH ARTICLE on
AI-based stock forecasting and prioritizes papers that can support:
    1. Related Work
    2. Research Gap
    3. Methodology Design
    4. Baseline Selection
    5. State-of-the-Art Comparison

The target research direction is:
    Historical/market data
    + Technical indicators
    + Fundamental variables
    + Financial news/sentiment
    + Multimodal/fusion modelling
    + Advanced AI/DL architectures
    + Optimization/metaheuristics
    + Explainability
    + Temporally sound validation

This is a transparent, deterministic, Python-first evidence-mining pipeline.
It reconstructs and deduplicates Scopus, ScienceDirect and Elicit exports,
profiles each paper from title/abstract/keywords, and assigns WORKING
RELEVANCE TIERS. The tiers are reading priorities, NOT study-quality scores
and NOT systematic-review inclusion/exclusion decisions.

Expected source files:
    Scopus_Query-*_Records.csv
    ScienceDirect_Query-*.bib
    EL0*.bib

Example:
    python literature_relevance_mining_v2.py \
        --input-dir ../source_files \
        --output-dir literature_mining_v2 \
        --expect-count 2808 \
        --development-set adjudicated_58.csv

Standard-library only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# 1. FROZEN PROJECT CONFIGURATION
# ============================================================

YEAR_WINDOW = {2024, 2025, 2026}

# Strong stock/equity evidence.
STOCK_TERMS = [
    "stock price", "stock prices", "stock market", "stock markets",
    "stock movement", "stock direction", "stock return", "stock returns",
    "stock trend", "stock trends", "share price", "share prices",
    "equity price", "equity prices", "equity market", "equity markets",
    "equity return", "equity returns", "stock index", "stock indices",
    "market index", "market indices", "stock forecasting", "stock prediction",
    "stock-price", "stock-market", "s&p 500", "s&p500", "nasdaq",
    "dow jones", "djia", "nifty", "nifty 50", "sensex",
    "shanghai stock", "sse composite", "csi 300", "hang seng",
    "nikkei", "kospi", "ftse", "stoxx", "bovespa", "nepse", "bist",
    "bombay stock exchange", "national stock exchange", "a-share",
    "equity index", "equity indices",
]

# Explicit eligible targets for the proposed research article.
PRICE_TERMS = [
    "stock price", "stock prices", "share price", "share prices",
    "equity price", "equity prices", "closing price", "closing prices",
    "close price", "close prices", "price prediction", "price forecasting",
    "price forecast", "future price", "future prices", "index value",
    "index level", "stock index forecasting", "stock index prediction",
]
RETURN_TERMS = [
    "stock return", "stock returns", "equity return", "equity returns",
    "return prediction", "return forecasting", "forecasting returns",
    "predicting returns", "market return", "market returns",
    "index return", "index returns",
]
DIRECTION_TERMS = [
    "stock movement", "stock movements", "price movement", "price movements",
    "movement prediction", "direction prediction", "directional prediction",
    "market direction", "price direction", "up/down", "up or down",
    "rise or fall", "bullish or bearish",
]
TREND_TERMS = [
    "stock trend", "stock trends", "trend prediction", "trend forecasting",
    "market trend", "market trends", "price trend", "price trends",
]

FORECAST_ACTION_TERMS = [
    "predict", "prediction", "forecast", "forecasting", "estimate",
    "estimation", "next-day", "next day", "future", "multi-step",
    "multistep", "multi-horizon", "multihorizon",
]

AI_TERMS = [
    "machine learning", "deep learning", "artificial intelligence",
    "neural network", "neural networks", "lstm", "gru", "cnn", "rnn",
    "transformer", "attention", "xgboost", "lightgbm", "catboost",
    "random forest", "support vector", "svm", "gradient boosting", "mlp",
    "graph neural", "gnn", "finbert", "bert", "roberta", "fingpt",
    "large language model", "large language models", "llm", "llms",
    "temporal convolution", "tcn", "ensemble learning", "hybrid model",
    "hybrid models", "deep neural", "convolutional", "recurrent neural",
    "artificial neural network", "artificial neural networks",
]

EMPIRICAL_TERMS = [
    "experiment", "experiments", "experimental", "dataset", "datasets",
    "data set", "data sets", "evaluation", "evaluated", "evaluate",
    "results show", "results demonstrate", "outperform", "outperforms",
    "benchmark", "benchmarks", "comparison", "comparative",
    "rmse", "mae", "mape", "mse", "r2", "r-squared", "accuracy",
    "precision", "recall", "f1", "auc", "backtest", "backtesting",
    "training", "testing", "train set", "test set", "validation set",
]

# Background / non-primary publication types.
REVIEW_TERMS = [
    "systematic review", "systematic literature review", "literature review",
    "bibliometric", "bibliometric analysis", "review article",
    "comprehensive survey", "survey on", "a survey of", "scoping review",
    "meta-analysis",
]
BOOK_CHAPTER_TERMS = ["chapter ", "book chapter"]

# Explicit adjacent targets useful as context but not central to the model.
VOLATILITY_RISK_TERMS = [
    "volatility forecasting", "volatility prediction", "forecast volatility",
    "predict volatility", "stock market volatility", "equity volatility",
    "systemic risk", "risk prediction", "value at risk", "var prediction",
]
PORTFOLIO_TERMS = [
    "portfolio optimization", "portfolio optimisation", "portfolio selection",
    "asset allocation", "stock selection", "investment ranking",
]
TRADING_ONLY_TERMS = [
    "trading strategy", "trading strategies", "algorithmic trading",
    "trading decision", "trading decisions", "buy/sell", "buy and sell",
    "market making", "trade execution",
]
SENTIMENT_ONLY_TERMS = [
    "sentiment classification", "sentiment analysis", "investor sentiment",
    "market sentiment", "emotion analysis", "opinion mining",
]
OTHER_FINANCE_TERMS = [
    "financial market", "financial markets", "finance", "financial time series",
    "market data", "asset price", "asset prices", "securities", "investment",
    "trading", "portfolio", "capital market", "financial forecasting",
]
NON_STOCK_DOMAIN_TERMS = {
    "Crypto": [
        "bitcoin", "ethereum", "cryptocurrency", "crypto currency",
        "cryptoasset", "crypto asset", "digital currency",
    ],
    "Forex": [
        "forex", "foreign exchange", "exchange rate", "currency market",
        "fx market",
    ],
    "Commodity": [
        "crude oil", "oil price", "gold price", "commodity price",
        "commodity market", "natural gas price", "carbon price",
    ],
    "Nonfinancial": [
        "traffic flow", "urban traffic", "weather forecast", "load forecasting",
        "electric load", "wind power", "solar power", "water demand",
        "disease forecasting", "covid-19 cases", "air quality",
        "tourism demand", "crop yield", "icd-10", "medical coding",
        "economic emission dispatch", "power system",
    ],
}

# Strategic themes for the proposed article.
HISTORICAL_TERMS = [
    "historical price", "historical prices", "price history", "ohlcv",
    "open high low close", "open-high-low-close", "closing price",
    "market data", "time series", "trading volume", "volume",
]
TECHNICAL_TERMS = [
    "technical indicator", "technical indicators", "technical analysis",
    "rsi", "relative strength index", "macd", "bollinger", "sma", "ema",
    "moving average", "moving averages", "stochastic oscillator",
    "atr", "average true range", "adx", "aroon", "momentum indicator", "cci",
]
FUNDAMENTAL_TERMS = [
    "fundamental analysis", "fundamental indicator", "fundamental indicators",
    "firm characteristic", "firm characteristics", "financial ratio",
    "financial ratios", "financial statement", "financial statements",
    "balance sheet", "income statement", "cash flow", "earnings per share",
    "eps", "return on equity", "roe", "return on assets", "roa",
    "price-to-earnings", "price to earnings", "p/e", "book value",
    "debt-to-equity", "debt to equity", "revenue growth", "profit margin",
    "earnings", "market capitalization", "market capitalisation",
]
NEWS_TERMS = [
    "financial news", "news headline", "news headlines", "news article",
    "news articles", "market news", "business news", "earnings call",
    "corporate disclosure", "corporate disclosures",
]
SOCIAL_TERMS = [
    "twitter", "tweet", "tweets", "reddit", "social media", "stocktwits",
    "weibo",
]
SENTIMENT_TERMS = [
    "sentiment", "sentiment score", "sentiment scores", "polarity",
    "opinion mining", "emotion analysis",
]
BERT_TERMS = ["finbert", "bert", "roberta"]
LLM_TERMS = [
    "large language model", "large language models", "llm", "llms",
    "fingpt", "chatgpt", "gpt-4", "gpt4",
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
    "transformer", "attention mechanism", "self-attention", "self attention",
    "informer", "itransformer", "temporal fusion transformer",
]
RECURRENT_TERMS = [
    "lstm", "gru", "long short-term memory", "long short term memory",
    "bilstm", "bi-lstm", "bigru", "bi-gru", "rnn", "recurrent neural",
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
    "whale optimization", "whale optimisation", "sparrow search",
    "firefly", "ant colony", "bee colony", "cuckoo search",
    "beluga whale", "moth flame", "bat algorithm",
]
XAI_TERMS = [
    "explainable", "explainability", "interpretable", "interpretability",
    "xai", "shap", "lime", "feature importance",
    "attention visualization", "attention visualisation",
]
TEMPORAL_VALIDATION_TERMS = [
    "walk-forward", "walk forward", "rolling window", "rolling-window",
    "expanding window", "expanding-window", "chronological split",
    "temporal split", "time-series cross-validation",
    "time series cross validation", "out-of-sample", "out of sample",
    "forward validation", "purged",
]
INDIA_TERMS = [
    "india", "indian stock", "indian stocks", "nifty", "nifty 50",
    "nse", "sensex", "bse", "bombay stock exchange",
    "national stock exchange",
]
MULTISTOCK_TERMS = [
    "multi-stock", "multiple stocks", "multiple companies", "several stocks",
    "cross-market", "cross market", "cross-country", "cross country",
    "global indices", "multiple indices", "cross-sectional",
]


# ============================================================
# 2. TEXT / FILE UTILITIES
# ============================================================

def clean_text(s):
    s = s or ""
    s = s.replace("\\&", "&").replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def normalize_doi(s):
    s = clean_text(s).lower()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s.rstrip(" .;,}")


def normalize_title(s):
    s = unicodedata.normalize("NFKD", clean_text(s))
    s = s.encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def make_uid(title, doi):
    doi = normalize_doi(doi)
    return "doi:" + doi if doi else "title:" + normalize_title(title)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def phrase_hits(text, phrases):
    """Token-aware phrase matching."""
    t = (text or "").lower()
    hits = set()
    for phrase in phrases:
        p = phrase.lower()
        pat = re.escape(p)
        left = r"(?<![a-z0-9])" if p and p[0].isalnum() else ""
        right = r"(?![a-z0-9])" if p and p[-1].isalnum() else ""
        if re.search(left + pat + right, t):
            hits.add(phrase)
    return sorted(hits)


def split_sentences(text):
    text = clean_text(text)
    if not text:
        return []
    return [
        x.strip()
        for x in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        if x.strip()
    ]


def sentence_level_stock_target_evidence(title, abstract):
    """
    Stronger than global keyword co-occurrence:
    finds sentences where stock/equity evidence and an eligible target/action
    co-occur in the same textual context.
    """
    eligible_target_terms = PRICE_TERMS + RETURN_TERMS + DIRECTION_TERMS + TREND_TERMS
    evidence = []

    for sentence in [title] + split_sentences(abstract):
        stock = phrase_hits(sentence, STOCK_TERMS)
        target = phrase_hits(sentence, eligible_target_terms)
        action = phrase_hits(sentence, FORECAST_ACTION_TERMS)
        if stock and (target or action):
            evidence.append(sentence)

    return evidence[:8]


# ============================================================
# 3. SIMPLE BIBTEX PARSER
# ============================================================

def bib_entry_chunks(text):
    starts = [m.start() for m in re.finditer(r"(?m)^@\w+\s*[\{\(]", text)]
    starts.append(len(text))
    return [text[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]


def parse_bib_entry(chunk):
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
            depth, j = 1, value_start
            while j < len(body) and depth:
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                j += 1
            value = body[value_start:j - 1] if depth == 0 else body[value_start:]
            pos = j
        else:
            j, escaped = value_start, False
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


def parse_bib(path):
    with open(path, encoding="utf-8-sig", errors="ignore") as f:
        text = f.read()
    out = []
    for chunk in bib_entry_chunks(text):
        e = parse_bib_entry(chunk)
        if e:
            out.append(e)
    return out


def source_query_id(filename):
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
# 4. RECONSTRUCT FROZEN 2024–2026 CORPUS
# ============================================================

def merge_sources(input_dir, expect_count):
    input_dir = Path(input_dir)
    records = {}
    manifest = []
    out_of_window = []

    # Scopus CSV
    for path in sorted(input_dir.glob("Scopus_Query-*_Records.csv")):
        qid = source_query_id(path.name)
        rows = read_csv(path)
        manifest.append({
            "Source_File": path.name, "Database": "Scopus",
            "Query_ID": qid, "Parsed_Records": len(rows),
            "SHA256": sha256_file(path),
        })

        for row in rows:
            title = row.get("Title", "")
            doi = normalize_doi(row.get("DOI", ""))
            try:
                year = int(row.get("Year", ""))
            except Exception:
                continue

            if year not in YEAR_WINDOW:
                out_of_window.append({
                    "Database": "Scopus", "Query_ID": qid,
                    "Title": title, "Year": year, "DOI": doi,
                    "Source_File": path.name,
                })
                continue

            uid = make_uid(title, doi)
            rec = records.get(uid, {
                "UID": uid, "Title": title, "Authors": row.get("Authors", ""),
                "Year": year, "Venue": row.get("Source title", ""),
                "DOI": doi, "Abstract": row.get("Abstract", "") or "",
                "Keywords": (
                    (row.get("Author Keywords", "") or "")
                    + ("; " + row.get("Index Keywords", "")
                       if row.get("Index Keywords", "") else "")
                ),
                "Document_Type": row.get("Document Type", "") or "",
                "Scopus": "No", "ScienceDirect": "No", "Elicit": "No",
                "Query_Hits": set(),
            })
            rec["Scopus"] = "Yes"
            rec["Query_Hits"].add(qid)
            records[uid] = rec

    # ScienceDirect
    for path in sorted(input_dir.glob("ScienceDirect_Query-*.bib")):
        qid = source_query_id(path.name)
        entries = parse_bib(path)
        manifest.append({
            "Source_File": path.name, "Database": "ScienceDirect",
            "Query_ID": qid, "Parsed_Records": len(entries),
            "SHA256": sha256_file(path),
        })

        for e in entries:
            title = clean_text(e.get("title", ""))
            doi = normalize_doi(e.get("doi", ""))
            try:
                year = int(e.get("year", ""))
            except Exception:
                continue

            if year not in YEAR_WINDOW:
                out_of_window.append({
                    "Database": "ScienceDirect", "Query_ID": qid,
                    "Title": title, "Year": year, "DOI": doi,
                    "Source_File": path.name,
                })
                continue

            uid = make_uid(title, doi)
            rec = records.get(uid, {
                "UID": uid, "Title": title,
                "Authors": clean_text(e.get("author", "")),
                "Year": year, "Venue": clean_text(e.get("journal", "")),
                "DOI": doi, "Abstract": clean_text(e.get("abstract", "")),
                "Keywords": clean_text(e.get("keywords", "")),
                "Document_Type": e.get("_type", "").title(),
                "Scopus": "No", "ScienceDirect": "No", "Elicit": "No",
                "Query_Hits": set(),
            })
            rec["ScienceDirect"] = "Yes"
            rec["Query_Hits"].add(qid)

            if not rec["Abstract"] and clean_text(e.get("abstract", "")):
                rec["Abstract"] = clean_text(e.get("abstract", ""))
            if not rec["Keywords"] and clean_text(e.get("keywords", "")):
                rec["Keywords"] = clean_text(e.get("keywords", ""))
            if not rec["Venue"]:
                rec["Venue"] = clean_text(e.get("journal", ""))

            records[uid] = rec

    # Elicit
    for path in sorted(input_dir.glob("EL0*.bib")):
        qid = source_query_id(path.name)
        entries = parse_bib(path)
        manifest.append({
            "Source_File": path.name, "Database": "Elicit",
            "Query_ID": qid, "Parsed_Records": len(entries),
            "SHA256": sha256_file(path),
        })

        for e in entries:
            title = clean_text(e.get("title", ""))
            doi = normalize_doi(e.get("doi", ""))
            try:
                year = int(e.get("year", ""))
            except Exception:
                continue

            if year not in YEAR_WINDOW:
                out_of_window.append({
                    "Database": "Elicit", "Query_ID": qid,
                    "Title": title, "Year": year, "DOI": doi,
                    "Source_File": path.name,
                })
                continue

            uid = make_uid(title, doi)
            rec = records.get(uid, {
                "UID": uid, "Title": title,
                "Authors": clean_text(e.get("author", "")),
                "Year": year, "Venue": clean_text(e.get("journal", "")),
                "DOI": doi, "Abstract": clean_text(e.get("abstract", "")),
                "Keywords": clean_text(e.get("keywords", "")),
                "Document_Type": e.get("_type", "").title(),
                "Scopus": "No", "ScienceDirect": "No", "Elicit": "No",
                "Query_Hits": set(),
            })
            rec["Elicit"] = "Yes"
            rec["Query_Hits"].add(qid)

            if not rec["Abstract"] and clean_text(e.get("abstract", "")):
                rec["Abstract"] = clean_text(e.get("abstract", ""))
            if not rec["Keywords"] and clean_text(e.get("keywords", "")):
                rec["Keywords"] = clean_text(e.get("keywords", ""))

            records[uid] = rec

    ordered = []
    for i, uid in enumerate(sorted(records), start=1):
        rec = records[uid].copy()
        rec["Master_ID"] = f"M-{i:04d}"
        rec["Query_Hits"] = "; ".join(sorted(rec["Query_Hits"]))
        ordered.append(rec)

    if expect_count is not None and len(ordered) != expect_count:
        raise RuntimeError(
            f"Corpus integrity check failed: expected {expect_count} "
            f"unique records but found {len(ordered)}."
        )

    return ordered, manifest, out_of_window


# ============================================================
# 5. ARTICLE-RELEVANCE PROFILING
# ============================================================

THEME_GROUPS = {
    "Historical_Data": HISTORICAL_TERMS,
    "Technical_Indicators": TECHNICAL_TERMS,
    "Fundamental_Data": FUNDAMENTAL_TERMS,
    "Financial_News": NEWS_TERMS,
    "Social_Media": SOCIAL_TERMS,
    "Sentiment": SENTIMENT_TERMS,
    "FinBERT_BERT_RoBERTa": BERT_TERMS,
    "LLM_FinGPT": LLM_TERMS,
    "Multimodal": MULTIMODAL_TERMS,
    "Fusion": FUSION_TERMS,
    "Transformer_Attention": TRANSFORMER_TERMS,
    "LSTM_GRU": RECURRENT_TERMS,
    "Optimization": OPTIMIZATION_TERMS,
    "Metaheuristic": METAHEURISTIC_TERMS,
    "XAI": XAI_TERMS,
    "Temporal_Validation": TEMPORAL_VALIDATION_TERMS,
    "Indian_Market": INDIA_TERMS,
    "Multi_Stock": MULTISTOCK_TERMS,
}


def yes_unclear(text, terms):
    hits = phrase_hits(text, terms)
    return ("Yes" if hits else "Unclear"), "; ".join(hits[:20])


def detect_targets(text):
    targets = []
    if phrase_hits(text, PRICE_TERMS):
        targets.append("Price")
    if phrase_hits(text, RETURN_TERMS):
        targets.append("Return")
    if phrase_hits(text, DIRECTION_TERMS):
        targets.append("Direction/Movement")
    if phrase_hits(text, TREND_TERMS):
        targets.append("Trend")
    return targets


def nonstock_domain_hits(text):
    labels = []
    evidence = []
    for label, terms in NON_STOCK_DOMAIN_TERMS.items():
        hits = phrase_hits(text, terms)
        if hits:
            labels.append(label)
            evidence.extend(hits)
    return labels, evidence


def document_class(rec, text):
    doc_type = (rec.get("Document_Type") or "").lower()
    title = (rec.get("Title") or "").lower()

    if phrase_hits(title + " " + text[:600], REVIEW_TERMS) or "review" in doc_type:
        return "Review/Survey"
    if "chapter" in doc_type or phrase_hits(title, BOOK_CHAPTER_TERMS):
        return "Book Chapter"
    if "conference" in doc_type or "proceedings" in doc_type:
        return "Conference"
    return "Primary/Other"


def profile_record(rec):
    title = rec.get("Title", "") or ""
    abstract = rec.get("Abstract", "") or ""
    keywords = rec.get("Keywords", "") or ""
    text = " ".join([title, abstract, keywords]).lower()

    stock_hits = phrase_hits(text, STOCK_TERMS)
    targets = detect_targets(text)
    target_hits = phrase_hits(
        text,
        PRICE_TERMS + RETURN_TERMS + DIRECTION_TERMS + TREND_TERMS
    )
    ai_hits = phrase_hits(text, AI_TERMS)
    empirical_hits = phrase_hits(text, EMPIRICAL_TERMS)
    action_hits = phrase_hits(text, FORECAST_ACTION_TERMS)
    strong_context = sentence_level_stock_target_evidence(title, abstract)

    volatility_hits = phrase_hits(text, VOLATILITY_RISK_TERMS)
    portfolio_hits = phrase_hits(text, PORTFOLIO_TERMS)
    trading_hits = phrase_hits(text, TRADING_ONLY_TERMS)
    sentiment_only_hits = phrase_hits(text, SENTIMENT_ONLY_TERMS)

    finance_hits = phrase_hits(text, OTHER_FINANCE_TERMS)
    nonstock_labels, nonstock_evidence = nonstock_domain_hits(text)

    doc_class = document_class(rec, text)

    # Evidence flags
    stock_explicit = bool(stock_hits)
    eligible_target_explicit = bool(targets)
    ai_explicit = bool(ai_hits)
    empirical_explicit = bool(empirical_hits)
    forecast_action_explicit = bool(action_hits)
    strong_stock_target_context = bool(strong_context)

    # If the same paper mentions a non-stock domain and explicit stock forecasting,
    # do not automatically demote it: mixed-market comparison papers can be useful.
    obvious_nonstock_only = bool(nonstock_labels) and not stock_explicit

    # Adjacent-topic flag: useful context but not central evidence.
    adjacent_only = (
        (volatility_hits or portfolio_hits or trading_hits or sentiment_only_hits)
        and not eligible_target_explicit
    )

    direct_core = (
        stock_explicit
        and eligible_target_explicit
        and ai_explicit
        and (empirical_explicit or strong_stock_target_context)
        and not obvious_nonstock_only
    )

    # Candidate if one core component is not explicit but stock/forecast/AI context exists.
    potential = (
        not direct_core
        and not obvious_nonstock_only
        and (
            (stock_explicit and ai_explicit and (forecast_action_explicit or eligible_target_explicit))
            or (finance_hits and ai_explicit and eligible_target_explicit)
        )
    )

    # Strategic theme detection
    theme_values = {}
    theme_evidence = {}
    for name, terms in THEME_GROUPS.items():
        value, evidence = yes_unclear(text, terms)
        theme_values[name] = value
        theme_evidence[name + "_Evidence"] = evidence

    strategic_theme_names = [
        "Technical_Indicators", "Fundamental_Data", "Financial_News",
        "Sentiment", "FinBERT_BERT_RoBERTa", "LLM_FinGPT",
        "Multimodal", "Fusion", "Transformer_Attention",
        "Optimization", "Metaheuristic", "XAI",
        "Temporal_Validation", "Indian_Market", "Multi_Stock",
    ]
    strategic_theme_count = sum(
        theme_values[x] == "Yes" for x in strategic_theme_names
    )

    # Relevance score = reading priority, not study quality.
    score = 0
    if stock_explicit:
        score += 4
    if eligible_target_explicit:
        score += 4
    if strong_stock_target_context:
        score += 2
    if ai_explicit:
        score += 3
    if empirical_explicit:
        score += 2

    weights = {
        "Technical_Indicators": 2,
        "Fundamental_Data": 3,
        "Financial_News": 2,
        "Sentiment": 2,
        "FinBERT_BERT_RoBERTa": 2,
        "LLM_FinGPT": 2,
        "Multimodal": 3,
        "Fusion": 2,
        "Transformer_Attention": 1,
        "LSTM_GRU": 1,
        "Optimization": 2,
        "Metaheuristic": 2,
        "XAI": 2,
        "Temporal_Validation": 2,
        "Indian_Market": 1,
        "Multi_Stock": 1,
    }
    for name, weight in weights.items():
        if theme_values[name] == "Yes":
            score += weight

    if doc_class == "Review/Survey":
        score -= 2
    if doc_class == "Book Chapter":
        score -= 1
    if obvious_nonstock_only:
        score -= 8
    if adjacent_only:
        score -= 3

    # Working research-use tiers.
    if doc_class == "Review/Survey":
        tier = "C_Context_Background"
        role = "Background synthesis / terminology / gap orientation"
    elif doc_class == "Book Chapter":
        tier = "C_Context_Background"
        role = "Contextual evidence only; not preferred as primary comparator"
    elif direct_core and strategic_theme_count >= 2:
        tier = "A_Direct_Methodological_Priority"
        role = "Detailed extraction for gap and methodology design"
    elif direct_core:
        tier = "B_Direct_Stock_Forecasting"
        role = "Related work / baseline / state-of-the-art comparison"
    elif potential:
        tier = "B2_Potentially_Relevant"
        role = "Check abstract/full text if strategically useful"
    elif stock_explicit or finance_hits:
        tier = "C_Context_Background"
        role = "Context only; not a priority for detailed extraction"
    else:
        tier = "D_Low_Priority"
        role = "No immediate use for the research article"

    # Elicit priority: intentionally narrow because Elicit usage is limited.
    # "High" is reserved for papers closest to the proposed framework.
    rare_framework_combo = (
        (
            theme_values["Technical_Indicators"] == "Yes"
            and theme_values["Fundamental_Data"] == "Yes"
        )
        or (
            theme_values["Fundamental_Data"] == "Yes"
            and (
                theme_values["Sentiment"] == "Yes"
                or theme_values["Financial_News"] == "Yes"
            )
        )
        or (
            theme_values["Multimodal"] == "Yes"
            and theme_values["Fusion"] == "Yes"
        )
    )

    data_theme_present = (
        theme_values["Technical_Indicators"] == "Yes"
        or theme_values["Fundamental_Data"] == "Yes"
        or theme_values["Financial_News"] == "Yes"
        or theme_values["Sentiment"] == "Yes"
    )

    direct_primary = direct_core and doc_class == "Primary/Other"

    if direct_primary and (score >= 24 or rare_framework_combo):
        elicit_priority = "High"
    elif direct_primary and (
        score >= 22
        or (
            data_theme_present
            and (
                theme_values["XAI"] == "Yes"
                or theme_values["Temporal_Validation"] == "Yes"
                or theme_values["Metaheuristic"] == "Yes"
            )
        )
    ):
        elicit_priority = "Medium"
    else:
        elicit_priority = "Low"

    # Research-gap dimensions explicitly evidenced in metadata.
    gap_dimensions = [
        k for k in strategic_theme_names
        if theme_values[k] == "Yes"
    ]

    out = dict(rec)
    out.update({
        "Document_Class": doc_class,
        "Stock_Domain": "Yes" if stock_explicit else "Unclear",
        "Eligible_Target": (
            "; ".join(targets) if targets else "Unclear"
        ),
        "AI_ML_DL": "Yes" if ai_explicit else "Unclear",
        "Empirical_Evaluation": "Yes" if empirical_explicit else "Unclear",
        "Strong_Stock_Target_Context": "Yes" if strong_stock_target_context else "No",
        "Adjacent_Target_Context": (
            "Yes" if (volatility_hits or portfolio_hits or trading_hits or sentiment_only_hits)
            else "No"
        ),
        "NonStock_Domain_Context": "; ".join(nonstock_labels),
        "Direct_Core_Evidence": "Yes" if direct_core else "No",
        "Strategic_Theme_Count": strategic_theme_count,
        "Relevance_Score": score,
        "Working_Relevance_Tier": tier,
        "Research_Use": role,
        "Elicit_Priority": elicit_priority,
        "Gap_Dimensions_Evidenced": "; ".join(gap_dimensions),
        "Stock_Evidence": "; ".join(stock_hits[:20]),
        "Target_Evidence": "; ".join(target_hits[:20]),
        "AI_Evidence": "; ".join(ai_hits[:20]),
        "Empirical_Evidence": "; ".join(empirical_hits[:20]),
        "Stock_Target_Sentence_Evidence": " || ".join(strong_context),
        "Adjacent_Evidence": "; ".join(
            (volatility_hits + portfolio_hits + trading_hits + sentiment_only_hits)[:20]
        ),
        "NonStock_Evidence": "; ".join(nonstock_evidence[:20]),
    })
    out.update(theme_values)
    out.update(theme_evidence)
    return out


# ============================================================
# 6. DEVELOPMENT-SET DIAGNOSTIC (OPTIONAL)
# ============================================================

def development_diagnostic(profiled_rows, development_set_path):
    if not development_set_path:
        return [], {}

    dev = read_csv(Path(development_set_path))
    by_id = {r["Master_ID"]: r for r in profiled_rows}
    out = []

    expected_group = {
        "Include": {"A_Direct_Methodological_Priority", "B_Direct_Stock_Forecasting"},
        "Maybe": {"B2_Potentially_Relevant", "A_Direct_Methodological_Priority",
                  "B_Direct_Stock_Forecasting"},
        "Exclude": {"C_Context_Background", "D_Low_Priority", "B2_Potentially_Relevant"},
    }

    for d in dev:
        p = by_id.get(d.get("Master_ID", ""))
        if not p:
            continue
        ref = d.get("Adjudicated_Decision", "")
        tier = p["Working_Relevance_Tier"]
        aligned = tier in expected_group.get(ref, set())
        out.append({
            "Master_ID": d.get("Master_ID", ""),
            "Reference_Decision": ref,
            "Reference_Code": d.get("Adjudicated_Exclusion_Code", ""),
            "V2_Working_Tier": tier,
            "V2_Relevance_Score": p["Relevance_Score"],
            "V2_Elicit_Priority": p["Elicit_Priority"],
            "Aligned_For_Development": "Yes" if aligned else "No",
            "Title": p["Title"],
            "Eligible_Target": p["Eligible_Target"],
            "Gap_Dimensions_Evidenced": p["Gap_Dimensions_Evidenced"],
        })

    summary = {
        "development_records_found": len(out),
        "development_alignment_count": sum(
            r["Aligned_For_Development"] == "Yes" for r in out
        ),
        "development_alignment_fraction": (
            sum(r["Aligned_For_Development"] == "Yes" for r in out) / len(out)
            if out else None
        ),
        "note": (
            "This diagnostic is for rule development only. "
            "It is not a screening-validity statistic and should not be reported "
            "as independent validation in the research article."
        ),
    }
    return out, summary


# ============================================================
# 7. OUTPUTS
# ============================================================

def create_outputs(master, profiled, manifest, out_of_window, output_dir, dev_path=None):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    write_csv(outdir / "01_source_manifest.csv", manifest)
    write_csv(outdir / "02_out_of_window.csv", out_of_window)
    write_csv(outdir / "03_master_profiled_2808.csv", profiled)

    tier_order = {
        "A_Direct_Methodological_Priority": 0,
        "B_Direct_Stock_Forecasting": 1,
        "B2_Potentially_Relevant": 2,
        "C_Context_Background": 3,
        "D_Low_Priority": 4,
    }

    ranked = sorted(
        profiled,
        key=lambda r: (
            tier_order.get(r["Working_Relevance_Tier"], 99),
            -int(r["Relevance_Score"]),
            -int(r["Year"]),
            r["Title"].lower(),
        ),
    )
    write_csv(outdir / "04_ranked_literature_for_research_article.csv", ranked)

    # Direct methodological papers for structured extraction.
    direct = [
        r for r in ranked
        if r["Working_Relevance_Tier"] in {
            "A_Direct_Methodological_Priority",
            "B_Direct_Stock_Forecasting",
        }
    ]
    write_csv(outdir / "05_direct_stock_forecasting_evidence.csv", direct)

    # Highest-value Elicit usage: High only, to conserve remaining credits.
    elicit_high = [
        r for r in ranked
        if r["Elicit_Priority"] == "High"
    ]
    write_csv(outdir / "06_elicit_high_priority.csv", elicit_high)

    # Broader article-development shortlist for local/manual reading.
    article_development = [
        r for r in ranked
        if r["Elicit_Priority"] in {"High", "Medium"}
    ]
    write_csv(
        outdir / "06b_article_development_shortlist.csv",
        article_development,
    )

    # Research-gap matrix: abstract-level evidence only.
    gap_fields = [
        "Master_ID", "Title", "Year", "Venue", "DOI",
        "Working_Relevance_Tier", "Relevance_Score", "Elicit_Priority",
        "Eligible_Target", "Historical_Data", "Technical_Indicators",
        "Fundamental_Data", "Financial_News", "Social_Media", "Sentiment",
        "FinBERT_BERT_RoBERTa", "LLM_FinGPT", "Multimodal", "Fusion",
        "Transformer_Attention", "LSTM_GRU", "Optimization",
        "Metaheuristic", "XAI", "Temporal_Validation", "Indian_Market",
        "Multi_Stock", "Gap_Dimensions_Evidenced",
    ]
    write_csv(
        outdir / "07_abstract_level_gap_matrix.csv",
        article_development,
        gap_fields,
    )

    # Theme-specific shortlists.
    theme_outputs = {
        "08_technical_fundamental.csv": lambda r:
            r["Technical_Indicators"] == "Yes" or r["Fundamental_Data"] == "Yes",
        "09_news_sentiment_nlp.csv": lambda r:
            r["Financial_News"] == "Yes" or r["Sentiment"] == "Yes"
            or r["FinBERT_BERT_RoBERTa"] == "Yes" or r["LLM_FinGPT"] == "Yes",
        "10_multimodal_fusion.csv": lambda r:
            r["Multimodal"] == "Yes" or r["Fusion"] == "Yes",
        "11_optimization_metaheuristics.csv": lambda r:
            r["Optimization"] == "Yes" or r["Metaheuristic"] == "Yes",
        "12_explainability_xai.csv": lambda r:
            r["XAI"] == "Yes",
        "13_temporal_validation.csv": lambda r:
            r["Temporal_Validation"] == "Yes",
        "14_transformer_attention.csv": lambda r:
            r["Transformer_Attention"] == "Yes",
        "15_indian_market.csv": lambda r:
            r["Indian_Market"] == "Yes",
    }

    for filename, predicate in theme_outputs.items():
        subset = [
            r for r in ranked
            if r["Working_Relevance_Tier"] in {
                "A_Direct_Methodological_Priority",
                "B_Direct_Stock_Forecasting",
                "B2_Potentially_Relevant",
            }
            and predicate(r)
        ]
        write_csv(outdir / filename, subset)

    # Counts for planning, not PRISMA.
    tier_counts = Counter(r["Working_Relevance_Tier"] for r in profiled)
    elicit_counts = Counter(r["Elicit_Priority"] for r in profiled)

    strategic_counts = {}
    for name in THEME_GROUPS:
        strategic_counts[name] = sum(
            r[name] == "Yes"
            for r in direct
        )

    summary = {
        "purpose": (
            "Research-article literature relevance mining, not systematic-review screening."
        ),
        "corpus_size": len(profiled),
        "tier_counts": dict(tier_counts),
        "direct_evidence_count": len(direct),
        "elicit_high_priority_count": len(elicit_high),
        "article_development_shortlist_count": len(article_development),
        "elicit_priority_distribution_all_records": dict(elicit_counts),
        "direct_evidence_theme_counts": strategic_counts,
        "year_counts": dict(Counter(str(r["Year"]) for r in profiled)),
        "source_membership_counts": {
            "Scopus": sum(r["Scopus"] == "Yes" for r in profiled),
            "ScienceDirect": sum(r["ScienceDirect"] == "Yes" for r in profiled),
            "Elicit": sum(r["Elicit"] == "Yes" for r in profiled),
        },
        "interpretation": {
            "A_Direct_Methodological_Priority": (
                "Direct stock forecasting + AI/ML/DL + empirical evidence, "
                "with two or more strategic themes relevant to the proposed model."
            ),
            "B_Direct_Stock_Forecasting": (
                "Direct empirical AI stock forecasting useful for related work, "
                "baselines or state-of-the-art comparison."
            ),
            "B2_Potentially_Relevant": (
                "Close to scope but one core element is not explicit in metadata; "
                "check only if the topic is strategically useful."
            ),
            "C_Context_Background": (
                "Reviews, book chapters, volatility/risk/trading/sentiment context, "
                "or general finance evidence."
            ),
            "D_Low_Priority": (
                "No immediate methodological use for the proposed research article."
            ),
        },
        "warning": (
            "Relevance_Score is a reading-priority score, not a study-quality score. "
            "Absence of a tag means 'not evidenced in title/abstract/keywords', not "
            "that the feature is absent from the full paper."
        ),
    }

    # Optional development-set diagnostic.
    diagnostic_rows, diagnostic_summary = development_diagnostic(
        profiled, dev_path
    )
    if diagnostic_rows:
        write_csv(
            outdir / "16_development_set_diagnostic.csv",
            diagnostic_rows,
        )
        summary["development_set_diagnostic"] = diagnostic_summary

    with open(outdir / "00_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Human-readable text summary.
    with open(outdir / "00_summary.txt", "w", encoding="utf-8") as f:
        f.write("RESEARCH-ARTICLE LITERATURE MINING V2\n")
        f.write("=" * 60 + "\n")
        f.write(f"Corpus: {len(profiled)}\n\n")
        f.write("Working relevance tiers:\n")
        for k, v in sorted(tier_counts.items()):
            f.write(f"  {k}: {v}\n")
        f.write(f"\nDirect stock-forecasting evidence: {len(direct)}\n")
        f.write(f"Elicit High priority: {len(elicit_high)}\n")
        f.write(f"Article-development shortlist (High+Medium): {len(article_development)}\n")
        if diagnostic_summary:
            f.write("\nDevelopment-set diagnostic:\n")
            for k, v in diagnostic_summary.items():
                f.write(f"  {k}: {v}\n")

    return summary


# ============================================================
# 8. MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--expect-count", type=int, default=2808)
    ap.add_argument(
        "--development-set",
        default=None,
        help=(
            "Optional adjudicated 58-record development file. "
            "Used only for diagnostic comparison, not for independent validation."
        ),
    )
    args = ap.parse_args()

    print("\n=== V2 STAGE 1: Reconstruct frozen corpus ===")
    master, manifest, out_of_window = merge_sources(
        args.input_dir, args.expect_count
    )
    print(f"Unique records: {len(master)}")
    print(f"Out-of-window retrieval records logged: {len(out_of_window)}")

    print("\n=== V2 STAGE 2: Research-relevance profiling ===")
    profiled = [profile_record(r) for r in master]

    print("\n=== V2 STAGE 3: Create research-article shortlists ===")
    summary = create_outputs(
        master, profiled, manifest, out_of_window,
        args.output_dir, args.development_set
    )

    print(json.dumps(summary, indent=2))
    print("\nOutputs written to:", args.output_dir)
    print(
        "\nThis pipeline ranks literature for a research article. "
        "It does not generate PRISMA or systematic-review inclusion counts."
    )


if __name__ == "__main__":
    main()
