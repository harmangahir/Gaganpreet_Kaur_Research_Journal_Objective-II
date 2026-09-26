from __future__ import annotations
import argparse
import re
from itertools import combinations
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
import networkx as nx

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10

def clean_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).strip())

def has_substantive_content(x):
    s = clean_text(x).lower()
    if not s:
        return False
    blocked = [
        "not reported", "not applicable", "not explicitly reported",
        "not clearly reported", "none reported", "not stated",
        "no explicit", "unclear"
    ]
    return not any(b in s for b in blocked)

def contains_any(text, keywords):
    t = clean_text(text).lower()
    return any(k in t for k in keywords)

def modality_profile(row):
    tech = has_substantive_content(row.get("Technical Indicators", ""))
    fund = has_substantive_content(row.get("Fundamental Variables", ""))
    news = has_substantive_content(row.get("News/Sentiment Data", ""))

    parts = []
    if tech:
        parts.append("Technical")
    if fund:
        parts.append("Fundamental")
    if news:
        parts.append("News/Sentiment")
    return " + ".join(parts) if parts else "Historical / Other"

def model_family(text):
    t = clean_text(text).lower()
    if ("graph" in t or "gnn" in t) and any(k in t for k in ["transformer", "attention", "bert"]):
        return "Graph + attention"
    if any(k in t for k in ["transformer", "attention", "bert", "fingpt"]):
        return "Transformer/attention"
    if any(k in t for k in ["lstm", "gru", "rnn", "bilstm"]):
        return "RNN/LSTM/GRU"
    if any(k in t for k in ["graph", "gnn"]):
        return "Graph-based"
    if any(k in t for k in ["random forest", "xgboost", "svm", "logistic regression"]):
        return "Classical ML"
    return "Hybrid/other"

def bool_theme_columns(df):
    out = pd.DataFrame(index=df.index)
    out["Technical indicators"] = df["Technical Indicators"].apply(has_substantive_content)
    out["Fundamental variables"] = df["Fundamental Variables"].apply(has_substantive_content)
    out["News / sentiment"] = df["News/Sentiment Data"].apply(has_substantive_content)
    out["Transformer / attention"] = df["Model Architecture"].apply(
        lambda x: contains_any(x, ["transformer", "attention", "bert", "fingpt"])
    )
    out["Adaptive / explicit fusion"] = df["Fusion Strategy"].apply(
        lambda x: contains_any(
            x, ["fusion", "cross-attention", "co-attention", "gated", "adaptive",
                "weighted", "concaten", "late fusion", "decision-level", "feature-level"]
        )
    )
    out["Optimization / HPO"] = df["Optimization / Hyperparameter Method"].apply(has_substantive_content)
    out["XAI / interpretability"] = df["Explainability / XAI"].apply(has_substantive_content)
    out["Temporal validation"] = df["Temporal Validation / Data Split"].apply(has_substantive_content)
    out["Walk-forward / rolling"] = df["Walk-Forward / Rolling Evaluation"].apply(
        lambda x: has_substantive_content(x) and not contains_any(x, ["single temporal split"])
    )
    out["Leakage control"] = df["Temporal Alignment / Leakage Control"].apply(has_substantive_content)
    out["Ablation study"] = df["Ablation / Component Analysis"].apply(has_substantive_content)
    return out

def rigor_profile(row):
    score = sum([
        bool(row["Temporal validation"]),
        bool(row["Walk-forward / rolling"]),
        bool(row["Leakage control"]),
        bool(row["Optimization / HPO"]),
        bool(row["XAI / interpretability"]),
        bool(row["Ablation study"]),
    ])
    if score >= 4:
        return "High rigor"
    if score >= 2:
        return "Moderate rigor"
    return "Basic rigor"


def draw_lollipop(ax, labels, values, title, panel_letter):
    total = int(sum(values))
    y = list(range(len(labels)))
    ax.hlines(y=y, xmin=0, xmax=values, linewidth=1.2)
    ax.scatter(values, y, s=24, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(f"{panel_letter}. {title}", fontweight="bold", loc="left")
    ax.set_xlabel("Number of papers")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    xmax = max(values) if len(values) else 1
    ax.set_xlim(0, xmax * 1.28)
    for yi, val in zip(y, values):
        pct = 100.0 * val / total if total else 0.0
        ax.text(val + xmax * 0.02, yi, f"{int(val)} ({pct:.1f}%)", va="center", ha="left", fontsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="Figure_RW_Panels.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    themes = bool_theme_columns(df)
    n = len(df)

    modality_series = pd.Series({
        "Technical": int(themes["Technical indicators"].sum()),
        "Fundamental": int(themes["Fundamental variables"].sum()),
        "News/sentiment": int(themes["News / sentiment"].sum()),
        "Technical + Fundamental": int(((themes["Technical indicators"]) & (themes["Fundamental variables"])).sum()),
        "Technical + News": int(((themes["Technical indicators"]) & (themes["News / sentiment"])).sum()),
        "Technical + Fundamental + News": int(((themes["Technical indicators"]) & (themes["Fundamental variables"]) & (themes["News / sentiment"])).sum()),
    }).sort_values(ascending=False)

    model_series = df["Model Architecture"].apply(model_family).value_counts()
    model_order = ["RNN/LSTM/GRU", "Transformer/attention", "Graph-based", "Graph + attention", "Classical ML", "Hybrid/other"]
    model_series = model_series.reindex([m for m in model_order if m in model_series.index])

    rigor_series = pd.Series({
        "Temporal split": int(themes["Temporal validation"].sum()),
        "Walk-forward/rolling": int(themes["Walk-forward / rolling"].sum()),
        "Leakage control": int(themes["Leakage control"].sum()),
        "Optimization/HPO": int(themes["Optimization / HPO"].sum()),
        "XAI/interpretable": int(themes["XAI / interpretability"].sum()),
    }).sort_values(ascending=False)

    transparency_series = pd.Series({
        "Explicit fusion": int(themes["Adaptive / explicit fusion"].sum()),
        "Ablation study": int(themes["Ablation study"].sum()),
        "XAI/interpretable": int(themes["XAI / interpretability"].sum()),
        "Walk-forward/rolling": int(themes["Walk-forward / rolling"].sum()),
        "Leakage control": int(themes["Leakage control"].sum()),
    }).sort_values(ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    draw_lollipop(axes[0], list(modality_series.index), list(modality_series.values), "Information streams", "A")
    draw_lollipop(axes[1], list(model_series.index), list(model_series.values), "Model families", "B")
    draw_lollipop(axes[2], list(rigor_series.index), list(rigor_series.values), "Evaluation rigor", "C")
    draw_lollipop(axes[3], list(transparency_series.index), list(transparency_series.values), "Analytical transparency", "D")

    fig.suptitle(
        f"Summary of methodological patterns in the extracted related-work corpus",
        fontsize=13, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    out = Path(args.output)
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.resolve()}")

if __name__ == "__main__":
    main()
