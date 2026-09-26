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


def node_layout(labels, counts, y_top=0.94, y_bottom=0.06, gap=0.018):
    total = sum(counts.values())
    usable = y_top - y_bottom - gap * max(0, len(labels)-1)
    scale = usable / total if total else 0
    positions = {}
    y = y_top
    for label in labels:
        h = counts[label] * scale
        positions[label] = [y-h, y, h]
        y = y-h-gap
    return positions, scale

def flow_patch(ax, x0, x1, y0b, y0t, y1b, y1t, alpha=0.28):
    c = (x1 - x0) * 0.45
    verts = [
        (x0, y0b),
        (x0+c, y0b), (x1-c, y1b), (x1, y1b),
        (x1, y1t),
        (x1-c, y1t), (x0+c, y0t), (x0, y0t),
        (x0, y0b),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), alpha=alpha, linewidth=0))

def draw_nodes(ax, x, width, labels, pos, counts, align):
    for lab in labels:
        b, t, h = pos[lab]
        ax.add_patch(Rectangle((x, b), width, h, linewidth=0.8, facecolor="white", edgecolor="black"))
        if align == "left":
            ax.text(x - 0.012, (b+t)/2, f"{lab}\n(n={counts[lab]})", ha="right", va="center", fontsize=9)
        elif align == "middle":
            ax.text(x + width/2, (b+t)/2, f"{lab}\n(n={counts[lab]})", ha="center", va="center", fontsize=8.5)
        else:
            ax.text(x + width + 0.012, (b+t)/2, f"{lab}\n(n={counts[lab]})", ha="left", va="center", fontsize=9)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="Figure_RW_Sankey.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    td = bool_theme_columns(df)
    tmp = pd.DataFrame({
        "Modality": df.apply(modality_profile, axis=1),
        "Model": df["Model Architecture"].apply(model_family),
        "Rigor": td.apply(rigor_profile, axis=1)
    })

    left_labels = list(tmp["Modality"].value_counts().index)
    mid_labels = list(tmp["Model"].value_counts().index)
    right_labels = ["High rigor", "Moderate rigor", "Basic rigor"]

    left_counts = tmp["Modality"].value_counts().to_dict()
    mid_counts = tmp["Model"].value_counts().to_dict()
    right_counts = tmp["Rigor"].value_counts().reindex(right_labels, fill_value=0).to_dict()

    left_pos, scale = node_layout(left_labels, left_counts)
    mid_pos, _ = node_layout(mid_labels, mid_counts)
    right_pos, _ = node_layout(right_labels, right_counts)

    lm = tmp.groupby(["Modality", "Model"]).size().to_dict()
    mr = tmp.groupby(["Model", "Rigor"]).size().to_dict()

    left_offsets = {k: left_pos[k][0] for k in left_labels}
    mid_in_offsets = {k: mid_pos[k][0] for k in mid_labels}
    mid_out_offsets = {k: mid_pos[k][0] for k in mid_labels}
    right_offsets = {k: right_pos[k][0] for k in right_labels}

    fig, ax = plt.subplots(figsize=(14, 8))
    xL, xM, xR = 0.08, 0.48, 0.88
    w = 0.026

    for l in left_labels:
        for m in mid_labels:
            n = lm.get((l, m), 0)
            if n <= 0:
                continue
            h = n * scale
            y0b, y0t = left_offsets[l], left_offsets[l] + h
            y1b, y1t = mid_in_offsets[m], mid_in_offsets[m] + h
            flow_patch(ax, xL + w, xM, y0b, y0t, y1b, y1t)
            left_offsets[l] += h
            mid_in_offsets[m] += h

    for m in mid_labels:
        for r in right_labels:
            n = mr.get((m, r), 0)
            if n <= 0:
                continue
            h = n * scale
            y0b, y0t = mid_out_offsets[m], mid_out_offsets[m] + h
            y1b, y1t = right_offsets[r], right_offsets[r] + h
            flow_patch(ax, xM + w, xR, y0b, y0t, y1b, y1t)
            mid_out_offsets[m] += h
            right_offsets[r] += h

    draw_nodes(ax, xL, w, left_labels, left_pos, left_counts, "left")
    draw_nodes(ax, xM, w, mid_labels, mid_pos, mid_counts, "middle")
    draw_nodes(ax, xR, w, right_labels, right_pos, right_counts, "right")

    ax.text(xL + w/2, 0.99, "Input modality profile", ha="center", va="top", fontweight="bold")
    ax.text(xM + w/2, 0.99, "Model family", ha="center", va="top", fontweight="bold")
    ax.text(xR + w/2, 0.99, "Methodological rigor", ha="center", va="top", fontweight="bold")
    ax.set_title("Flow of the extracted related-work corpus across modality profile, model family, and rigor level", fontweight="bold", pad=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.tight_layout()
    out = Path(args.output)
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.resolve()}")

if __name__ == "__main__":
    main()
