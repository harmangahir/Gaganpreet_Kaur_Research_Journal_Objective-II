#!/usr/bin/env python3
"""
figure_related_work_sankey_q1.py

Create a publication-style Sankey diagram closely matching the requested
journal figure aesthetic:

Input modality profile -> Principal model family -> Reported methodological safeguards

Ribbon color is determined by the principal model family and is retained
through both stages, as in the reference style.

Usage
-----
python figure_related_work_sankey_q1.py \
    --input "03_Elicit - 34 articles extracted results.csv" \
    --output "Figure_RW_Sankey_Q1.pdf"
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle

# ---------------------------------------------------------------------
# Figure typography
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.2,
    "axes.titlesize": 10.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
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

    if tech and fund and news:
        return "Technical + Fundamental + News/Sentiment"
    if tech and fund:
        return "Technical + Fundamental"
    if tech and news:
        return "Technical + News/Sentiment"
    if fund and news:
        return "Fundamental + News/Sentiment"
    if tech:
        return "Technical"
    if fund:
        return "Fundamental"
    if news:
        return "News/Sentiment"
    return "Historical / Other"

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

def safeguards_frame(df):
    s = pd.DataFrame(index=df.index)
    s["Temporal validation"] = df["Temporal Validation / Data Split"].apply(has_substantive_content)
    s["Walk-forward / rolling"] = df["Walk-Forward / Rolling Evaluation"].apply(
        lambda x: has_substantive_content(x) and not contains_any(x, ["single temporal split"])
    )
    s["Leakage control"] = df["Temporal Alignment / Leakage Control"].apply(has_substantive_content)
    s["Optimization / HPO"] = df["Optimization / Hyperparameter Method"].apply(has_substantive_content)
    s["XAI / interpretability"] = df["Explainability / XAI"].apply(has_substantive_content)
    s["Ablation study"] = df["Ablation / Component Analysis"].apply(has_substantive_content)
    return s

def safeguards_group(row):
    score = int(row.sum())
    if score >= 4:
        return "4-6 safeguards"
    if score >= 2:
        return "2-3 safeguards"
    return "0-1 safeguards"

# ---------------------------------------------------------------------
# Sankey geometry
# ---------------------------------------------------------------------
def make_column_layout(labels, counts, top=0.88, bottom=0.075, gap=0.013):
    """
    Returns {label: (bottom, top, height)} and unit scale.
    All columns share the same unit height so ribbons conserve width.
    """
    total = sum(counts[l] for l in labels)
    available = top - bottom - gap * max(0, len(labels)-1)
    scale = available / total
    pos = {}
    y = top
    for lab in labels:
        h = counts[lab] * scale
        pos[lab] = (y-h, y, h)
        y -= h + gap
    return pos, scale

def ribbon(ax, x0, x1, y0b, y0t, y1b, y1t, color, alpha=0.42):
    """
    Smooth Sankey ribbon using cubic Bezier curves.
    """
    curvature = 0.42 * (x1 - x0)
    verts = [
        (x0, y0b),
        (x0 + curvature, y0b),
        (x1 - curvature, y1b),
        (x1, y1b),

        (x1, y1t),

        (x1 - curvature, y1t),
        (x0 + curvature, y0t),
        (x0, y0t),

        (x0, y0b),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    patch = PathPatch(
        MplPath(verts, codes),
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        zorder=1,
    )
    ax.add_patch(patch)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="Figure_RW_Sankey_Q1.pdf")
    ap.add_argument("--png", default=None, help="Optional PNG preview")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    sf = safeguards_frame(df)

    work = pd.DataFrame({
        "Input": df.apply(modality_profile, axis=1),
        "Model": df["Model Architecture"].apply(model_family),
        "Safeguards": sf.apply(safeguards_group, axis=1),
    })

    # -----------------------------------------------------------------
    # Explicit ordering for a stable, journal-like layout
    # -----------------------------------------------------------------
    left_preferred = [
        "Technical + News/Sentiment",
        "News/Sentiment",
        "Technical + Fundamental",
        "Technical",
        "Historical / Other",
        "Fundamental + News/Sentiment",
        "Technical + Fundamental + News/Sentiment",
        "Fundamental",
    ]
    left_counts_all = work["Input"].value_counts().to_dict()
    left_labels = [x for x in left_preferred if left_counts_all.get(x, 0) > 0]
    left_counts = {x: left_counts_all[x] for x in left_labels}

    mid_preferred = [
        "Transformer/attention",
        "Hybrid/other",
        "RNN/LSTM/GRU",
        "Graph + attention",
        "Graph-based",
        "Classical ML",
    ]
    mid_counts_all = work["Model"].value_counts().to_dict()
    mid_labels = [x for x in mid_preferred if mid_counts_all.get(x, 0) > 0]
    mid_counts = {x: mid_counts_all[x] for x in mid_labels}

    right_labels = ["4-6 safeguards", "2-3 safeguards", "0-1 safeguards"]
    right_counts_all = work["Safeguards"].value_counts().to_dict()
    right_counts = {x: right_counts_all.get(x, 0) for x in right_labels}

    # -----------------------------------------------------------------
    # Palette: muted/pastel, modeled on the supplied reference style
    # Ribbons are colored by principal model family.
    # -----------------------------------------------------------------
    family_colors = {
        "Transformer/attention": "#4FC3E0",  # cyan
        "Hybrid/other":          "#9BCF4B",  # lime green
        "RNN/LSTM/GRU":          "#F2A15A",  # orange
        "Graph + attention":     "#D65AA6",  # magenta
        "Graph-based":           "#E76F51",  # coral red
        "Classical ML":          "#E9B949",  # ochre
    }

    left_node_colors = {
        "Technical + News/Sentiment": "#8C66E8",
        "News/Sentiment": "#E37C4C",
        "Technical + Fundamental": "#59A7A5",
        "Technical": "#668FB9",
        "Historical / Other": "#A6A6A6",
        "Fundamental + News/Sentiment": "#D483B8",
        "Technical + Fundamental + News/Sentiment": "#7DAB55",
        "Fundamental": "#C5A454",
    }

    right_node_color = "#A8A8A8"

    left_pos, scale = make_column_layout(left_labels, left_counts)
    mid_pos, _ = make_column_layout(mid_labels, mid_counts)
    right_pos, _ = make_column_layout(right_labels, right_counts)

    lm = work.groupby(["Input", "Model"]).size().to_dict()
    mr = work.groupby(["Model", "Safeguards"]).size().to_dict()

    # Flow offsets within nodes
    l_off = {k: left_pos[k][0] for k in left_labels}
    m_in_off = {k: mid_pos[k][0] for k in mid_labels}
    m_out_off = {k: mid_pos[k][0] for k in mid_labels}
    r_off = {k: right_pos[k][0] for k in right_labels}

    # -----------------------------------------------------------------
    # Canvas
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14.2, 8.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    xL, xM, xR = 0.075, 0.485, 0.905
    node_w = 0.0105

    # Left -> Middle
    # Ordering each left node by middle category keeps ribbons organized.
    for l in left_labels:
        for m in mid_labels:
            n = lm.get((l, m), 0)
            if n <= 0:
                continue
            h = n * scale
            y0b, y0t = l_off[l], l_off[l] + h
            y1b, y1t = m_in_off[m], m_in_off[m] + h
            ribbon(
                ax,
                xL + node_w, xM,
                y0b, y0t, y1b, y1t,
                family_colors.get(m, "#B0B0B0"),
                alpha=0.43,
            )
            l_off[l] += h
            m_in_off[m] += h

    # Middle -> Right
    for m in mid_labels:
        for r in right_labels:
            n = mr.get((m, r), 0)
            if n <= 0:
                continue
            h = n * scale
            y0b, y0t = m_out_off[m], m_out_off[m] + h
            y1b, y1t = r_off[r], r_off[r] + h
            ribbon(
                ax,
                xM + node_w, xR,
                y0b, y0t, y1b, y1t,
                family_colors.get(m, "#B0B0B0"),
                alpha=0.43,
            )
            m_out_off[m] += h
            r_off[r] += h

    # -----------------------------------------------------------------
    # Nodes
    # -----------------------------------------------------------------
    # Left: colored nodes, labels to the right of each node (reference style)
    for lab in left_labels:
        b, t, h = left_pos[lab]
        ax.add_patch(Rectangle(
            (xL, b), node_w, h,
            facecolor=left_node_colors.get(lab, "#999999"),
            edgecolor="#555555", linewidth=0.45, zorder=3
        ))
        ax.text(
            xL + node_w + 0.008, (b+t)/2,
            f"{lab}\n(n={left_counts[lab]})",
            ha="left", va="center", fontsize=8.4, color="#333333", zorder=4
        )

    # Middle: colored nodes, labels to the right of each node
    for lab in mid_labels:
        b, t, h = mid_pos[lab]
        ax.add_patch(Rectangle(
            (xM, b), node_w, h,
            facecolor=family_colors.get(lab, "#BBBBBB"),
            edgecolor="#555555", linewidth=0.45, zorder=3
        ))
        ax.text(
            xM + node_w + 0.008, (b+t)/2,
            f"{lab}\n(n={mid_counts[lab]})",
            ha="left", va="center", fontsize=8.4, color="#333333", zorder=4
        )

    # Right: neutral grey nodes, labels to the left of nodes
    for lab in right_labels:
        b, t, h = right_pos[lab]
        ax.add_patch(Rectangle(
            (xR, b), node_w, h,
            facecolor=right_node_color,
            edgecolor="#555555", linewidth=0.45, zorder=3
        ))
        ax.text(
            xR - 0.008, (b+t)/2,
            f"{lab}\n(n={right_counts[lab]})",
            ha="right", va="center", fontsize=8.4, color="#333333", zorder=4
        )

    # -----------------------------------------------------------------
    # Column headings
    # -----------------------------------------------------------------
    ax.text(
        0.03, 0.953,
        "Input modality profile",
        ha="left", va="center", fontsize=10.7, fontweight="bold"
    )
    ax.text(
        0.42, 0.953,
        "Principal model family",
        ha="left", va="center", fontsize=10.7, fontweight="bold"
    )
    ax.text(
        0.80, 0.953,
        "Reported methodological safeguards",
        ha="left", va="center", fontsize=10.7, fontweight="bold"
    )

    # Small note instead of a large chart title
    ax.text(
        0.99, 0.025,
        "n = 34",
        ha="right", va="bottom", fontsize=8.0, style="italic", color="#444444"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.tight_layout(pad=0.8)

    out = Path(args.output)
    plt.savefig(out, format="pdf", bbox_inches="tight", facecolor="white")

    if args.png:
        plt.savefig(args.png, dpi=300, bbox_inches="tight", facecolor="white")

    plt.close(fig)
    print(f"Saved: {out.resolve()}")
    if args.png:
        print(f"Saved: {Path(args.png).resolve()}")

if __name__ == "__main__":
    main()
