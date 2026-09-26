#!/usr/bin/env python3
"""
related_work_visuals.py

Create manuscript-ready visualizations from the Elicit extraction CSV
for the related-work section of the stock-forecasting research article.

Recommended figures:
1. Sankey: Modality profile -> Model family -> Methodological rigor
2. Bubble plot: Year vs theme coverage
3. Bubble-bar / lollipop: Theme counts across papers
4. Optional co-occurrence network: theme co-occurrence map

Usage
-----
python related_work_visuals.py \
    --input "Elicit - extract-results-review-5e20178e-b81d-49a9-a404-15f3804482ec (2).csv" \
    --output-dir related_work_figures

Notes
-----
- This script is intended for an ORIGINAL RESEARCH ARTICLE, not a review paper.
- Use 1–2 figures in the main manuscript; keep the rest for supplementary material.
"""

from __future__ import annotations
import argparse
import re
from itertools import combinations
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx


# ----------------------------
# utility functions
# ----------------------------
def clean_text(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def has_substantive_content(x):
    s = clean_text(x).lower()
    if s == "":
        return False
    blocked = [
        "not reported",
        "not applicable",
        "not explicitly reported",
        "not clearly reported",
        "unclear",
        "none reported",
        "not stated",
        "no explicit",
    ]
    return not any(b in s for b in blocked)


def contains_any(text, keywords):
    t = clean_text(text).lower()
    return any(k in t for k in keywords)


# ----------------------------
# categorization logic
# ----------------------------
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

    if parts:
        return " + ".join(parts)
    return "Historical / Other"


def model_family(text):
    t = clean_text(text).lower()

    if ("graph" in t or "gnn" in t) and (
        "transformer" in t or "attention" in t or "bert" in t
    ):
        return "Graph + Attention"

    if any(k in t for k in ["transformer", "attention", "bert", "fingpt"]):
        return "Transformer / Attention"

    if any(k in t for k in ["lstm", "gru", "rnn", "bilstm"]):
        return "RNN / LSTM / GRU"

    if any(k in t for k in ["graph", "gnn"]):
        return "Graph-based"

    if any(k in t for k in ["random forest", "xgboost", "svm", "logistic regression"]):
        return "Classical ML"

    return "Hybrid / Other"


def fusion_type(text):
    t = clean_text(text).lower()
    if not has_substantive_content(text):
        return "Not reported / NA"

    if any(k in t for k in ["late", "decision-level", "decision level"]):
        return "Late fusion"

    if any(k in t for k in ["cross-attention", "co-attention", "gated", "adaptive", "weighted"]):
        return "Adaptive / Attention fusion"

    if any(k in t for k in ["feature-level", "feature level", "concaten", "early fusion", "intermediate"]):
        return "Feature / Early fusion"

    if "graph" in t:
        return "Graph fusion"

    return "Other fusion"


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
            x,
            [
                "fusion",
                "cross-attention",
                "co-attention",
                "gated",
                "adaptive",
                "weighted",
                "concaten",
                "late fusion",
                "decision-level",
                "feature-level",
            ],
        )
    )
    out["Optimization / HPO"] = df["Optimization / Hyperparameter Method"].apply(
        has_substantive_content
    )
    out["XAI / interpretability"] = df["Explainability / XAI"].apply(has_substantive_content)
    out["Temporal validation"] = df["Temporal Validation / Data Split"].apply(
        has_substantive_content
    )
    out["Walk-forward / rolling"] = df["Walk-Forward / Rolling Evaluation"].apply(
        lambda x: has_substantive_content(x) and not contains_any(x, ["single temporal split"])
    )
    out["Leakage control"] = df["Temporal Alignment / Leakage Control"].apply(
        has_substantive_content
    )
    out["Ablation study"] = df["Ablation / Component Analysis"].apply(has_substantive_content)

    return out


def rigor_profile(row):
    score = sum(
        [
            bool(row["Temporal validation"]),
            bool(row["Walk-forward / rolling"]),
            bool(row["Leakage control"]),
            bool(row["Optimization / HPO"]),
            bool(row["XAI / interpretability"]),
            bool(row["Ablation study"]),
        ]
    )
    if score >= 4:
        return "High rigor"
    if score >= 2:
        return "Moderate rigor"
    return "Basic rigor"


# ----------------------------
# figures
# ----------------------------
def save_plotly(fig, stem: Path):
    html_path = stem.with_suffix(".html")
    png_path = stem.with_suffix(".png")
    fig.write_html(str(html_path))
    try:
        fig.write_image(str(png_path), scale=2)
    except Exception as e:
        print(f"PNG export skipped for {stem.name}: {e}")
    print(f"Saved: {html_path}")
    if png_path.exists():
        print(f"Saved: {png_path}")


def make_sankey(df, outdir: Path):
    theme_df = bool_theme_columns(df)

    tmp = pd.DataFrame({
        "Modality profile": df.apply(modality_profile, axis=1),
        "Model family": df["Model Architecture"].apply(model_family),
        "Rigor profile": theme_df.apply(rigor_profile, axis=1),
    })

    left_mid = tmp.groupby(["Modality profile", "Model family"]).size().reset_index(name="count")
    mid_right = tmp.groupby(["Model family", "Rigor profile"]).size().reset_index(name="count")

    nodes = list(pd.unique(
        pd.concat([
            left_mid["Modality profile"],
            left_mid["Model family"],
            mid_right["Rigor profile"]
        ], ignore_index=True)
    ))
    node_index = {n: i for i, n in enumerate(nodes)}

    source = []
    target = []
    value = []

    for _, r in left_mid.iterrows():
        source.append(node_index[r["Modality profile"]])
        target.append(node_index[r["Model family"]])
        value.append(int(r["count"]))

    for _, r in mid_right.iterrows():
        source.append(node_index[r["Model family"]])
        target.append(node_index[r["Rigor profile"]])
        value.append(int(r["count"]))

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=18,
                    thickness=18,
                    line=dict(width=0.5),
                    label=nodes,
                ),
                link=dict(source=source, target=target, value=value),
            )
        ]
    )
    fig.update_layout(
        title="Related-work landscape: modality profile \u2192 model family \u2192 methodological rigor",
        font_size=11,
    )
    save_plotly(fig, outdir / "figure1_sankey_modality_model_rigor")


def make_year_theme_bubble(df, outdir: Path):
    theme_df = bool_theme_columns(df)
    work = df.copy()

    work["Year"] = pd.to_numeric(work["Year"], errors="coerce")
    work = work.dropna(subset=["Year"]).copy()
    work["Year"] = work["Year"].astype(int)

    long_rows = []
    for idx, row in work.iterrows():
        for theme in theme_df.columns:
            if bool(theme_df.loc[idx, theme]):
                long_rows.append({"Year": row["Year"], "Theme": theme})

    long_df = pd.DataFrame(long_rows)
    counts = long_df.groupby(["Year", "Theme"]).size().reset_index(name="Count")

    fig = px.scatter(
        counts,
        x="Year",
        y="Theme",
        size="Count",
        hover_data=["Count"],
        title="Theme coverage across years in the extracted related-work corpus",
    )
    fig.update_layout(yaxis_title="", xaxis_title="Year")
    save_plotly(fig, outdir / "figure2_bubble_year_theme")


def make_theme_bubble_bar(df, outdir: Path):
    theme_df = bool_theme_columns(df)
    counts = theme_df.sum().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    y = range(len(counts))
    ax.hlines(y=y, xmin=0, xmax=counts.values)
    ax.scatter(counts.values, y, s=counts.values * 35)

    ax.set_yticks(list(y))
    ax.set_yticklabels(counts.index)
    ax.set_xlabel("Number of papers")
    ax.set_title("Theme prevalence in the extracted related-work corpus")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = outdir / "figure3_bubble_bar_theme_counts.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def make_theme_network(df, outdir: Path):
    theme_df = bool_theme_columns(df)
    themes = list(theme_df.columns)
    co = pd.DataFrame(0, index=themes, columns=themes, dtype=int)

    for _, row in theme_df.iterrows():
        active = [t for t in themes if bool(row[t])]
        for a, b in combinations(active, 2):
            co.loc[a, b] += 1
            co.loc[b, a] += 1

    G = nx.Graph()
    for t in themes:
        G.add_node(t, count=int(theme_df[t].sum()))

    for i, a in enumerate(themes):
        for b in themes[i+1:]:
            w = int(co.loc[a, b])
            if w > 0:
                G.add_edge(a, b, weight=w)

    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(10, 8))

    edge_widths = [G[u][v]["weight"] * 0.2 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.35, ax=ax)

    node_sizes = [G.nodes[n]["count"] * 220 for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

    ax.set_title("Optional theme co-occurrence network for related work")
    ax.axis("off")
    plt.tight_layout()
    out = outdir / "figure4_theme_cooccurrence_network.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def make_summary_table(df, outdir: Path):
    theme_df = bool_theme_columns(df)
    tmp = pd.DataFrame({
        "Modality profile": df.apply(modality_profile, axis=1),
        "Model family": df["Model Architecture"].apply(model_family),
        "Fusion type": df["Fusion Strategy"].apply(fusion_type),
        "Year": pd.to_numeric(df["Year"], errors="coerce"),
    })
    summary = {
        "n_papers": len(df),
        "modality_profile_counts": tmp["Modality profile"].value_counts().to_dict(),
        "model_family_counts": tmp["Model family"].value_counts().to_dict(),
        "fusion_type_counts": tmp["Fusion type"].value_counts().to_dict(),
        "theme_counts": theme_df.sum().sort_values(ascending=False).to_dict(),
    }
    pd.DataFrame({
        "Theme": list(theme_df.sum().sort_values(ascending=False).index),
        "Count": list(theme_df.sum().sort_values(ascending=False).values)
    }).to_csv(outdir / "theme_counts.csv", index=False)
    pd.DataFrame({
        "Category": tmp["Modality profile"].value_counts().index,
        "Count": tmp["Modality profile"].value_counts().values
    }).to_csv(outdir / "modality_profile_counts.csv", index=False)
    pd.DataFrame({
        "Category": tmp["Model family"].value_counts().index,
        "Count": tmp["Model family"].value_counts().values
    }).to_csv(outdir / "model_family_counts.csv", index=False)
    print("Summary:")
    for k, v in summary.items():
        print(k, ":", v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to Elicit export CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for figures")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} papers from {args.input}")

    make_sankey(df, outdir)
    make_year_theme_bubble(df, outdir)
    make_theme_bubble_bar(df, outdir)
    make_theme_network(df, outdir)
    make_summary_table(df, outdir)

    print(f"\nAll outputs written to: {outdir}")


if __name__ == "__main__":
    main()
