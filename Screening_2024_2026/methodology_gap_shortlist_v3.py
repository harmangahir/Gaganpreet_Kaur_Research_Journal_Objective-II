#!/usr/bin/env python3
"""
methodology_gap_shortlist_v3.py

Purpose
-------
Build a compact, methodology-defining evidence set for a RESEARCH ARTICLE
from the V2 literature-mining outputs.

This is NOT systematic-review screening. It is a transparent ranking and
evidence-mapping utility for:
    - related-work selection,
    - closest-competitor identification,
    - research-gap analysis,
    - methodology design,
    - baseline selection,
    - targeted Elicit/full-text extraction.

Default input:
    literature_mining_v2/06b_article_development_shortlist.csv

Default target:
    60 papers

Example:
    python methodology_gap_shortlist_v3.py \
        --input literature_mining_v2/06b_article_development_shortlist.csv \
        --output-dir methodology_gap_v3 \
        --target 60

The resulting 60-paper set is a reading/extraction priority set, NOT a formal
"included study" corpus.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


YES = "Yes"


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------

def read_csv(path: Path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames=None):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def is_yes(r, col):
    return (r.get(col) or "").strip() == YES


# ---------------------------------------------------------------------
# PROFILING FOR ARTICLE DEVELOPMENT
# ---------------------------------------------------------------------

def profile(r):
    """
    Add research-article-specific evidence dimensions.
    These are abstract/keyword-level signals only.
    """

    historical = is_yes(r, "Historical_Data")
    technical = is_yes(r, "Technical_Indicators")
    fundamental = is_yes(r, "Fundamental_Data")

    financial_news = is_yes(r, "Financial_News")
    social = is_yes(r, "Social_Media")
    sentiment = is_yes(r, "Sentiment")
    textual = financial_news or social or sentiment

    multimodal = is_yes(r, "Multimodal")
    fusion = is_yes(r, "Fusion")
    transformer = is_yes(r, "Transformer_Attention")
    recurrent = is_yes(r, "LSTM_GRU")
    finbert = is_yes(r, "FinBERT_BERT_RoBERTa")
    llm = is_yes(r, "LLM_FinGPT")

    optimization = is_yes(r, "Optimization")
    metaheuristic = is_yes(r, "Metaheuristic")
    xai = is_yes(r, "XAI")
    temporal = is_yes(r, "Temporal_Validation")
    india = is_yes(r, "Indian_Market")
    multi_stock = is_yes(r, "Multi_Stock")

    # Four data families corresponding closely to the planned research article.
    family_flags = {
        "Historical/Market": historical,
        "Technical": technical,
        "Fundamental": fundamental,
        "Textual/Sentiment": textual,
    }
    data_family_count = sum(family_flags.values())

    # "Closest competitor" means the metadata explicitly evidences the same
    # multi-source direction as the planned framework. It does not mean the
    # paper is better or worse than the proposed work.
    closest_competitor = (
        (technical and fundamental and textual)
        or
        (fundamental and textual and (multimodal or fusion))
        or
        (technical and fundamental and (multimodal or fusion))
        or
        (data_family_count >= 3 and fusion)
    )

    # Particularly important gap-design combinations.
    fusion_rich = (multimodal or fusion) and data_family_count >= 2
    robust_fusion = fusion_rich and (temporal or xai or optimization or metaheuristic)
    modern_text = textual and (finbert or llm)
    optimized_forecast = optimization or metaheuristic
    explainable_forecast = xai
    temporally_aware = temporal

    # Reading-priority score only; not study quality.
    score = 0

    tier = r.get("Working_Relevance_Tier", "")
    if tier == "A_Direct_Methodological_Priority":
        score += 5
    elif tier == "B_Direct_Stock_Forecasting":
        score += 3
    elif tier == "B2_Potentially_Relevant":
        score += 1

    # Data integration closest to intended framework.
    score += 1 if historical else 0
    score += 2 if technical else 0
    score += 4 if fundamental else 0
    score += 2 if textual else 0
    score += 3 if multimodal else 0
    score += 3 if fusion else 0

    # Modern modelling / NLP.
    score += 2 if transformer else 0
    score += 1 if recurrent else 0
    score += 2 if finbert else 0
    score += 2 if llm else 0

    # Methodological rigor / novelty dimensions.
    score += 2 if optimization else 0
    score += 2 if metaheuristic else 0
    score += 3 if xai else 0
    score += 4 if temporal else 0

    # Direct contextual value.
    score += 1 if india else 0
    score += 1 if multi_stock else 0

    # Reward explicit integration.
    if closest_competitor:
        score += 5
    if robust_fusion:
        score += 3

    # Evidence roles. A paper can serve multiple roles.
    roles = []
    if closest_competitor:
        roles.append("Closest competing multimodal framework")
    if fundamental:
        roles.append("Fundamental-data integration")
    if textual:
        roles.append("News/sentiment integration")
    if modern_text:
        roles.append("Advanced financial NLP/LLM")
    if multimodal or fusion:
        roles.append("Multimodal/fusion architecture")
    if optimized_forecast:
        roles.append("Optimization/metaheuristic design")
    if xai:
        roles.append("Explainability/XAI")
    if temporal:
        roles.append("Temporal validation/leakage control")
    if transformer:
        roles.append("Transformer/attention architecture")
    if india:
        roles.append("Indian-market evidence")
    if multi_stock:
        roles.append("Multi-stock/multi-market validation")

    # Primary role for organization.
    if closest_competitor:
        primary_role = "Closest_Competitor"
    elif fusion_rich:
        primary_role = "Fusion_Architecture"
    elif fundamental:
        primary_role = "Fundamental_Integration"
    elif modern_text:
        primary_role = "Advanced_NLP"
    elif textual:
        primary_role = "Sentiment_News"
    elif temporal:
        primary_role = "Temporal_Validation"
    elif xai:
        primary_role = "XAI"
    elif optimized_forecast:
        primary_role = "Optimization"
    elif transformer:
        primary_role = "Advanced_Architecture"
    elif india:
        primary_role = "Indian_Market"
    else:
        primary_role = "Supporting_Baseline"

    out = dict(r)
    out.update({
        "Data_Family_Count": data_family_count,
        "Historical_Family": "Yes" if historical else "No",
        "Technical_Family": "Yes" if technical else "No",
        "Fundamental_Family": "Yes" if fundamental else "No",
        "Textual_Sentiment_Family": "Yes" if textual else "No",
        "Closest_Competitor": "Yes" if closest_competitor else "No",
        "Fusion_Rich": "Yes" if fusion_rich else "No",
        "Robust_Fusion": "Yes" if robust_fusion else "No",
        "Modern_Text_NLP": "Yes" if modern_text else "No",
        "Article_Methodology_Score": score,
        "Primary_Research_Role": primary_role,
        "Research_Roles": "; ".join(roles),
    })
    return out


# ---------------------------------------------------------------------
# GAP COUNTS
# ---------------------------------------------------------------------

def combo_counts(rows):
    def count(predicate):
        return sum(1 for r in rows if predicate(r))

    def y(r, c):
        return is_yes(r, c)

    return {
        "Technical + Fundamental": count(
            lambda r: y(r, "Technical_Indicators") and y(r, "Fundamental_Data")
        ),
        "Technical + Textual/Sentiment": count(
            lambda r: y(r, "Technical_Indicators")
            and (
                y(r, "Financial_News")
                or y(r, "Social_Media")
                or y(r, "Sentiment")
            )
        ),
        "Fundamental + Textual/Sentiment": count(
            lambda r: y(r, "Fundamental_Data")
            and (
                y(r, "Financial_News")
                or y(r, "Social_Media")
                or y(r, "Sentiment")
            )
        ),
        "Technical + Fundamental + Textual/Sentiment": count(
            lambda r: y(r, "Technical_Indicators")
            and y(r, "Fundamental_Data")
            and (
                y(r, "Financial_News")
                or y(r, "Social_Media")
                or y(r, "Sentiment")
            )
        ),
        "Historical + Technical + Fundamental + Textual/Sentiment": count(
            lambda r: y(r, "Historical_Data")
            and y(r, "Technical_Indicators")
            and y(r, "Fundamental_Data")
            and (
                y(r, "Financial_News")
                or y(r, "Social_Media")
                or y(r, "Sentiment")
            )
        ),
        "Multimodal + Fusion": count(
            lambda r: y(r, "Multimodal") and y(r, "Fusion")
        ),
        "Fusion + Optimization/Metaheuristic": count(
            lambda r: y(r, "Fusion")
            and (y(r, "Optimization") or y(r, "Metaheuristic"))
        ),
        "Fusion + XAI": count(
            lambda r: y(r, "Fusion") and y(r, "XAI")
        ),
        "Fusion + Temporal Validation": count(
            lambda r: y(r, "Fusion") and y(r, "Temporal_Validation")
        ),
        "Fundamental + XAI": count(
            lambda r: y(r, "Fundamental_Data") and y(r, "XAI")
        ),
        "Fundamental + Temporal Validation": count(
            lambda r: y(r, "Fundamental_Data") and y(r, "Temporal_Validation")
        ),
        "Textual/Sentiment + FinBERT/BERT/RoBERTa": count(
            lambda r: (
                y(r, "Financial_News") or y(r, "Sentiment")
            ) and y(r, "FinBERT_BERT_RoBERTa")
        ),
        "Textual/Sentiment + LLM/FinGPT": count(
            lambda r: (
                y(r, "Financial_News") or y(r, "Sentiment")
            ) and y(r, "LLM_FinGPT")
        ),
        "Indian Market + Multimodal/Fusion": count(
            lambda r: y(r, "Indian_Market")
            and (y(r, "Multimodal") or y(r, "Fusion"))
        ),
        "Multi-stock + Temporal Validation": count(
            lambda r: y(r, "Multi_Stock") and y(r, "Temporal_Validation")
        ),
    }


# ---------------------------------------------------------------------
# DIVERSITY-BALANCED SHORTLIST
# ---------------------------------------------------------------------

def select_shortlist(rows, target):
    """
    Deterministic, diversity-balanced selection.

    We first take strongest closest competitors, then ensure representation
    of the methodological dimensions needed to design the new model.
    Remaining places are filled by overall methodology score.
    """

    ranked = sorted(
        rows,
        key=lambda r: (
            -int(r["Article_Methodology_Score"]),
            -int(r.get("Year") or 0),
            (r.get("Title") or "").lower(),
        )
    )

    selected = []
    selected_ids = set()
    selection_reason = {}

    def add_candidates(predicate, quota, reason):
        added = 0
        for r in ranked:
            if len(selected) >= target or added >= quota:
                break
            mid = r["Master_ID"]
            if mid in selected_ids:
                continue
            if predicate(r):
                selected.append(r)
                selected_ids.add(mid)
                selection_reason[mid] = reason
                added += 1

    # The quotas create coverage; they are reading-management parameters,
    # not scientific inclusion quotas.
    add_candidates(
        lambda r: r["Closest_Competitor"] == "Yes",
        min(15, target),
        "Closest competing framework"
    )
    add_candidates(
        lambda r: is_yes(r, "Fundamental_Data"),
        min(10, target),
        "Fundamental-data evidence"
    )
    add_candidates(
        lambda r: r["Fusion_Rich"] == "Yes",
        min(10, target),
        "Multimodal/fusion evidence"
    )
    add_candidates(
        lambda r: r["Modern_Text_NLP"] == "Yes",
        min(8, target),
        "Advanced financial NLP/LLM"
    )
    add_candidates(
        lambda r: is_yes(r, "Optimization") or is_yes(r, "Metaheuristic"),
        min(8, target),
        "Optimization/metaheuristic evidence"
    )
    add_candidates(
        lambda r: is_yes(r, "XAI"),
        min(8, target),
        "Explainability evidence"
    )
    add_candidates(
        lambda r: is_yes(r, "Temporal_Validation"),
        min(8, target),
        "Temporal-validation evidence"
    )
    add_candidates(
        lambda r: is_yes(r, "Indian_Market"),
        min(6, target),
        "Indian-market evidence"
    )
    add_candidates(
        lambda r: is_yes(r, "Multi_Stock"),
        min(6, target),
        "Multi-stock/multi-market evidence"
    )

    # Fill remaining positions by overall article-methodology score.
    for r in ranked:
        if len(selected) >= target:
            break
        mid = r["Master_ID"]
        if mid not in selected_ids:
            selected.append(r)
            selected_ids.add(mid)
            selection_reason[mid] = "Overall methodology relevance"

    # Add selection reason and final extraction rank.
    output = []
    for rank, r in enumerate(selected, start=1):
        rr = dict(r)
        rr["Extraction_Rank"] = rank
        rr["Selection_Reason"] = selection_reason[r["Master_ID"]]
        output.append(rr)

    return output


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        required=True,
        help="V2 article-development shortlist CSV (normally 06b_...)."
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--target",
        type=int,
        default=60,
        help="Target size for detailed methodology extraction. Default: 60."
    )
    args = ap.parse_args()

    rows = read_csv(Path(args.input))
    profiled = [profile(r) for r in rows]

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Full 260-paper profile.
    ranked = sorted(
        profiled,
        key=lambda r: (
            -int(r["Article_Methodology_Score"]),
            -int(r.get("Year") or 0),
            (r.get("Title") or "").lower(),
        )
    )
    write_csv(outdir / "01_profiled_article_development_pool.csv", ranked)

    # Key evidence groups.
    groups = {
        "02_closest_competitors.csv":
            lambda r: r["Closest_Competitor"] == "Yes",

        "03_fundamental_integration.csv":
            lambda r: is_yes(r, "Fundamental_Data"),

        "04_news_sentiment_nlp.csv":
            lambda r: r["Textual_Sentiment_Family"] == "Yes",

        "05_multimodal_fusion.csv":
            lambda r: is_yes(r, "Multimodal") or is_yes(r, "Fusion"),

        "06_optimization_metaheuristics.csv":
            lambda r: is_yes(r, "Optimization") or is_yes(r, "Metaheuristic"),

        "07_xai.csv":
            lambda r: is_yes(r, "XAI"),

        "08_temporal_validation.csv":
            lambda r: is_yes(r, "Temporal_Validation"),

        "09_indian_market.csv":
            lambda r: is_yes(r, "Indian_Market"),

        "10_multi_stock.csv":
            lambda r: is_yes(r, "Multi_Stock"),
    }

    group_counts = {}
    for filename, predicate in groups.items():
        subset = [r for r in ranked if predicate(r)]
        write_csv(outdir / filename, subset)
        group_counts[filename] = len(subset)

    # Compact extraction set.
    target = min(max(args.target, 1), len(ranked))
    shortlist = select_shortlist(ranked, target)
    write_csv(outdir / "00_methodology_extraction_shortlist.csv", shortlist)

    # Elicit-specific set: same shortlist, but explicit file name.
    write_csv(outdir / "00_elicit_extraction_set.csv", shortlist)

    # Gap-combination map over the 260-paper article-development pool.
    combos = combo_counts(profiled)
    write_csv(
        outdir / "11_gap_combination_counts.csv",
        [{"Combination": k, "Metadata_Evidence_Count": v} for k, v in combos.items()],
        ["Combination", "Metadata_Evidence_Count"]
    )

    # Primary-role distribution in the compact shortlist.
    role_counts = Counter(r["Primary_Research_Role"] for r in shortlist)
    reason_counts = Counter(r["Selection_Reason"] for r in shortlist)

    summary = {
        "purpose": "Methodology/gap evidence prioritization for a research article.",
        "input_article_development_pool": len(rows),
        "methodology_extraction_target": target,
        "methodology_extraction_shortlist": len(shortlist),
        "group_counts_within_article_development_pool": group_counts,
        "gap_combination_counts": combos,
        "shortlist_primary_role_counts": dict(role_counts),
        "shortlist_selection_reason_counts": dict(reason_counts),
        "notes": [
            "The shortlist is a reading/extraction priority set, not a systematic-review included-study set.",
            "Counts are based only on explicit evidence in title/abstract/keywords.",
            "A missing tag means not evidenced in metadata, not necessarily absent from the full text.",
            "Use full text/Elicit to verify the actual research gap before claiming novelty."
        ]
    }

    with open(outdir / "00_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nDetailed extraction set written to: {outdir / '00_methodology_extraction_shortlist.csv'}")


if __name__ == "__main__":
    main()
