#!/usr/bin/env python3
"""
analyze_qc_disagreements.py

Diagnose why the first deterministic screening rules disagreed with the
human reviewer. This script does NOT change any screening decisions.
It only summarizes the QC errors so that Version 2 rules can be revised
transparently and rerun over the full corpus.

Usage:

python analyze_qc_disagreements.py \
    --screened screening_results/stage3_tagged.csv \
    --manual screening_results/stage4_manual_review_completed.csv \
    --output-dir screening_results/qc_diagnosis
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


LABELS = {"Include", "Exclude", "Maybe", "Background"}


def read_csv(path: Path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def split_terms(value: str):
    return [x.strip() for x in (value or "").split(";") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screened", required=True)
    ap.add_argument("--manual", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    screened = {r["Master_ID"]: r for r in read_csv(Path(args.screened))}
    manual = read_csv(Path(args.manual))
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    qc = [
        r for r in manual
        if (r.get("Queue_Reason") or "").startswith("QC sample")
        and (r.get("Human_Decision") or "").strip() in LABELS
    ]

    disagreements = [
        r for r in qc
        if r["AI_Decision"] != r["Human_Decision"]
    ]

    include_to_exclude = [
        r for r in disagreements
        if r["AI_Decision"] == "Include"
        and r["Human_Decision"] == "Exclude"
    ]

    exclude_to_include = [
        r for r in disagreements
        if r["AI_Decision"] == "Exclude"
        and r["Human_Decision"] == "Include"
    ]

    def enrich(rows):
        enriched = []
        for m in rows:
            s = screened.get(m["Master_ID"], {})
            enriched.append({
                "Master_ID": m["Master_ID"],
                "AI_Decision": m["AI_Decision"],
                "Human_Decision": m["Human_Decision"],
                "AI_Exclusion_Code": m.get("Exclusion_Code", ""),
                "Human_Exclusion_Code": m.get("Human_Exclusion_Code", ""),
                "Title": m.get("Title", ""),
                "Year": m.get("Year", ""),
                "Venue": m.get("Venue", ""),
                "DOI": m.get("DOI", ""),
                "AI_Reason": m.get("AI_Reason", ""),
                "Reviewer_Notes": m.get("Reviewer_Notes", ""),
                "Primary_Theme": s.get("Primary_Theme", ""),
                "Evidence_Level": s.get("Evidence_Level", ""),
                "Prediction_Target": s.get("Prediction_Target", ""),
                "Stock_Evidence": s.get("Stock_Evidence", ""),
                "Prediction_Evidence": s.get("Prediction_Evidence", ""),
                "AI_Evidence": s.get("AI_Evidence", ""),
                "NonTarget_Evidence": s.get("NonTarget_Evidence", ""),
                "Technical_Indicators": s.get("Technical_Indicators", ""),
                "Fundamental_Data": s.get("Fundamental_Data", ""),
                "Financial_News": s.get("Financial_News", ""),
                "Sentiment": s.get("Sentiment", ""),
                "Multimodal": s.get("Multimodal", ""),
                "Fusion": s.get("Fusion", ""),
                "Optimization": s.get("Optimization", ""),
                "XAI": s.get("XAI", ""),
                "Temporal_Validation": s.get("Temporal_Validation", ""),
            })
        return enriched

    all_dis = enrich(disagreements)
    inc_exc = enrich(include_to_exclude)
    exc_inc = enrich(exclude_to_include)

    write_csv(outdir / "all_qc_disagreements.csv", all_dis)
    write_csv(outdir / "false_includes_AI_include_human_exclude.csv", inc_exc)
    write_csv(outdir / "false_excludes_AI_exclude_human_include.csv", exc_inc)

    # Human exclusion codes for false automatic Includes
    false_include_human_codes = Counter(
        r.get("Human_Exclusion_Code", "") or "BLANK"
        for r in include_to_exclude
    )

    # Which AI exclusion codes caused false automatic Excludes?
    false_exclude_ai_codes = Counter(
        r.get("Exclusion_Code", "") or "BLANK"
        for r in exclude_to_include
    )

    # Themes and evidence levels in disagreements
    theme_counter = Counter()
    evidence_counter = Counter()
    stock_terms = Counter()
    prediction_terms = Counter()
    ai_terms = Counter()
    nontarget_terms = Counter()

    for r in all_dis:
        theme_counter[r.get("Primary_Theme", "") or "BLANK"] += 1
        evidence_counter[r.get("Evidence_Level", "") or "BLANK"] += 1

        for x in split_terms(r.get("Stock_Evidence", "")):
            stock_terms[x] += 1
        for x in split_terms(r.get("Prediction_Evidence", "")):
            prediction_terms[x] += 1
        for x in split_terms(r.get("AI_Evidence", "")):
            ai_terms[x] += 1
        for x in split_terms(r.get("NonTarget_Evidence", "")):
            nontarget_terms[x] += 1

    # Full QC confusion table
    confusion = defaultdict(Counter)
    for r in qc:
        confusion[r["AI_Decision"]][r["Human_Decision"]] += 1

    summary = {
        "qc_records_completed": len(qc),
        "qc_disagreements": len(disagreements),
        "AI_Include_to_Human_Exclude": len(include_to_exclude),
        "AI_Exclude_to_Human_Include": len(exclude_to_include),
        "other_disagreement_types": len(disagreements) - len(include_to_exclude) - len(exclude_to_include),
        "false_include_human_exclusion_codes": dict(false_include_human_codes),
        "false_exclude_ai_exclusion_codes": dict(false_exclude_ai_codes),
        "disagreement_primary_themes": dict(theme_counter),
        "disagreement_evidence_levels": dict(evidence_counter),
        "top_stock_evidence_terms_in_disagreements": dict(stock_terms.most_common(20)),
        "top_prediction_evidence_terms_in_disagreements": dict(prediction_terms.most_common(20)),
        "top_ai_evidence_terms_in_disagreements": dict(ai_terms.most_common(20)),
        "top_non_target_terms_in_disagreements": dict(nontarget_terms.most_common(20)),
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "interpretation": (
            "Use this diagnosis to revise deterministic rules. "
            "Do not change individual unreviewed records manually. "
            "If a systematic pattern is identified, modify the rule and rerun "
            "the entire 2,808-record corpus as Screening Version 2."
        ),
    }

    with open(outdir / "qc_disagreement_analysis.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
