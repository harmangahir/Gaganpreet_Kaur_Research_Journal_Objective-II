#!/usr/bin/env python3
"""
select_elicit_core_v4.py

Purpose
-------
Reduce the 60-paper methodology/gap shortlist to a compact Elicit/full-text
core for a RESEARCH ARTICLE.

This is NOT systematic-review screening.

The selector prioritizes:
1. closest competing multimodal frameworks,
2. papers combining technical + fundamental + textual/sentiment data,
3. fusion + temporal validation,
4. fusion + optimization/metaheuristics,
5. fundamental + XAI,
6. advanced financial NLP/LLM,
7. Indian-market multimodal/fusion evidence,
8. multi-stock + temporal validation.

Example:
    python select_elicit_core_v4.py \
        --input methodology_gap_v3/00_methodology_extraction_shortlist.csv \
        --output-dir elicit_core_v4 \
        --target 35
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    rows=list(rows)
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields=list(rows[0].keys()) if rows else []
    with open(path,"w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def y(r, c):
    return (r.get(c) or "").strip() == "Yes"


def textual(r):
    return y(r,"Financial_News") or y(r,"Social_Media") or y(r,"Sentiment")


def score_record(r):
    # Start from V3 methodology score.
    try:
        score=int(float(r.get("Article_Methodology_Score") or 0))
    except Exception:
        score=0

    # Strongly reward combinations closest to planned framework.
    if y(r,"Technical_Indicators") and y(r,"Fundamental_Data") and textual(r):
        score += 12

    if (
        y(r,"Historical_Data")
        and y(r,"Technical_Indicators")
        and y(r,"Fundamental_Data")
        and textual(r)
    ):
        score += 10

    if y(r,"Multimodal") and y(r,"Fusion"):
        score += 6

    if y(r,"Fusion") and y(r,"Temporal_Validation"):
        score += 7

    if y(r,"Fusion") and (y(r,"Optimization") or y(r,"Metaheuristic")):
        score += 6

    if y(r,"Fundamental_Data") and y(r,"XAI"):
        score += 5

    if y(r,"Fundamental_Data") and y(r,"Temporal_Validation"):
        score += 6

    if textual(r) and y(r,"FinBERT_BERT_RoBERTa"):
        score += 4

    if textual(r) and y(r,"LLM_FinGPT"):
        score += 5

    if y(r,"Indian_Market") and (y(r,"Multimodal") or y(r,"Fusion")):
        score += 4

    if y(r,"Multi_Stock") and y(r,"Temporal_Validation"):
        score += 5

    return score


def category_flags(r):
    return {
        "Closest_Competitor": (r.get("Closest_Competitor") == "Yes"),
        "Three_Data_Families": (
            y(r,"Technical_Indicators")
            and y(r,"Fundamental_Data")
            and textual(r)
        ),
        "Four_Data_Families": (
            y(r,"Historical_Data")
            and y(r,"Technical_Indicators")
            and y(r,"Fundamental_Data")
            and textual(r)
        ),
        "Fusion_Temporal": y(r,"Fusion") and y(r,"Temporal_Validation"),
        "Fusion_Optimization": y(r,"Fusion") and (
            y(r,"Optimization") or y(r,"Metaheuristic")
        ),
        "Fundamental_XAI": y(r,"Fundamental_Data") and y(r,"XAI"),
        "Fundamental_Temporal": y(r,"Fundamental_Data") and y(r,"Temporal_Validation"),
        "Advanced_NLP": textual(r) and (
            y(r,"FinBERT_BERT_RoBERTa") or y(r,"LLM_FinGPT")
        ),
        "India_Multimodal_Fusion": y(r,"Indian_Market") and (
            y(r,"Multimodal") or y(r,"Fusion")
        ),
        "MultiStock_Temporal": y(r,"Multi_Stock") and y(r,"Temporal_Validation"),
    }


def select_core(rows, target):
    enriched=[]
    for r in rows:
        rr=dict(r)
        rr["Elicit_Core_Score"]=score_record(r)
        flags=category_flags(r)
        rr["Priority_Categories"]="; ".join(k for k,v in flags.items() if v)
        enriched.append(rr)

    ranked=sorted(
        enriched,
        key=lambda r:(
            -int(r["Elicit_Core_Score"]),
            -int(r.get("Year") or 0),
            (r.get("Title") or "").lower()
        )
    )

    selected=[]
    ids=set()
    reason={}

    def add(pred, quota, why):
        added=0
        for r in ranked:
            if len(selected)>=target or added>=quota:
                break
            if r["Master_ID"] in ids:
                continue
            if pred(r):
                selected.append(r)
                ids.add(r["Master_ID"])
                reason[r["Master_ID"]]=why
                added+=1

    # Ensure coverage of the sparse/high-value combinations first.
    add(lambda r: "Four_Data_Families" in r["Priority_Categories"], 6,
        "Four data families")
    add(lambda r: "Three_Data_Families" in r["Priority_Categories"], 8,
        "Technical + fundamental + textual")
    add(lambda r: r.get("Closest_Competitor")=="Yes", 12,
        "Closest competitor")
    add(lambda r: "Fusion_Temporal" in r["Priority_Categories"], 5,
        "Fusion + temporal validation")
    add(lambda r: "Fusion_Optimization" in r["Priority_Categories"], 5,
        "Fusion + optimization/metaheuristic")
    add(lambda r: "Fundamental_XAI" in r["Priority_Categories"], 4,
        "Fundamental + XAI")
    add(lambda r: "Fundamental_Temporal" in r["Priority_Categories"], 4,
        "Fundamental + temporal validation")
    add(lambda r: "Advanced_NLP" in r["Priority_Categories"], 5,
        "Advanced financial NLP/LLM")
    add(lambda r: "India_Multimodal_Fusion" in r["Priority_Categories"], 4,
        "Indian market + multimodal/fusion")
    add(lambda r: "MultiStock_Temporal" in r["Priority_Categories"], 4,
        "Multi-stock + temporal validation")

    # Fill remaining positions by overall core score.
    for r in ranked:
        if len(selected)>=target:
            break
        if r["Master_ID"] not in ids:
            selected.append(r)
            ids.add(r["Master_ID"])
            reason[r["Master_ID"]]="Overall methodology relevance"

    out=[]
    for i,r in enumerate(selected,1):
        rr=dict(r)
        rr["Elicit_Core_Rank"]=i
        rr["Core_Selection_Reason"]=reason[r["Master_ID"]]
        out.append(rr)

    return ranked, out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--target", type=int, default=35)
    args=ap.parse_args()

    rows=read_csv(Path(args.input))
    ranked, core=select_core(rows, min(args.target, len(rows)))

    outdir=Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    write_csv(outdir/"01_ranked_60.csv", ranked)
    write_csv(outdir/"00_elicit_core.csv", core)

    # Extraction template: designed for research-methodology decisions.
    extraction_fields=[
        "Master_ID","Title","Year","Venue","DOI",
        "Elicit_Core_Rank","Core_Selection_Reason",
        "Prediction_Target","Forecast_Horizon","Market_Country",
        "Number_of_Stocks","Data_Frequency","Data_Period",
        "Historical_Variables","Technical_Indicators_Used",
        "Fundamental_Variables_Used","Macroeconomic_Variables",
        "News_Source","Social_Media_Source","Sentiment_Model",
        "FinBERT_BERT_RoBERTa_Used","LLM_FinGPT_Used",
        "Model_Family","Architecture","Attention_Mechanism",
        "Multimodal_Design","Fusion_Strategy",
        "Optimization_Method","Metaheuristic_Method",
        "Train_Validation_Test_Protocol","Temporal_Split",
        "Walk_Forward_or_Rolling","Normalization_Boundary",
        "Feature_Selection_Boundary","Leakage_Control",
        "Baselines","Evaluation_Metrics","Statistical_Tests",
        "Ablation_Study","XAI_Method","Economic_Evaluation",
        "Main_Contribution","Reported_Limitations","Future_Work",
        "Closest_to_Our_Framework","Gap_Relevance_Notes"
    ]

    template=[]
    for r in core:
        row={k:"" for k in extraction_fields}
        for k in ["Master_ID","Title","Year","Venue","DOI",
                  "Elicit_Core_Rank","Core_Selection_Reason"]:
            row[k]=r.get(k,"")
        template.append(row)

    write_csv(outdir/"02_elicit_extraction_template.csv", template, extraction_fields)

    # Elicit prompt text.
    prompt = """Elicit extraction instructions for the attached papers

For each paper, extract ONLY information explicitly supported by the paper.
Use "Not reported" when the paper does not provide the information. Do not infer.

Required fields:
1. Prediction target
2. Forecast horizon
3. Market/country
4. Number of stocks/assets
5. Data frequency
6. Data period
7. Historical market variables
8. Technical indicators
9. Fundamental variables
10. Macroeconomic variables
11. News source
12. Social-media source
13. Sentiment model
14. FinBERT/BERT/RoBERTa use
15. LLM/FinGPT use
16. Model family
17. Architecture
18. Attention mechanism
19. Multimodal design
20. Fusion strategy
21. Optimization method
22. Metaheuristic method
23. Train/validation/test protocol
24. Temporal split method
25. Walk-forward/rolling evaluation
26. Normalization boundary
27. Feature-selection boundary
28. Explicit leakage-control mechanism
29. Baselines
30. Evaluation metrics
31. Statistical significance tests
32. Ablation study
33. Explainability/XAI method
34. Economic/trading evaluation
35. Main methodological contribution
36. Reported limitations
37. Future work

Important:
- Preserve the authors' terminology.
- Do not treat absence from the abstract as absence from the paper.
- Do not infer train-only preprocessing, temporal alignment, or leakage control.
- Mark such fields "Not reported" unless explicitly stated.
"""
    (outdir/"03_elicit_prompt.txt").write_text(prompt,encoding="utf-8")

    cat_counts=Counter()
    for r in core:
        for cat in (r.get("Priority_Categories") or "").split(";"):
            cat=cat.strip()
            if cat:
                cat_counts[cat]+=1

    summary={
        "input_methodology_shortlist":len(rows),
        "elicit_core_target":args.target,
        "elicit_core_selected":len(core),
        "priority_category_counts_in_core":dict(cat_counts),
        "notes":[
            "This is a research-article methodology evidence set, not a systematic-review inclusion set.",
            "Use the extraction template and Elicit prompt for full-text-supported evidence.",
            "Do not claim novelty from metadata counts alone."
        ]
    }
    with open(outdir/"00_summary.json","w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2)

    print(json.dumps(summary,indent=2))
    print(f"\nCore Elicit set: {outdir/'00_elicit_core.csv'}")
    print(f"Extraction template: {outdir/'02_elicit_extraction_template.csv'}")
    print(f"Elicit prompt: {outdir/'03_elicit_prompt.txt'}")


if __name__=="__main__":
    main()
