#!/usr/bin/env python3
"""
adjudicate_qc_disagreements.py

Structured, blinded adjudication of the 58 QC disagreements from Screening V1.

Why:
The first QC round showed low agreement and several apparent inconsistencies in
manual decisions. This script does NOT ask for a direct Include/Exclude judgment
first. Instead, it asks the reviewer to answer the frozen eligibility criteria
one-by-one and derives the adjudicated decision.

Frozen primary-scope definition:
- Primary empirical research
- Stock/equity/stock-index domain
- Eligible prediction target: PRICE, RETURN, DIRECTION/MOVEMENT, or TREND
- AI/ML/DL forecasting method
- Empirical predictive evaluation

Volatility-only, risk-only, portfolio-only, trading-only, sentiment-only,
non-financial, and pure econometric studies are not primary eligible evidence.
Reviews/surveys/bibliometric studies are Background.

Recommended:
Use a second independent reviewer if available. If not, perform this adjudication
as a fresh blinded re-review and report that transparently.

Example:
python adjudicate_qc_disagreements.py \
  --screened screening_results/stage3_tagged.csv \
  --false-includes screening_results/qc_diagnosis/false_includes_AI_include_human_exclude.csv \
  --false-excludes screening_results/qc_diagnosis/false_excludes_AI_exclude_human_include.csv \
  --output screening_results/qc_diagnosis/adjudicated_58.csv \
  --reviewer "Reviewer-2"

The script automatically resumes from the output file if it exists.
"""

from __future__ import annotations
import argparse, csv, os, textwrap
from datetime import datetime, timezone
from pathlib import Path


VALID_YNU = {"y": "Yes", "n": "No", "u": "Unclear"}
TARGETS = {
    "p": "Price",
    "r": "Return",
    "d": "Direction/Movement",
    "t": "Trend",
    "v": "Volatility/Risk only",
    "o": "Other/non-eligible target",
    "u": "Unclear",
}


def read_csv(path: Path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def wrap(label, value, width=108):
    print(f"\n{label}:")
    value = (value or "").strip()
    if not value:
        print("  [Not available]")
        return
    for line in textwrap.wrap(value, width=width):
        print("  " + line)


def ask_ynu(prompt):
    while True:
        x = input(f"{prompt} [y/n/u]: ").strip().lower()
        if x in VALID_YNU:
            return VALID_YNU[x]
        print("Use y = Yes, n = No, u = Unclear.")


def ask_target():
    print("\nEligible primary targets are Price, Return, Direction/Movement, or Trend.")
    print("Volatility/Risk alone is NOT a primary eligible target under the frozen scope.")
    print("[p] Price  [r] Return  [d] Direction/Movement  [t] Trend")
    print("[v] Volatility/Risk only  [o] Other/non-eligible  [u] Unclear")
    while True:
        x = input("Primary prediction target: ").strip().lower()
        if x in TARGETS:
            return TARGETS[x]
        print("Use p, r, d, t, v, o, or u.")


def derive_decision(primary, review, stock, target, ai, empirical):
    # Background first
    if review == "Yes":
        return "Background", "E09", "Review/survey/bibliometric paper."

    # Any uncertainty on a core criterion -> Maybe
    if "Unclear" in {primary, stock, ai, empirical} or target == "Unclear":
        return "Maybe", "", "At least one core eligibility criterion remains unclear."

    if primary == "No":
        return "Exclude", "E10", "Not a primary empirical research study."

    if stock == "No":
        return "Exclude", "E06", "No stock/equity/stock-index domain."

    if target in {"Volatility/Risk only", "Other/non-eligible target"}:
        return "Exclude", "E12", "Prediction target is outside the frozen primary scope."

    if ai == "No":
        return "Exclude", "E08", "No AI/ML/DL forecasting method."

    if empirical == "No":
        return "Exclude", "E13", "No empirical predictive evaluation established."

    return "Include", "", "All frozen primary eligibility criteria are satisfied."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screened", required=True)
    ap.add_argument("--false-includes", required=True)
    ap.add_argument("--false-excludes", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--reviewer", default="")
    ap.add_argument("--show-old-decisions", action="store_true",
                    help="Not recommended for blinded adjudication.")
    args = ap.parse_args()

    screened_rows = read_csv(Path(args.screened))
    screened = {r["Master_ID"]: r for r in screened_rows}

    disagreement_rows = (
        read_csv(Path(args.false_includes))
        + read_csv(Path(args.false_excludes))
    )

    # Unique by Master_ID, stable order
    seen = set()
    base = []
    for r in disagreement_rows:
        if r["Master_ID"] not in seen:
            base.append(r)
            seen.add(r["Master_ID"])

    output = Path(args.output)

    if output.exists():
        rows = read_csv(output)
    else:
        rows = []
        for d in base:
            s = screened.get(d["Master_ID"], {})
            rows.append({
                "Master_ID": d["Master_ID"],
                "Title": d.get("Title", s.get("Title", "")),
                "Year": d.get("Year", s.get("Year", "")),
                "Venue": d.get("Venue", s.get("Venue", "")),
                "DOI": d.get("DOI", s.get("DOI", "")),
                "Abstract": s.get("Abstract", ""),
                "Keywords": s.get("Keywords", ""),
                "Old_AI_Decision": d.get("AI_Decision", ""),
                "Old_Human_Decision": d.get("Human_Decision", ""),
                "Old_AI_Exclusion_Code": d.get("AI_Exclusion_Code", ""),
                "Old_Human_Exclusion_Code": d.get("Human_Exclusion_Code", ""),
                "Primary_Empirical": "",
                "Review_Survey": "",
                "Stock_Equity_Domain": "",
                "Prediction_Target": "",
                "AI_ML_DL_Method": "",
                "Empirical_Predictive_Evaluation": "",
                "Adjudicated_Decision": "",
                "Adjudicated_Exclusion_Code": "",
                "Adjudication_Rationale": "",
                "Reviewer": "",
                "Reviewed_UTC": "",
                "Reviewer_Notes": "",
            })

    fields = list(rows[0].keys())

    def reviewed_count():
        return sum(bool((r.get("Adjudicated_Decision") or "").strip()) for r in rows)

    print("=" * 82)
    print("BLINDED STRUCTURED ADJUDICATION OF QC DISAGREEMENTS")
    print("=" * 82)
    print(f"Records: {len(rows)}")
    print(f"Already adjudicated: {reviewed_count()}")
    print(f"Remaining: {len(rows) - reviewed_count()}")
    print("Frozen eligible targets: Price, Return, Direction/Movement, Trend")
    print("=" * 82)

    for r in rows:
        if (r.get("Adjudicated_Decision") or "").strip():
            continue

        print("\n" + "=" * 82)
        print(f"Record {reviewed_count()+1}/{len(rows)} | {r['Master_ID']}")
        print("=" * 82)
        print(f"Year: {r.get('Year','')}")
        print(f"Venue: {r.get('Venue','')}")
        print(f"DOI: {r.get('DOI','')}")
        wrap("TITLE", r.get("Title", ""))
        wrap("ABSTRACT", r.get("Abstract", ""))
        wrap("KEYWORDS", r.get("Keywords", ""))

        if args.show_old_decisions:
            print("\n--- Previous decisions (VISIBLE) ---")
            print("Old AI:", r.get("Old_AI_Decision", ""),
                  r.get("Old_AI_Exclusion_Code", ""))
            print("Old Human:", r.get("Old_Human_Decision", ""),
                  r.get("Old_Human_Exclusion_Code", ""))

        print("\nAnswer the frozen criteria independently.")

        r["Primary_Empirical"] = ask_ynu(
            "1. Is this a primary empirical research study?"
        )
        r["Review_Survey"] = ask_ynu(
            "2. Is it a review/survey/bibliometric article?"
        )
        r["Stock_Equity_Domain"] = ask_ynu(
            "3. Is the empirical target/domain stock, equity, or a stock index?"
        )
        r["Prediction_Target"] = ask_target()
        r["AI_ML_DL_Method"] = ask_ynu(
            "5. Does it use an AI/ML/DL forecasting method?"
        )
        r["Empirical_Predictive_Evaluation"] = ask_ynu(
            "6. Does it empirically evaluate predictive/forecasting performance?"
        )

        decision, code, rationale = derive_decision(
            r["Primary_Empirical"],
            r["Review_Survey"],
            r["Stock_Equity_Domain"],
            r["Prediction_Target"],
            r["AI_ML_DL_Method"],
            r["Empirical_Predictive_Evaluation"],
        )

        print("\nDerived adjudication:")
        print("  Decision:", decision)
        print("  Code:", code or "[none]")
        print("  Rationale:", rationale)

        confirm = input(
            "Accept derived adjudication? [Enter=yes / r=redo / q=save & quit]: "
        ).strip().lower()

        if confirm == "q":
            write_csv(output, rows, fields)
            print(f"Saved: {output}")
            return

        if confirm == "r":
            for col in [
                "Primary_Empirical", "Review_Survey", "Stock_Equity_Domain",
                "Prediction_Target", "AI_ML_DL_Method",
                "Empirical_Predictive_Evaluation"
            ]:
                r[col] = ""
            continue

        r["Adjudicated_Decision"] = decision
        r["Adjudicated_Exclusion_Code"] = code
        r["Adjudication_Rationale"] = rationale
        r["Reviewer"] = args.reviewer
        r["Reviewed_UTC"] = datetime.now(timezone.utc).isoformat()
        r["Reviewer_Notes"] = input(
            "Optional adjudication note (Enter to leave blank): "
        ).strip()

        write_csv(output, rows, fields)
        print(f"Saved automatically. Progress: {reviewed_count()}/{len(rows)}")

    write_csv(output, rows, fields)
    print("\n" + "=" * 82)
    print(f"ADJUDICATION COMPLETE: {reviewed_count()}/{len(rows)}")
    print(f"Saved to: {output}")
    print("=" * 82)


if __name__ == "__main__":
    main()
