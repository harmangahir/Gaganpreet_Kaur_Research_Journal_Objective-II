#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, os, textwrap
from datetime import datetime, timezone
from pathlib import Path

VALID_DECISIONS = {"i":"Include","e":"Exclude","b":"Background","m":"Maybe"}

EXCLUSION_CODES = {
"E01":"Cryptocurrency/digital-asset forecasting without stock/equity forecasting",
"E02":"Forex/currency forecasting without stock/equity forecasting",
"E03":"Commodity/energy forecasting without stock/equity forecasting",
"E04":"Portfolio optimization/asset allocation without an eligible stock forecasting task",
"E05":"Trading/RL strategy without an eligible stock forecasting task",
"E06":"Non-financial time-series forecasting / no stock-equity context",
"E07":"Sentiment analysis related to stocks but no eligible stock forecasting task",
"E08":"Pure econometric/statistical forecasting without AI/ML/DL",
"E09":"Review/survey/bibliometric paper",
"E10":"Editorial/commentary/non-primary research item",
"E11":"Outside frozen 2024-2026 publication window",
"E12":"Stock-market topic but no eligible price/return/direction/trend forecasting task",
"E13":"AI/ML forecasting present but direct stock/equity relevance not established",
"E14":"Duplicate",
"E15":"Insufficient metadata",
}

def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_rows(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)

def show(label, value, width=105):
    print(f"\n{label}:")
    value=(value or "").strip()
    if not value:
        print("  [Not available]")
    else:
        for line in textwrap.wrap(value, width=width):
            print("  "+line)

def ask_code():
    print("\nExclusion codes:")
    for c,r in EXCLUSION_CODES.items():
        print(f"  {c}: {r}")
    while True:
        x=input("Enter exclusion code: ").strip().upper()
        if x in EXCLUSION_CODES:
            return x
        print("Use E01-E15.")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--reviewer", default="")
    ap.add_argument("--show-ai", action="store_true")
    args=ap.parse_args()

    inp=Path(args.input); out=Path(args.output)
    source=out if out.exists() else inp
    rows=read_rows(source)
    extras=["Human_Decision","Human_Exclusion_Code","Reviewer_Notes","Reviewer","Reviewed_UTC"]
    fields=list(rows[0].keys())
    for e in extras:
        if e not in fields:
            fields.append(e)
    for r in rows:
        for e in extras:
            r.setdefault(e,"")

    def done_count():
        return sum((r.get("Human_Decision") or "").strip() != "" for r in rows)

    print("="*78)
    print("MANUAL TITLE/ABSTRACT REVIEW")
    print("="*78)
    print(f"Queue: {len(rows)}")
    print(f"Already reviewed: {done_count()}")
    print(f"Remaining: {len(rows)-done_count()}")
    print("Mode:", "AI visible" if args.show_ai else "BLINDED to AI decision")
    print("="*78)

    for r in rows:
        if (r.get("Human_Decision") or "").strip():
            continue

        print("\n"+"="*78)
        print(f"Progress {done_count()+1}/{len(rows)} | {r.get('Master_ID','')}")
        print("="*78)
        print(f"Year: {r.get('Year','')}")
        print(f"Venue: {r.get('Venue','')}")
        print(f"DOI: {r.get('DOI','')}")
        show("TITLE", r.get("Title",""))
        show("ABSTRACT", r.get("Abstract",""))
        show("KEYWORDS", r.get("Keywords",""))

        if args.show_ai:
            print(f"\nQueue reason: {r.get('Queue_Reason','')}")
            print(f"AI decision: {r.get('AI_Decision','')}")
            print(f"AI exclusion code: {r.get('Exclusion_Code','')}")
            show("AI reason", r.get("AI_Reason",""))

        print("\n[i] Include  [e] Exclude  [b] Background  [m] Maybe/full-text  [s] Skip  [q] Save & quit")
        while True:
            cmd=input("> ").strip().lower()
            if cmd=="q":
                write_rows(out,rows,fields)
                print(f"Saved: {out} | Reviewed {done_count()}/{len(rows)}")
                return
            if cmd=="s":
                break
            if cmd not in VALID_DECISIONS:
                print("Use i, e, b, m, s, or q.")
                continue

            decision=VALID_DECISIONS[cmd]
            r["Human_Decision"]=decision
            if decision=="Exclude":
                r["Human_Exclusion_Code"]=ask_code()
            elif decision=="Background":
                r["Human_Exclusion_Code"]="E09"
            else:
                r["Human_Exclusion_Code"]=""
            r["Reviewer_Notes"]=input("Reviewer note (Enter to leave blank): ").strip()
            r["Reviewer"]=args.reviewer
            r["Reviewed_UTC"]=datetime.now(timezone.utc).isoformat()
            write_rows(out,rows,fields)
            print(f"Saved automatically. Progress: {done_count()}/{len(rows)}")
            break

    write_rows(out,rows,fields)
    print(f"\nQUEUE COMPLETE: {done_count()}/{len(rows)}")
    print(f"Saved to: {out}")

if __name__=="__main__":
    main()
