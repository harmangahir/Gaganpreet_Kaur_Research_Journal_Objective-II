#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from collections import Counter
from pathlib import Path

LABELS=["Include","Maybe","Exclude","Background"]

def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields=list(rows[0].keys()) if rows else []
    with open(path,"w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def kappa(pairs):
    if not pairs: return None
    n=len(pairs)
    po=sum(a==b for a,b in pairs)/n
    ca=Counter(a for a,b in pairs); cb=Counter(b for a,b in pairs)
    pe=sum((ca[l]/n)*(cb[l]/n) for l in LABELS)
    if pe==1: return 1.0 if po==1 else None
    return (po-pe)/(1-pe)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--screened", required=True)
    ap.add_argument("--manual", required=True)
    ap.add_argument("--output-dir", required=True)
    args=ap.parse_args()

    screened=read_csv(Path(args.screened))
    manual=read_csv(Path(args.manual))
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)

    completed=[r for r in manual if (r.get("Human_Decision") or "").strip() in LABELS]
    pending=len(manual)-len(completed)

    qc=[r for r in completed if (r.get("Queue_Reason") or "").startswith("QC sample")]
    pairs=[(r["AI_Decision"],r["Human_Decision"]) for r in qc]
    disagreements=[r for r in qc if r["AI_Decision"]!=r["Human_Decision"]]
    agreement=sum(a==b for a,b in pairs)/len(pairs) if pairs else None

    manual_by_id={r["Master_ID"]:r for r in manual}
    reconciled=[]

    for r in screened:
        rr=dict(r)
        m=manual_by_id.get(r["Master_ID"])
        hd=(m.get("Human_Decision","").strip() if m else "")
        hc=(m.get("Human_Exclusion_Code","").strip() if m else "")
        notes=(m.get("Reviewer_Notes","").strip() if m else "")

        rr["AI_Original_Decision"]=r["Screening_Decision"]
        if hd in LABELS:
            rr["Final_Decision"]=hd
            rr["Manual_Override"]="Verified" if hd==r["Screening_Decision"] else "Changed"
            if hd=="Exclude":
                rr["Final_Exclusion_Code"]=hc or r.get("Exclusion_Code","")
            elif hd=="Background":
                rr["Final_Exclusion_Code"]="E09"
            else:
                rr["Final_Exclusion_Code"]=""
        else:
            rr["Final_Decision"]=r["Screening_Decision"]
            rr["Manual_Override"]="Not reviewed"
            rr["Final_Exclusion_Code"]=r.get("Exclusion_Code","")
        rr["Manual_Reviewer_Notes"]=notes
        reconciled.append(rr)

    fulltext=[r for r in reconciled if r["Final_Decision"] in ("Include","Maybe")]
    counts=Counter(r["Final_Decision"] for r in reconciled)

    write_csv(outdir/"reconciled_screening.csv",reconciled,list(reconciled[0].keys()))
    write_csv(outdir/"full_text_queue.csv",fulltext,list(reconciled[0].keys()))
    write_csv(outdir/"qc_disagreements.csv",disagreements,list(manual[0].keys()))

    matrix={a:{b:0 for b in LABELS} for a in LABELS}
    for a,b in pairs: matrix[a][b]+=1

    summary={
        "status":"QC_COMPLETE" if pending==0 else "QC_INCOMPLETE",
        "manual_review_queue_total":len(manual),
        "manual_review_completed":len(completed),
        "manual_review_pending":pending,
        "qc_sample_completed":len(pairs),
        "qc_raw_agreement":agreement,
        "qc_cohens_kappa":kappa(pairs),
        "qc_disagreements":len(disagreements),
        "qc_confusion_matrix_ai_rows_human_columns":matrix,
        "reconciled_decision_counts":dict(counts),
        "records_proceeding_to_full_text":len(fulltext),
        "warning":"Do not freeze PRISMA/title-abstract counts while status is QC_INCOMPLETE. If disagreements are systematic, revise rules and rerun the complete automated screen."
    }
    with open(outdir/"qc_summary.json","w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2)

    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
