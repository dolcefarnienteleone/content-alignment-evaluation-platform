"""
Finalize relationship rules v0.1
------------------------------

Example
-------
python finalize_relationship_rules_5b1.py \
  --review-queue data/alignment/5b/relationship_review_queue_5b_reviewed.csv \
  --base-rules data/alignment/5b/relationship_rules_5b.csv \
  --output-dir data/alignment/5b/rules_v0.1
"""

#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

ACTIVE_RULE_COLUMNS = [
    "source_concept","source_category","source_role",
    "target_concept","target_category","target_role",
    "relationship_type","relationship_strength","directional",
    "rule_status","rule_source","notes",
]
ALLOWED_DECISIONS = {"approve_relationship","reject_relationship","defer"}
ALLOWED_TYPES = {"equivalent_to","related_to","manifests_as","contributes_to","elicits"}

def txt(x):
    if x is None: return ""
    try:
        if pd.isna(x): return ""
    except Exception:
        pass
    return re.sub(r"\s+"," ",str(x)).strip()

def key(x): return txt(x).casefold()

def safe_float(x, default=np.nan):
    try:
        if pd.isna(x): return default
        return float(x)
    except Exception:
        return default

def boolish(x):
    if isinstance(x,bool): return x
    return key(x) in {"1","true","yes","y","t"}

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--review-queue", type=Path, required=True)
    p.add_argument("--base-rules", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()

def validate_review(df):
    issues=[]
    required=[
        "source_role","source_category","source_concept",
        "target_role","target_category","target_concept",
        "human_relationship_decision","human_relationship_type",
        "human_relationship_strength","human_review_notes","human_review_status",
    ]
    missing=[c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Review queue missing required columns: {missing}. Columns={list(df.columns)}")
    for i,r in df.iterrows():
        decision=key(r["human_relationship_decision"])
        status=key(r["human_review_status"])
        rel_type=key(r["human_relationship_type"])
        strength=safe_float(r["human_relationship_strength"])
        rownum=i+2
        if status!="completed":
            issues.append([rownum,"error","human_review_status_not_completed",txt(r["human_review_status"])])
        if decision not in ALLOWED_DECISIONS:
            issues.append([rownum,"error","invalid_human_relationship_decision",txt(r["human_relationship_decision"])])
        if decision=="approve_relationship":
            if rel_type not in ALLOWED_TYPES:
                issues.append([rownum,"error","invalid_or_missing_relationship_type",txt(r["human_relationship_type"])])
            if pd.isna(strength) or strength<=0 or strength>1:
                issues.append([rownum,"error","invalid_or_missing_relationship_strength",txt(r["human_relationship_strength"])])
    return pd.DataFrame(issues, columns=["row_number","severity","issue","detail"])

def standardize_base_rules(base):
    out=base.copy()
    for c in ACTIVE_RULE_COLUMNS:
        if c not in out.columns:
            if c=="relationship_strength": out[c]=0.8
            elif c=="directional": out[c]=True
            elif c=="rule_status": out[c]="approved"
            else: out[c]=""
    out=out[ACTIVE_RULE_COLUMNS].copy()
    out["relationship_strength"]=out["relationship_strength"].map(lambda x:safe_float(x,0.8))
    out["directional"]=out["directional"].map(boolish)
    out["rule_status"]="approved"
    out["rule_source"]=out["rule_source"].map(lambda x:txt(x) or "preexisting_curated_rule")
    return out

def review_to_governed_rules(review):
    rows = []
    for _, r in review.iterrows():
        decision = key(r["human_relationship_decision"])

        if decision == "approve_relationship":
            status = "approved"
            rel_type = txt(r["human_relationship_type"])
            strength = float(r["human_relationship_strength"])
        elif decision == "reject_relationship":
            status = "rejected"
            rel_type = ""
            strength = 0.0
        elif decision == "defer":
            status = "deferred"
            rel_type = ""
            strength = 0.0
        else:
            continue

        rows.append({
            "source_concept": txt(r["source_concept"]),
            "source_category": txt(r["source_category"]),
            "source_role": txt(r["source_role"]),
            "target_concept": txt(r["target_concept"]),
            "target_category": txt(r["target_category"]),
            "target_role": txt(r["target_role"]),
            "relationship_type": rel_type,
            "relationship_strength": strength,
            "directional": True,
            "rule_status": status,
            "rule_source": "5b1_human_review_v0.1",
            "notes": txt(r["human_review_notes"]),
        })

    return pd.DataFrame(rows, columns=ACTIVE_RULE_COLUMNS)

def dedupe_rules(df):
    out=df.copy()
    out["_k"]=(
        out["source_role"].map(key)+"||"+out["source_category"].map(key)+"||"+out["source_concept"].map(key)+"||"+
        out["target_role"].map(key)+"||"+out["target_category"].map(key)+"||"+out["target_concept"].map(key)
    )
    out["_p"]=np.where(out["rule_source"].map(key)=="5b1_human_review_v0.1",2,1)
    return (
        out.sort_values(["_k","_p","relationship_strength"],ascending=[True,False,False])
        .drop_duplicates("_k",keep="first")
        .drop(columns=["_k","_p"])
        .reset_index(drop=True)
    )

def main():
    args=parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    review=pd.read_csv(args.review_queue)
    base=pd.read_csv(args.base_rules)

    validation=validate_review(review)
    validation.to_csv(args.output_dir/"relationship_rules_v0.1_validation.csv",index=False)
    errors=int((validation["severity"]=="error").sum()) if not validation.empty else 0
    if errors:
        print(validation.to_string(index=False))
        raise ValueError(f"Relationship review validation failed with {errors} error(s).")

    base_std=standardize_base_rules(base)
    reviewed=review_to_governed_rules(review)
    final_rules=dedupe_rules(pd.concat([base_std,reviewed],ignore_index=True))
    final_rules.to_csv(args.output_dir/"relationship_rules_v0.1.csv",index=False)

    change=review.copy()
    change["active_in_relationship_rules_v0.1"]=change["human_relationship_decision"].map(key)=="approve_relationship"
    change["final_relationship_type"]=np.where(change["active_in_relationship_rules_v0.1"],change["human_relationship_type"],"")
    change["final_relationship_strength"]=np.where(change["active_in_relationship_rules_v0.1"],change["human_relationship_strength"],np.nan)
    change.to_csv(args.output_dir/"relationship_rules_v0.1_change_log.csv",index=False)

    decisions=review["human_relationship_decision"].map(key)
    summary=pd.DataFrame([
        {"metric":"review_rows","value":len(review)},
        {"metric":"review_completed_rows","value":int((review["human_review_status"].map(key)=="completed").sum())},
        {"metric":"approve_relationship_rows","value":int((decisions=="approve_relationship").sum())},
        {"metric":"reject_relationship_rows","value":int((decisions=="reject_relationship").sum())},
        {"metric":"defer_rows","value":int((decisions=="defer").sum())},
        {"metric":"base_rule_count","value":len(base_std)},
        {"metric":"human_governed_rule_rows","value":len(reviewed)},
        {"metric":"human_approved_rule_rows","value":int((reviewed["rule_status"]=="approved").sum())},
        {"metric":"human_rejected_rule_rows","value":int((reviewed["rule_status"]=="rejected").sum())},
        {"metric":"human_deferred_rule_rows","value":int((reviewed["rule_status"]=="deferred").sum())},
        {"metric":"final_active_rule_count","value":len(final_rules)},
        {"metric":"validation_error_count","value":errors},
    ])
    summary.to_csv(args.output_dir/"relationship_rules_v0.1_summary.csv",index=False)

    print("\n=== Relationship Rules v0.1 finalized ===")
    print(summary.to_string(index=False))
    print("\nFinal rule status:")
    print(final_rules["rule_status"].value_counts().to_string())
    print("\nApproved relationship types:")
    print(final_rules.loc[final_rules["rule_status"]=="approved","relationship_type"].value_counts().to_string())
    print("\nActive rule sources:")
    print(final_rules["rule_source"].value_counts().to_string())

if __name__=="__main__":
    main()
