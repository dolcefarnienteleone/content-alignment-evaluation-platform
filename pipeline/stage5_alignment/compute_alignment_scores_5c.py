"""
Compute Alignment V1 scores for a given set of pairs, availability, and pair summary.

Example
-------
python compute_alignment_scores_5c.py \
  --pairs data/alignment/5b_v0.1/concept_pair_candidates_5b.csv \
  --availability data/alignment/5b_v0.1/content_comparison_availability_5b.csv \
  --pair-summary data/alignment/5b_v0.1/content_pair_summary_5b.csv \
  --output-dir data/alignment/5c

"""
#!/usr/bin/env python3
from __future__ import annotations
import argparse, math, re
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

AUDIENCE_COMPARISONS={"content_to_audience","intent_to_audience"}
SUPPORTED_RELATION_TYPES={"exact","equivalent_to","related_to","manifests_as","contributes_to","elicits"}

def txt(x:Any)->str:
    if x is None:return ""
    try:
        if pd.isna(x):return ""
    except Exception:pass
    return re.sub(r"\s+"," ",str(x)).strip()

def key(x:Any)->str:return txt(x).casefold()

def safe_float(x:Any,default=np.nan)->float:
    try:
        if pd.isna(x):return default
        return float(x)
    except Exception:return default

def clip01(x:Any,default=np.nan)->float:
    v=safe_float(x,default)
    if pd.isna(v):return v
    return float(max(0.0,min(1.0,v)))

def parse_args():
    p=argparse.ArgumentParser(description="5C Alignment V1 scorer")
    p.add_argument("--pairs",type=Path,required=True)
    p.add_argument("--availability",type=Path,required=True)
    p.add_argument("--pair-summary",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--audience-support-transform",choices=["sqrt","linear","none"],default="sqrt")
    p.add_argument("--confidence-floor",type=float,default=0.50)
    return p.parse_args()

def load_pairs(path:Path)->pd.DataFrame:
    df=pd.read_csv(path)
    req=["content_id","comparison_type","source_role","source_node_id","source_category","source_concept","target_role","target_node_id","target_category","target_concept","relationship_type","relationship_strength","relationship_supported"]
    miss=[c for c in req if c not in df.columns]
    if miss:raise ValueError(f"Pairs file missing required columns: {miss}. Columns={list(df.columns)}")
    for c in ["content_id","comparison_type","source_role","source_node_id","source_category","source_concept","target_role","target_node_id","target_category","target_concept","relationship_type","relationship_source"]:
        if c in df.columns:df[c]=df[c].map(txt)
    df["relationship_supported"]=df["relationship_supported"].map(lambda x:x if isinstance(x,bool) else key(x) in {"true","1","yes","y"})
    for c in ["relationship_strength","mean_pair_confidence","source_avg_confidence","target_avg_confidence","source_audience_support_rate","target_audience_support_rate","source_annotation_count","target_annotation_count","source_unique_comment_count","target_unique_comment_count","pair_rank_for_source","semantic_similarity"]:
        if c not in df.columns:df[c]=np.nan
    return df

def load_availability(path:Path)->pd.DataFrame:
    df=pd.read_csv(path)
    req=["content_id","comparison_type","source_role","target_role","alignment_computable","alignment_status"]
    miss=[c for c in req if c not in df.columns]
    if miss:raise ValueError(f"Availability file missing columns: {miss}")
    for c in ["content_id","comparison_type","source_role","target_role","alignment_status"]:df[c]=df[c].map(txt)
    df["alignment_computable"]=df["alignment_computable"].map(lambda x:x if isinstance(x,bool) else key(x) in {"true","1","yes","y"})
    return df

def load_pair_summary(path:Path)->pd.DataFrame:
    df=pd.read_csv(path)
    req=["content_id","comparison_type","source_concept_count","target_concept_count"]
    miss=[c for c in req if c not in df.columns]
    if miss:raise ValueError(f"Pair summary missing columns: {miss}")
    for c in ["content_id","comparison_type"]:df[c]=df[c].map(txt)
    return df

def audience_support_transform(v:Any,mode:str)->float:
    x=clip01(v,np.nan)
    if pd.isna(x):return 0.0
    if mode=="none":return 1.0
    if mode=="linear":return x
    return math.sqrt(x)

def derive_pair_confidence(row:pd.Series,floor:float)->float:
    v=clip01(row.get("mean_pair_confidence"),np.nan)
    if pd.isna(v):
        vals=[clip01(row.get("source_avg_confidence"),np.nan),clip01(row.get("target_avg_confidence"),np.nan)]
        vals=[x for x in vals if not pd.isna(x)]
        v=float(np.mean(vals)) if vals else 1.0
    return float(max(floor,min(1.0,v)))

def supported_mask(df:pd.DataFrame)->pd.Series:
    return df["relationship_supported"].astype(bool) & df["relationship_type"].map(key).isin(SUPPORTED_RELATION_TYPES)

def choose_best_supported_match(pairs:pd.DataFrame,audience_support_mode:str,confidence_floor:float)->pd.DataFrame:
    if pairs.empty:return pd.DataFrame()
    w=pairs.copy()
    w["pair_confidence_for_score"]=w.apply(lambda r:derive_pair_confidence(r,confidence_floor),axis=1)
    w["relationship_strength_for_score"]=w["relationship_strength"].map(lambda x:clip01(x,0.0))
    w["audience_support_factor"]=w.apply(lambda r:audience_support_transform(r.get("target_audience_support_rate"),audience_support_mode) if r["comparison_type"] in AUDIENCE_COMPARISONS else 1.0,axis=1)
    w["pair_score_without_audience"]=w["relationship_strength_for_score"]*w["pair_confidence_for_score"]
    w["pair_score_with_audience"]=w["pair_score_without_audience"]*w["audience_support_factor"]
    sup=w[supported_mask(w)].copy()
    if sup.empty:return pd.DataFrame()
    sup=sup.sort_values(["content_id","comparison_type","source_node_id","pair_score_with_audience","relationship_strength_for_score","pair_confidence_for_score","semantic_similarity"],ascending=[True,True,True,False,False,False,False])
    best=sup.drop_duplicates(["content_id","comparison_type","source_node_id"],keep="first").reset_index(drop=True)
    best["is_best_supported_match"]=True
    return best

def build_source_scores(pairs,best,availability):
    rows=[]
    for _,a in availability[availability["alignment_computable"]].iterrows():
        cid,comp=a["content_id"],a["comparison_type"]
        sub=pairs[(pairs["content_id"]==cid)&(pairs["comparison_type"]==comp)].copy()
        if sub.empty:continue
        source_cols=["source_role","source_node_id","source_category","source_concept","source_avg_confidence","source_annotation_count","source_unique_comment_count","source_audience_support_rate"]
        src=sub[source_cols].drop_duplicates("source_node_id").reset_index(drop=True)
        bm=best[(best["content_id"]==cid)&(best["comparison_type"]==comp)].copy() if not best.empty else pd.DataFrame()
        lookup={r["source_node_id"]:r for _,r in bm.iterrows()} if not bm.empty else {}
        for _,s in src.iterrows():
            sid=s["source_node_id"]
            if sid in lookup:
                b=lookup[sid]
                rows.append({"content_id":cid,"comparison_type":comp,"source_role":s["source_role"],"source_node_id":sid,"source_category":s["source_category"],"source_concept":s["source_concept"],"source_has_supported_match":True,"best_target_role":b["target_role"],"best_target_node_id":b["target_node_id"],"best_target_category":b["target_category"],"best_target_concept":b["target_concept"],"best_relationship_type":b["relationship_type"],"best_relationship_source":b.get("relationship_source",""),"best_relationship_strength":b["relationship_strength_for_score"],"best_pair_confidence":b["pair_confidence_for_score"],"best_target_audience_support_rate":safe_float(b.get("target_audience_support_rate")),"best_audience_support_factor":b["audience_support_factor"],"source_alignment_contribution":b["pair_score_with_audience"]})
            else:
                rows.append({"content_id":cid,"comparison_type":comp,"source_role":s["source_role"],"source_node_id":sid,"source_category":s["source_category"],"source_concept":s["source_concept"],"source_has_supported_match":False,"best_target_role":"","best_target_node_id":"","best_target_category":"","best_target_concept":"","best_relationship_type":"","best_relationship_source":"","best_relationship_strength":0.0,"best_pair_confidence":0.0,"best_target_audience_support_rate":np.nan,"best_audience_support_factor":0.0 if comp in AUDIENCE_COMPARISONS else 1.0,"source_alignment_contribution":0.0})
    return pd.DataFrame(rows)

def build_comparison_scores(source_scores,availability,pair_summary):
    rows=[]
    ps_idx={(r["content_id"],r["comparison_type"]):r for _,r in pair_summary.iterrows()}
    for _,a in availability.iterrows():
        cid,comp=a["content_id"],a["comparison_type"]
        if not bool(a["alignment_computable"]):
            rows.append({"content_id":cid,"comparison_type":comp,"source_role":a["source_role"],"target_role":a["target_role"],"alignment_computable":False,"alignment_status":"insufficient_normalized_signal","source_concept_count":np.nan,"target_concept_count":np.nan,"matched_source_concept_count":np.nan,"matched_target_concept_count":np.nan,"source_concept_coverage":np.nan,"target_concept_coverage":np.nan,"weighted_relationship_strength":np.nan,"weighted_pair_confidence":np.nan,"weighted_audience_support_factor":np.nan,"alignment_score":np.nan})
            continue
        sub=source_scores[(source_scores["content_id"]==cid)&(source_scores["comparison_type"]==comp)].copy()
        ps=ps_idx.get((cid,comp))
        source_count=int(ps["source_concept_count"]) if ps is not None else len(sub)
        target_count=int(ps["target_concept_count"]) if ps is not None else np.nan
        matched=sub[sub["source_has_supported_match"]].copy()
        matched_source_count=int(matched["source_node_id"].nunique()) if not matched.empty else 0
        matched_target_count=int(ps["target_concepts_with_supported_match"]) if ps is not None and "target_concepts_with_supported_match" in ps.index else np.nan
        source_cov=matched_source_count/source_count if source_count else 0.0
        target_cov=safe_float(ps["target_concept_match_coverage"],np.nan) if ps is not None and "target_concept_match_coverage" in ps.index else np.nan
        if matched.empty:
            w_strength=0.0; w_conf=0.0; aud=0.0 if comp in AUDIENCE_COMPARISONS else 1.0; score=0.0; status="ready_no_supported_relationship"
        else:
            w_strength=float(matched["best_relationship_strength"].mean())
            w_conf=float(matched["best_pair_confidence"].mean())
            aud=float(matched["best_audience_support_factor"].mean()) if comp in AUDIENCE_COMPARISONS else 1.0
            score=float(max(0.0,min(1.0,source_cov*w_strength*w_conf*aud)))
            status="ready"
        rows.append({"content_id":cid,"comparison_type":comp,"source_role":a["source_role"],"target_role":a["target_role"],"alignment_computable":True,"alignment_status":status,"source_concept_count":source_count,"target_concept_count":target_count,"matched_source_concept_count":matched_source_count,"matched_target_concept_count":matched_target_count,"source_concept_coverage":source_cov,"target_concept_coverage":target_cov,"weighted_relationship_strength":w_strength,"weighted_pair_confidence":w_conf,"weighted_audience_support_factor":aud,"alignment_score":score})
    return pd.DataFrame(rows)

def build_content_summary(scores):
    rows=[]
    for cid,g in scores.groupby("content_id"):
        row={"content_id":cid}
        for _,r in g.iterrows():
            p=r["comparison_type"]
            row[f"{p}_computable"]=bool(r["alignment_computable"])
            row[f"{p}_status"]=r["alignment_status"]
            row[f"{p}_score"]=r["alignment_score"]
            row[f"{p}_source_coverage"]=r["source_concept_coverage"]
            row[f"{p}_target_coverage"]=r["target_concept_coverage"]
            row[f"{p}_relationship_strength"]=r["weighted_relationship_strength"]
            row[f"{p}_audience_support_factor"]=r["weighted_audience_support_factor"]
        comp=g.loc[g["alignment_computable"],"alignment_score"].dropna()
        row["computable_comparison_count"]=int(g["alignment_computable"].sum())
        row["noncomputable_comparison_count"]=int((~g["alignment_computable"]).sum())
        row["mean_computable_alignment_score"]=float(comp.mean()) if len(comp) else np.nan
        row["overall_score_defined"]=False
        row["overall_score"]=np.nan
        row["overall_score_note"]="Not defined in Alignment V1; comparison scores remain separate."
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    args=parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    pairs=load_pairs(args.pairs); availability=load_availability(args.availability); pair_summary=load_pair_summary(args.pair_summary)
    best=choose_best_supported_match(pairs,args.audience_support_transform,args.confidence_floor)
    source_scores=build_source_scores(pairs,best,availability)
    comp_scores=build_comparison_scores(source_scores,availability,pair_summary)
    content_summary=build_content_summary(comp_scores)
    best.to_csv(args.output_dir/"alignment_pair_best_match_5c.csv",index=False)
    source_scores.to_csv(args.output_dir/"alignment_source_concept_score_5c.csv",index=False)
    comp_scores.to_csv(args.output_dir/"alignment_comparison_score_5c.csv",index=False)
    content_summary.to_csv(args.output_dir/"alignment_content_summary_5c.csv",index=False)
    build=pd.DataFrame([
        {"metric":"content_count","value":availability["content_id"].nunique()},
        {"metric":"comparison_rows","value":len(comp_scores)},
        {"metric":"comparison_rows_computable","value":int(comp_scores["alignment_computable"].sum())},
        {"metric":"comparison_rows_insufficient_signal","value":int((~comp_scores["alignment_computable"]).sum())},
        {"metric":"supported_best_match_rows","value":len(best)},
        {"metric":"source_concept_score_rows","value":len(source_scores)},
        {"metric":"matched_source_concept_rows","value":int(source_scores["source_has_supported_match"].sum()) if not source_scores.empty else 0},
        {"metric":"unmatched_source_concept_rows","value":int((~source_scores["source_has_supported_match"]).sum()) if not source_scores.empty else 0},
        {"metric":"audience_support_transform","value":args.audience_support_transform},
        {"metric":"confidence_floor","value":args.confidence_floor},
        {"metric":"overall_score_defined","value":False},
    ])
    build.to_csv(args.output_dir/"build_summary_5c.csv",index=False)
    print("\n=== 5C Alignment V1 scoring complete ===")
    print(build.to_string(index=False))
    print("\nComparison score summary:")
    print(comp_scores.groupby("comparison_type").agg(rows=("content_id","size"),computable=("alignment_computable","sum"),mean_score=("alignment_score","mean"),median_score=("alignment_score","median"),mean_source_coverage=("source_concept_coverage","mean"),mean_target_coverage=("target_concept_coverage","mean"),mean_relationship_strength=("weighted_relationship_strength","mean"),mean_audience_support=("weighted_audience_support_factor","mean")).round(4).to_string())
    print("\nAlignment status:")
    print(comp_scores.groupby(["comparison_type","alignment_status"]).size().to_string())
    print("\nPer-content scores:")
    cols=["content_id","intent_to_content_score","content_to_audience_score","intent_to_audience_score","computable_comparison_count"]
    cols=[c for c in cols if c in content_summary.columns]
    print(content_summary[cols].round(4).to_string(index=False))

if __name__=="__main__":main()
