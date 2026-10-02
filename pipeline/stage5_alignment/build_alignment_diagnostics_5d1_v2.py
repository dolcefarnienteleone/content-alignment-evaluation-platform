"""
Build deterministic alignment diagnostics for 5D.1 v2.

Example
-------
python build_alignment_diagnostics_5d1_v2.py \
  --source-scores data/alignment/5c/alignment_source_concept_score_5c.csv \
  --comparison-scores data/alignment/5c/alignment_comparison_score_5c.csv \
  --pairs data/alignment/5b_v0.1/concept_pair_candidates_5b.csv \
  --availability data/alignment/5b_v0.1/content_comparison_availability_5b.csv \
  --output-dir data/alignment/5d_v2
"""

#!/usr/bin/env python3
from __future__ import annotations
import argparse, math, re
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

AUDIENCE_COMPARISONS = {"content_to_audience", "intent_to_audience"}
SUPPORTED_TYPES = {"exact","equivalent_to","related_to","manifests_as","contributes_to","elicits"}
AUDIENCE_CATEGORY_TO_SIGNAL = {
    "creator perception": "unexpected_perception",
    "relationship perception": "unexpected_perception",
    "emotional response": "emergent_response",
    "behavioral response": "audience_action_signal",
    "discussion topics": "audience_discussion_signal",
    "community signals": "community_signal",
}
INTERPRETATION_SCOPE = (
    "Governed ontology/profile interpretation only; absence of a supported counterpart "
    "does not prove absence of human perception."
)

def txt(x: Any) -> str:
    if x is None: return ""
    try:
        if pd.isna(x): return ""
    except Exception: pass
    return re.sub(r"\s+", " ", str(x)).strip()

def key(x: Any) -> str: return txt(x).casefold()

def safe_float(x: Any, default=np.nan) -> float:
    try:
        if pd.isna(x): return default
        return float(x)
    except Exception: return default

def boolish(x: Any) -> bool:
    if isinstance(x, bool): return x
    return key(x) in {"true","1","yes","y","t"}

def parse_args():
    p=argparse.ArgumentParser(description="5D.1 v2 deterministic alignment diagnostics")
    p.add_argument("--source-scores", type=Path, required=True)
    p.add_argument("--comparison-scores", type=Path, required=True)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--availability", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--aligned-threshold", type=float, default=0.80)
    p.add_argument("--partial-threshold", type=float, default=0.60)
    return p.parse_args()

def load_source_scores(path):
    df=pd.read_csv(path)
    required=["content_id","comparison_type","source_role","source_node_id","source_category","source_concept",
              "source_has_supported_match","best_target_role","best_target_node_id","best_target_category",
              "best_target_concept","best_relationship_type","best_relationship_source","best_relationship_strength",
              "best_pair_confidence","best_target_audience_support_rate","best_audience_support_factor",
              "source_alignment_contribution"]
    miss=[c for c in required if c not in df.columns]
    if miss: raise ValueError(f"Source score file missing required columns: {miss}")
    for c in ["content_id","comparison_type","source_role","source_node_id","source_category","source_concept",
              "best_target_role","best_target_node_id","best_target_category","best_target_concept",
              "best_relationship_type","best_relationship_source"]:
        df[c]=df[c].map(txt)
    df["source_has_supported_match"]=df["source_has_supported_match"].map(boolish)
    return df

def load_comparison_scores(path):
    df=pd.read_csv(path)
    required=["content_id","comparison_type","alignment_computable","alignment_status","source_concept_coverage",
              "target_concept_coverage","weighted_relationship_strength","weighted_pair_confidence",
              "weighted_audience_support_factor","alignment_score"]
    miss=[c for c in required if c not in df.columns]
    if miss: raise ValueError(f"Comparison score file missing columns: {miss}")
    for c in ["content_id","comparison_type","alignment_status"]: df[c]=df[c].map(txt)
    df["alignment_computable"]=df["alignment_computable"].map(boolish)
    return df

def load_pairs(path):
    df=pd.read_csv(path)
    required=["content_id","comparison_type","source_role","source_node_id","source_category","source_concept",
              "target_role","target_node_id","target_category","target_concept","relationship_type",
              "relationship_supported","relationship_strength","target_audience_support_rate","target_avg_confidence"]
    miss=[c for c in required if c not in df.columns]
    if miss: raise ValueError(f"Pairs file missing columns: {miss}")
    for c in ["content_id","comparison_type","source_role","source_node_id","source_category","source_concept",
              "target_role","target_node_id","target_category","target_concept","relationship_type","relationship_source"]:
        if c in df.columns: df[c]=df[c].map(txt)
    df["relationship_supported"]=df["relationship_supported"].map(boolish)
    return df

def load_availability(path):
    df=pd.read_csv(path)
    required=["content_id","comparison_type","source_role","target_role","alignment_computable","alignment_status"]
    miss=[c for c in required if c not in df.columns]
    if miss: raise ValueError(f"Availability file missing columns: {miss}")
    for c in ["content_id","comparison_type","source_role","target_role","alignment_status"]: df[c]=df[c].map(txt)
    df["alignment_computable"]=df["alignment_computable"].map(boolish)
    return df

def classify_source_insights(source_scores, aligned_threshold, partial_threshold):
    rows=[]
    for _,r in source_scores.iterrows():
        matched=bool(r["source_has_supported_match"])
        strength=safe_float(r["best_relationship_strength"],0.0)
        if not matched:
            insight_type="missing_resonance"
            interpretation="No governed target counterpart surfaced in the current normalized profile."
        elif strength >= aligned_threshold:
            insight_type="aligned_concept"; interpretation="Strong governed source-target alignment."
        elif strength >= partial_threshold:
            insight_type="partial_alignment"; interpretation="Moderate governed source-target alignment."
        else:
            insight_type="partial_alignment"; interpretation="Weak but supported governed source-target alignment."
        rows.append({
            "content_id":r["content_id"],"comparison_type":r["comparison_type"],"insight_type":insight_type,
            "interpretation_scope":INTERPRETATION_SCOPE,"interpretation":interpretation,
            "source_role":r["source_role"],"source_node_id":r["source_node_id"],"source_category":r["source_category"],
            "source_concept":r["source_concept"],"target_role":r["best_target_role"] if matched else "",
            "target_node_id":r["best_target_node_id"] if matched else "","target_category":r["best_target_category"] if matched else "",
            "target_concept":r["best_target_concept"] if matched else "","relationship_type":r["best_relationship_type"] if matched else "",
            "relationship_source":r["best_relationship_source"] if matched else "","relationship_strength":strength if matched else 0.0,
            "pair_confidence":safe_float(r["best_pair_confidence"],0.0) if matched else 0.0,
            "audience_support_rate":safe_float(r["best_target_audience_support_rate"],np.nan),
            "audience_support_factor":safe_float(r["best_audience_support_factor"],np.nan),
            "evidence_strength":safe_float(r["source_alignment_contribution"],0.0),
            "source_has_supported_match":matched,
        })
    return pd.DataFrame(rows)

def classify_audience_signal(category):
    return AUDIENCE_CATEGORY_TO_SIGNAL.get(key(category), "other_audience_signal")

def build_unmatched_audience_signals(pairs, availability):
    rows=[]
    ready=availability[availability["alignment_computable"] & availability["comparison_type"].isin(AUDIENCE_COMPARISONS)]
    for _,a in ready.iterrows():
        cid,comp=a["content_id"],a["comparison_type"]
        sub=pairs[(pairs["content_id"]==cid)&(pairs["comparison_type"]==comp)].copy()
        if sub.empty: continue
        target_cols=[c for c in ["target_role","target_node_id","target_category","target_concept",
                    "target_audience_support_rate","target_avg_confidence","target_annotation_count","target_unique_comment_count"] if c in sub.columns]
        targets=sub[target_cols].drop_duplicates("target_node_id").reset_index(drop=True)
        supported=sub[sub["relationship_supported"] & sub["relationship_type"].map(key).isin(SUPPORTED_TYPES)]
        supported_targets=set(supported["target_node_id"])
        for _,t in targets.iterrows():
            tid=t["target_node_id"]
            if tid in supported_targets: continue
            cat=txt(t["target_category"]); signal_type=classify_audience_signal(cat)
            aud=safe_float(t.get("target_audience_support_rate"),0.0); conf=safe_float(t.get("target_avg_confidence"),0.0)
            ann=safe_float(t.get("target_annotation_count"),0.0); uniq=safe_float(t.get("target_unique_comment_count"),0.0)
            rows.append({
                "content_id":cid,"comparison_type":comp,"signal_type":signal_type,
                "interpretation_scope":INTERPRETATION_SCOPE,"source_role":a["source_role"],
                "target_role":t.get("target_role","audience_perception"),"target_node_id":tid,
                "target_category":cat,"target_concept":txt(t["target_concept"]),"audience_support_rate":aud,
                "audience_support_factor":math.sqrt(max(aud,0.0)),"target_avg_confidence":conf,
                "target_annotation_count":ann,"target_unique_comment_count":uniq,"evidence_strength":aud*max(conf,0.0),
                "has_supported_incoming_relationship":False,
            })
    out=pd.DataFrame(rows)
    if out.empty: return out
    out=out.sort_values(["content_id","comparison_type","signal_type","audience_support_rate","target_avg_confidence","target_unique_comment_count"],
                        ascending=[True,True,True,False,False,False]).reset_index(drop=True)
    out["rank_within_signal_type"]=out.groupby(["content_id","comparison_type","signal_type"]).cumcount()+1
    return out

def join_top(values,n=5):
    vals=[txt(v) for v in values if txt(v)]
    return " | ".join(vals[:n])

def build_content_diagnostic(insights,audience_signals,comparison_scores,availability):
    rows=[]
    lookup={(r["content_id"],r["comparison_type"]):r for _,r in comparison_scores.iterrows()}
    sigtypes=["unexpected_perception","emergent_response","audience_action_signal","audience_discussion_signal","community_signal","other_audience_signal"]
    for _,a in availability.iterrows():
        cid,comp=a["content_id"],a["comparison_type"]
        row={"content_id":cid,"comparison_type":comp,"source_role":a["source_role"],"target_role":a["target_role"],
             "alignment_computable":bool(a["alignment_computable"]),"alignment_status":a["alignment_status"],
             "interpretation_scope":INTERPRETATION_SCOPE}
        cs=lookup.get((cid,comp))
        for col in ["alignment_score","source_concept_coverage","target_concept_coverage","weighted_relationship_strength","weighted_pair_confidence","weighted_audience_support_factor"]:
            row[col]=safe_float(cs[col]) if cs is not None else np.nan
        sub=insights[(insights["content_id"]==cid)&(insights["comparison_type"]==comp)]
        aligned=sub[sub["insight_type"]=="aligned_concept"]; partial=sub[sub["insight_type"]=="partial_alignment"]; missing=sub[sub["insight_type"]=="missing_resonance"]
        row["aligned_concept_count"]=len(aligned); row["partial_alignment_count"]=len(partial); row["missing_resonance_count"]=len(missing)
        row["top_aligned_pairs"]=join_top([f'{r["source_concept"]} -> {r["target_concept"]}' for _,r in aligned.sort_values(["evidence_strength","relationship_strength"],ascending=[False,False]).iterrows()])
        row["top_partial_pairs"]=join_top([f'{r["source_concept"]} -> {r["target_concept"]}' for _,r in partial.sort_values(["evidence_strength","relationship_strength"],ascending=[False,False]).iterrows()])
        row["top_missing_concepts"]=join_top(missing["source_concept"].tolist())
        aud=audience_signals[(audience_signals["content_id"]==cid)&(audience_signals["comparison_type"]==comp)] if not audience_signals.empty else pd.DataFrame()
        for st in sigtypes:
            ss=aud[aud["signal_type"]==st] if not aud.empty else pd.DataFrame()
            row[f"{st}_count"]=len(ss)
            row[f"top_{st}s"]=join_top(ss.sort_values(["audience_support_rate","target_avg_confidence"],ascending=[False,False])["target_concept"].tolist()) if not ss.empty else ""
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    args=parse_args()
    if args.partial_threshold > args.aligned_threshold: raise ValueError("--partial-threshold must be <= --aligned-threshold")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    source_scores=load_source_scores(args.source_scores); comparison_scores=load_comparison_scores(args.comparison_scores)
    pairs=load_pairs(args.pairs); availability=load_availability(args.availability)
    insights=classify_source_insights(source_scores,args.aligned_threshold,args.partial_threshold)
    audience_signals=build_unmatched_audience_signals(pairs,availability)
    diagnostic=build_content_diagnostic(insights,audience_signals,comparison_scores,availability)
    insights.to_csv(args.output_dir/"alignment_insights_5d_v2.csv",index=False)
    audience_signals.to_csv(args.output_dir/"audience_unmatched_signals_5d_v2.csv",index=False)
    diagnostic.to_csv(args.output_dir/"content_alignment_diagnostic_5d_v2.csv",index=False)
    signal_counts=audience_signals["signal_type"].value_counts() if not audience_signals.empty else pd.Series(dtype=int)
    rows=[
        {"metric":"content_count","value":availability["content_id"].nunique()},
        {"metric":"comparison_rows","value":len(availability)},
        {"metric":"comparison_rows_computable","value":int(availability["alignment_computable"].sum())},
        {"metric":"comparison_rows_insufficient_signal","value":int((~availability["alignment_computable"]).sum())},
        {"metric":"source_insight_rows","value":len(insights)},
        {"metric":"aligned_concept_rows","value":int((insights["insight_type"]=="aligned_concept").sum())},
        {"metric":"partial_alignment_rows","value":int((insights["insight_type"]=="partial_alignment").sum())},
        {"metric":"missing_resonance_rows","value":int((insights["insight_type"]=="missing_resonance").sum())},
        {"metric":"audience_unmatched_signal_rows","value":len(audience_signals)},
        {"metric":"aligned_threshold","value":args.aligned_threshold},
        {"metric":"partial_threshold","value":args.partial_threshold},
    ]
    for st,c in signal_counts.items(): rows.append({"metric":f"{st}_rows","value":int(c)})
    summary=pd.DataFrame(rows); summary.to_csv(args.output_dir/"build_summary_5d_v2.csv",index=False)
    print("\n=== 5D.1 v2 Deterministic Alignment Diagnostics complete ==="); print(summary.to_string(index=False))
    print("\nSource insight counts:"); print(insights["insight_type"].value_counts().to_string())
    if not audience_signals.empty:
        print("\nAudience unmatched signal counts:"); print(audience_signals["signal_type"].value_counts().to_string())
        print("\nAudience unmatched signals by comparison:"); print(audience_signals.groupby(["comparison_type","signal_type"]).size().to_string())
    print("\nPer-comparison diagnostic summary:")
    agg=(diagnostic.groupby("comparison_type").agg(rows=("content_id","size"),computable=("alignment_computable","sum"),mean_alignment_score=("alignment_score","mean"),aligned=("aligned_concept_count","sum"),partial=("partial_alignment_count","sum"),missing=("missing_resonance_count","sum"),unexpected=("unexpected_perception_count","sum"),emergent_response=("emergent_response_count","sum"),action_signal=("audience_action_signal_count","sum"),discussion_signal=("audience_discussion_signal_count","sum"),community_signal=("community_signal_count","sum")).round(4))
    print(agg.to_string())
    focus=diagnostic[diagnostic["content_id"].isin(["youtube_gaw2OJ5PoDc","youtube_w0wF-O0EGsQ"])]
    if not focus.empty:
        print("\nFocus-case diagnostics:")
        cols=["content_id","comparison_type","alignment_status","alignment_score","top_aligned_pairs","top_partial_pairs","top_missing_concepts","top_unexpected_perceptions","top_emergent_responses","top_audience_action_signals","top_audience_discussion_signals","top_community_signals"]
        cols=[c for c in cols if c in focus.columns]
        print(focus[cols].to_string(index=False))

if __name__=="__main__": main()
