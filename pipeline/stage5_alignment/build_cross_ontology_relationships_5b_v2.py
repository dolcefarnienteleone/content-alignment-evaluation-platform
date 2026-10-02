"""
Build cross-ontology relationships v0.2
--------------------------------------

Example
-------
python build_cross_ontology_relationships_5b_v2.py \
  --normalized-annotations data/alignment/5a/normalized_annotations_5a.csv \
  --concept-profile data/alignment/5a/content_concept_profile_5a.csv \
  --role-profile data/alignment/5a/content_role_profile_5a.csv \
  --ontology-nodes data/ontology/v0.2/ontology_v02_nodes.csv \
  --relationship-rules data/alignment/5b/rules_v0.1/relationship_rules_v0.1.csv \
  --output-dir data/alignment/5b_v0.1
"""

#!/usr/bin/env python3
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

COMPARISONS=[
    ("intent_to_content","inferred_content_intent","ai_content_understanding"),
    ("content_to_audience","ai_content_understanding","audience_perception"),
    ("intent_to_audience","inferred_content_intent","audience_perception"),
]
CURATED_RELATION_TYPES={"equivalent_to","related_to","manifests_as","contributes_to","elicits"}
RULE_COLUMNS=[
    "source_concept","source_category","source_role",
    "target_concept","target_category","target_role",
    "relationship_type","relationship_strength","directional",
    "rule_status","rule_source","notes",
]

def txt(x):
    if x is None:return ""
    try:
        if pd.isna(x):return ""
    except Exception: pass
    return re.sub(r"\s+"," ",str(x)).strip()
def key(x): return txt(x).casefold()
def safe_float(x,default=np.nan):
    try:
        if pd.isna(x): return default
        return float(x)
    except Exception:return default
def boolish(x):
    if isinstance(x,bool): return x
    return key(x) in {"1","true","yes","y","t"}
def clipped(x): return float(max(0,min(1,x)))

def relation_priority(t):
    return {
        "exact":100,"equivalent_to":90,"manifests_as":80,
        "contributes_to":80,"elicits":80,"related_to":70,
        "semantic_strong":20,"semantic":10,"none":0
    }.get(key(t),0)

def load_concept_profile(path):
    df=pd.read_csv(path)
    req=["content_id","annotation_role","canonical_category","canonical_concept","ontology_node_id"]
    miss=[c for c in req if c not in df.columns]
    if miss: raise ValueError(f"Concept profile missing {miss}. Columns={list(df.columns)}")
    for c in req: df[c]=df[c].map(txt)
    for c in ["annotation_count","unique_comment_count","avg_confidence","max_confidence",
              "audience_mapped_signal_support_rate","concept_rank_within_role"]:
        if c not in df.columns: df[c]=np.nan
    return df

def load_role_profile(path):
    df=pd.read_csv(path)
    for c in ["content_id","annotation_role"]:
        if c not in df.columns: raise ValueError(f"Role profile missing {c}")
        df[c]=df[c].map(txt)
    return df

def load_normalized_annotations(path):
    df=pd.read_csv(path)
    for c in ["content_id","annotation_role","normalization_status"]:
        if c not in df.columns: raise ValueError(f"Normalized annotations missing {c}")
    df["content_id"]=df["content_id"].map(txt)
    df["annotation_role"]=df["annotation_role"].map(txt)
    return df

def load_rules(path):
    df=pd.read_csv(path)
    miss=[c for c in RULE_COLUMNS if c not in df.columns]
    if miss: raise ValueError(f"Relationship rules missing {miss}")
    df["relationship_strength"]=df["relationship_strength"].map(lambda x:clipped(safe_float(x,0.0)))
    df["directional"]=df["directional"].map(boolish)
    df["rule_status"]=df["rule_status"].map(key)

    allowed_status={"approved","rejected","deferred"}
    invalid_status=sorted(set(df["rule_status"])-allowed_status)
    if invalid_status:
        raise ValueError(f"Unsupported rule_status values: {invalid_status}")

    active=df[df["rule_status"]=="approved"].copy()
    invalid=sorted(set(active["relationship_type"].map(key))-CURATED_RELATION_TYPES)
    if invalid: raise ValueError(f"Unsupported curated relationship types: {invalid}")
    return df

def rule_matches(r,sr,sc,sg,tr,tc,tg):
    def eq_or_blank(rv,a):
        rr=key(rv)
        return (not rr) or rr==key(a)
    forward=(eq_or_blank(r["source_role"],sr) and key(r["source_concept"])==key(sc)
             and eq_or_blank(r["source_category"],sg)
             and eq_or_blank(r["target_role"],tr) and key(r["target_concept"])==key(tc)
             and eq_or_blank(r["target_category"],tg))
    if forward:return True
    if not bool(r["directional"]):
        return (eq_or_blank(r["source_role"],tr) and key(r["source_concept"])==key(tc)
                and eq_or_blank(r["source_category"],tg)
                and eq_or_blank(r["target_role"],sr) and key(r["target_concept"])==key(sc)
                and eq_or_blank(r["target_category"],sg))
    return False

def find_governed_rule(rules,sr,sc,sg,tr,tc,tg):
    hits=[r for _,r in rules.iterrows() if rule_matches(r,sr,sc,sg,tr,tc,tg)]
    if not hits:
        return None

    status_priority={"approved":3,"rejected":2,"deferred":1}
    hits=sorted(
        hits,
        key=lambda r:(
            status_priority.get(key(r["rule_status"]),0),
            relation_priority(r["relationship_type"]),
            safe_float(r["relationship_strength"],0)
        ),
        reverse=True
    )
    return hits[0]

def compute_embeddings(concepts,model_name):
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer(model_name)
    u=concepts[["canonical_category","canonical_concept"]].drop_duplicates().reset_index(drop=True)
    texts=[f'{r["canonical_category"]}: {r["canonical_concept"]}' for _,r in u.iterrows()]
    vecs=model.encode(texts,normalize_embeddings=True,show_progress_bar=True)
    return {(key(r["canonical_category"]),key(r["canonical_concept"])):np.asarray(vecs[i],dtype=float) for i,r in u.iterrows()}

def cosine(lookup,ca,a,cb,b):
    va=lookup.get((key(ca),key(a))); vb=lookup.get((key(cb),key(b)))
    if va is None or vb is None:return np.nan
    return clipped(float(np.dot(va,vb)))

def build_availability(normalized,roles):
    cids=sorted(set(normalized["content_id"])|set(roles["content_id"]))
    available=set(zip(roles["content_id"],roles["annotation_role"]))
    rows=[]
    for cid in cids:
        for comp,sr,tr in COMPARISONS:
            sa=(cid,sr) in available; ta=(cid,tr) in available
            rows.append({
                "content_id":cid,"comparison_type":comp,"source_role":sr,"target_role":tr,
                "source_profile_available":sa,"target_profile_available":ta,
                "alignment_computable":sa and ta,
                "alignment_status":"ready" if sa and ta else "insufficient_normalized_signal",
            })
    return pd.DataFrame(rows)

def pair_relationship(s,t,rules,emb,sem_thr,strong_thr):
    sc,sg,sr=txt(s["canonical_concept"]),txt(s["canonical_category"]),txt(s["annotation_role"])
    tc,tg,tr=txt(t["canonical_concept"]),txt(t["canonical_category"]),txt(t["annotation_role"])
    exact=key(sc)==key(tc)
    sem=cosine(emb,sg,sc,tg,tc)
    governed=find_governed_rule(rules,sr,sc,sg,tr,tc,tg)
    if exact:
        rt,st,src,sup="exact",1.0,"exact_concept_identity",True
    elif governed is not None and key(governed["rule_status"])=="approved":
        rt=txt(governed["relationship_type"])
        st=clipped(safe_float(governed["relationship_strength"],0.8))
        src=txt(governed["rule_source"]) or "curated_rule"
        sup=True
    elif governed is not None and key(governed["rule_status"])=="rejected":
        rt,st,src,sup="human_rejected",0.0,"5b1_human_review_v0.1",False
    elif governed is not None and key(governed["rule_status"])=="deferred":
        rt,st,src,sup="human_deferred",0.0,"5b1_human_review_v0.1",False
    elif not pd.isna(sem) and sem>=strong_thr:
        rt,st,src,sup="semantic_strong",sem,"semantic_model",True
    elif not pd.isna(sem) and sem>=sem_thr:
        rt,st,src,sup="semantic",sem,"semantic_model",True
    else:
        rt,st,src,sup="none",0.0,"no_supported_relation",False
    return {
        "exact_concept_match":exact,"semantic_similarity":sem,
        "relationship_type":rt,"relationship_strength":st,
        "relationship_source":src,"relationship_supported":sup,
        "source_avg_confidence":safe_float(s.get("avg_confidence")),
        "target_avg_confidence":safe_float(t.get("avg_confidence")),
        "source_audience_support_rate":safe_float(s.get("audience_mapped_signal_support_rate")),
        "target_audience_support_rate":safe_float(t.get("audience_mapped_signal_support_rate")),
    }

def build_pairs(cp,av,rules,emb,sem_thr,strong_thr):
    rows=[]
    for _,a in av[av["alignment_computable"]].iterrows():
        cid,comp,sr,tr=a["content_id"],a["comparison_type"],a["source_role"],a["target_role"]
        sdf=cp[(cp["content_id"]==cid)&(cp["annotation_role"]==sr)]
        tdf=cp[(cp["content_id"]==cid)&(cp["annotation_role"]==tr)]
        for _,s in sdf.iterrows():
            for _,t in tdf.iterrows():
                rel=pair_relationship(s,t,rules,emb,sem_thr,strong_thr)
                rows.append({
                    "content_id":cid,"comparison_type":comp,
                    "source_role":sr,"source_node_id":txt(s["ontology_node_id"]),
                    "source_category":txt(s["canonical_category"]),"source_concept":txt(s["canonical_concept"]),
                    "source_annotation_count":safe_float(s.get("annotation_count"),0),
                    "source_unique_comment_count":safe_float(s.get("unique_comment_count"),0),
                    "target_role":tr,"target_node_id":txt(t["ontology_node_id"]),
                    "target_category":txt(t["canonical_category"]),"target_concept":txt(t["canonical_concept"]),
                    "target_annotation_count":safe_float(t.get("annotation_count"),0),
                    "target_unique_comment_count":safe_float(t.get("unique_comment_count"),0),
                    **rel
                })
    out=pd.DataFrame(rows)
    if out.empty:return out
    out["mean_pair_confidence"]=out[["source_avg_confidence","target_avg_confidence"]].mean(axis=1,skipna=True)
    out["_ts"]=out["target_audience_support_rate"].fillna(-1)
    out=out.sort_values(["content_id","comparison_type","source_node_id","relationship_supported","relationship_strength","_ts","mean_pair_confidence"],
                        ascending=[True,True,True,False,False,False,False])
    out["pair_rank_for_source"]=out.groupby(["content_id","comparison_type","source_node_id"]).cumcount()+1
    return out.drop(columns="_ts")

def build_summary(pairs,av):
    rows=[]
    for _,a in av.iterrows():
        cid,comp=a["content_id"],a["comparison_type"]
        if not a["alignment_computable"]:
            rows.append({
                "content_id":cid,"comparison_type":comp,"source_role":a["source_role"],"target_role":a["target_role"],
                "alignment_computable":False,"alignment_status":a["alignment_status"],
                "source_concept_count":0,"target_concept_count":0,"pair_count":0,"supported_pair_count":0,
                "exact_pair_count":0,"curated_pair_count":0,"semantic_pair_count":0,"unrelated_pair_count":0,
                "source_concepts_with_supported_match":0,"source_concept_match_coverage":np.nan,
                "target_concepts_with_supported_match":0,"target_concept_match_coverage":np.nan,
                "deferred_pair_count":0,
            }); continue
        sub=pairs[(pairs["content_id"]==cid)&(pairs["comparison_type"]==comp)]
        sup=sub[sub["relationship_supported"]]
        sc=sub["source_node_id"].nunique(); tc=sub["target_node_id"].nunique()
        sm=sup["source_node_id"].nunique(); tm=sup["target_node_id"].nunique()
        rows.append({
            "content_id":cid,"comparison_type":comp,"source_role":a["source_role"],"target_role":a["target_role"],
            "alignment_computable":True,"alignment_status":"ready",
            "source_concept_count":sc,"target_concept_count":tc,"pair_count":len(sub),
            "supported_pair_count":len(sup),
            "exact_pair_count":int((sub["relationship_type"]=="exact").sum()),
            "curated_pair_count":int(sub["relationship_type"].isin(CURATED_RELATION_TYPES).sum()),
            "semantic_pair_count":int(sub["relationship_type"].isin(["semantic","semantic_strong"]).sum()),
            "unrelated_pair_count":int(sub["relationship_type"].isin(["none","human_rejected"]).sum()),
            "deferred_pair_count":int((sub["relationship_type"]=="human_deferred").sum()),
            "source_concepts_with_supported_match":sm,
            "source_concept_match_coverage":sm/sc if sc else np.nan,
            "target_concepts_with_supported_match":tm,
            "target_concept_match_coverage":tm/tc if tc else np.nan,
        })
    return pd.DataFrame(rows)

def build_catalog(pairs):
    if pairs.empty:return pd.DataFrame()
    return (pairs.groupby([
        "source_role","source_category","source_concept","source_node_id",
        "target_role","target_category","target_concept","target_node_id",
        "relationship_type","relationship_source","relationship_supported"],dropna=False)
        .agg(content_support_count=("content_id","nunique"),pair_occurrence_count=("content_id","size"),
             avg_semantic_similarity=("semantic_similarity","mean"),
             avg_relationship_strength=("relationship_strength","mean"),
             avg_pair_confidence=("mean_pair_confidence","mean"))
        .reset_index()
        .sort_values(["relationship_supported","content_support_count","avg_relationship_strength"],ascending=[False,False,False])
        .reset_index(drop=True))

def build_review_queue(catalog,strong_thr):
    if catalog.empty:return pd.DataFrame()
    q=catalog[catalog["relationship_type"].isin(["semantic","semantic_strong"])].copy()
    q["review_priority"]=np.select([
        (q["content_support_count"]>=2)&(q["avg_semantic_similarity"]>=strong_thr),
        q["content_support_count"]>=2,
        q["avg_semantic_similarity"]>=strong_thr
    ],["high","medium","medium"],default="low")
    q["human_relationship_decision"]=""
    q["human_relationship_type"]=""
    q["human_relationship_strength"]=np.nan
    q["human_review_notes"]=""
    q["human_review_status"]="pending"
    return q

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--normalized-annotations",type=Path,required=True)
    p.add_argument("--concept-profile",type=Path,required=True)
    p.add_argument("--role-profile",type=Path,required=True)
    p.add_argument("--ontology-nodes",type=Path,required=True)  # retained for CLI compatibility/audit
    p.add_argument("--relationship-rules",type=Path,required=True)
    p.add_argument("--model",default="sentence-transformers/distiluse-base-multilingual-cased-v1")
    p.add_argument("--semantic-threshold",type=float,default=0.60)
    p.add_argument("--strong-threshold",type=float,default=0.72)
    p.add_argument("--output-dir",type=Path,required=True)
    return p.parse_args()

def main():
    args=parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    normalized=load_normalized_annotations(args.normalized_annotations)
    cp=load_concept_profile(args.concept_profile)
    roles=load_role_profile(args.role_profile)
    rules=load_rules(args.relationship_rules)
    observed=cp[["canonical_category","canonical_concept"]].drop_duplicates()
    emb=compute_embeddings(observed,args.model)
    av=build_availability(normalized,roles)
    pairs=build_pairs(cp,av,rules,emb,args.semantic_threshold,args.strong_threshold)
    summary=build_summary(pairs,av)
    catalog=build_catalog(pairs)
    review=build_review_queue(catalog,args.strong_threshold)

    av.to_csv(args.output_dir/"content_comparison_availability_5b.csv",index=False)
    pairs.to_csv(args.output_dir/"concept_pair_candidates_5b.csv",index=False)
    summary.to_csv(args.output_dir/"content_pair_summary_5b.csv",index=False)
    catalog.to_csv(args.output_dir/"concept_relationship_catalog_5b.csv",index=False)
    review.to_csv(args.output_dir/"relationship_review_queue_5b.csv",index=False)

    bs=pd.DataFrame([
        {"metric":"content_count","value":normalized["content_id"].nunique()},
        {"metric":"observed_canonical_concepts","value":observed["canonical_concept"].nunique()},
        {"metric":"comparison_rows_expected","value":len(av)},
        {"metric":"comparison_rows_computable","value":int(av["alignment_computable"].sum())},
        {"metric":"comparison_rows_insufficient_signal","value":int((~av["alignment_computable"]).sum())},
        {"metric":"concept_pair_rows","value":len(pairs)},
        {"metric":"supported_pair_rows","value":int(pairs["relationship_supported"].sum()) if not pairs.empty else 0},
        {"metric":"exact_pair_rows","value":int((pairs["relationship_type"]=="exact").sum()) if not pairs.empty else 0},
        {"metric":"curated_pair_rows","value":int(pairs["relationship_type"].isin(CURATED_RELATION_TYPES).sum()) if not pairs.empty else 0},
        {"metric":"semantic_pair_rows","value":int(pairs["relationship_type"].isin(["semantic","semantic_strong"]).sum()) if not pairs.empty else 0},
        {"metric":"relationship_catalog_rows","value":len(catalog)},
        {"metric":"relationship_review_queue_rows","value":len(review)},
        {"metric":"governed_rule_count","value":len(rules)},
        {"metric":"approved_curated_rule_count","value":int((rules["rule_status"]=="approved").sum())},
        {"metric":"rejected_rule_count","value":int((rules["rule_status"]=="rejected").sum())},
        {"metric":"deferred_rule_count","value":int((rules["rule_status"]=="deferred").sum())},
        {"metric":"semantic_threshold","value":args.semantic_threshold},
        {"metric":"strong_threshold","value":args.strong_threshold},
        {"metric":"semantic_model","value":args.model},
        {"metric":"relationship_rules_used","value":str(args.relationship_rules)},
    ])
    bs.to_csv(args.output_dir/"build_summary_5b.csv",index=False)

    print("\n=== 5B v2 relationship & concept pairing build complete ===")
    print(bs.to_string(index=False))
    print("\nComparison availability:")
    print(av.groupby(["comparison_type","alignment_status"]).size().to_string())
    if not pairs.empty:
        print("\nRelationship types:")
        print(pairs["relationship_type"].value_counts().to_string())
        print("\nSupported relationship source:")
        print(pairs.loc[pairs["relationship_supported"],"relationship_source"].value_counts().to_string())
    print("\nNOTE: semantic / semantic_strong rows remaining after rerun are still-unreviewed semantic candidates, not human-governed truth.")

if __name__=="__main__":
    main()
