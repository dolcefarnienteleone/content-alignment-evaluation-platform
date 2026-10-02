#!/usr/bin/env python3
"""
Deliverable 5D.2 v2 — Deterministic Decision Insight Layer (Semantic Cleanup)

This script REPLACES the original 5D.2 script.
You do NOT need to run the original 5D.2 first.

Inputs come directly from 5D.1 v2:
- content_alignment_diagnostic_5d_v2.csv
- alignment_insights_5d_v2.csv
- audience_unmatched_signals_5d_v2.csv

Main semantic cleanup
---------------------
1. strongest_alignment is split into:
   - content_realization
   - audience_carrythrough

2. alignment_gap is split into:
   - realization_gap
   - audience_carrythrough_gap
   - end_to_end_gap

3. emerging_opportunity is renamed to:
   - emergent_audience_signal

4. Keep:
   - unexpected_perception
   - audience_demand
   - audience_discussion
   - community_signal
   - potential_risk_candidate

5. No overall score is created.

Example
--------
python build_decision_insights_5d2_v2.py \
  --diagnostic data/alignment/5d_v2/content_alignment_diagnostic_5d_v2.csv \
  --source-insights data/alignment/5d_v2/alignment_insights_5d_v2.csv \
  --audience-signals data/alignment/5d_v2/audience_unmatched_signals_5d_v2.csv \
  --output-dir data/alignment/5d2_v2

"""

from __future__ import annotations
import argparse, re
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

INTERPRETATION_SCOPE = (
    "Governed ontology/profile interpretation only; absence of a supported counterpart "
    "does not prove absence of human perception."
)

RISK_TERMS = {
    "criticism", "negative", "disappointment", "disappointed", "sad",
    "concern", "concerned", "frustration", "frustrated", "confusion",
    "confused", "complaint", "issue", "problem", "worry", "worried",
    "anger", "angry", "uncomfortable", "distress", "distressed",
}

EMERGENT_SIGNAL_TYPES = {"unexpected_perception", "emergent_response"}


def txt(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(x)).strip()


def key(x: Any) -> str:
    return txt(x).casefold()


def safe_float(x: Any, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def boolish(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return key(x) in {"true", "1", "yes", "y", "t"}


def require_columns(df, required, name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def parse_args():
    p = argparse.ArgumentParser(description="5D.2 v2 deterministic decision insight layer")
    p.add_argument("--diagnostic", type=Path, required=True)
    p.add_argument("--source-insights", type=Path, required=True)
    p.add_argument("--audience-signals", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--emergent-min-support", type=float, default=0.05)
    p.add_argument("--risk-min-support", type=float, default=0.05)
    p.add_argument("--top-k", type=int, default=3)
    return p.parse_args()


def load_diagnostic(path):
    df = pd.read_csv(path)
    require_columns(df, [
        "content_id","comparison_type","alignment_computable","alignment_status",
        "alignment_score","source_concept_coverage","weighted_relationship_strength"
    ], "diagnostic")
    for c in ["content_id","comparison_type","alignment_status"]:
        df[c] = df[c].map(txt)
    df["alignment_computable"] = df["alignment_computable"].map(boolish)
    return df


def load_source_insights(path):
    df = pd.read_csv(path)
    require_columns(df, [
        "content_id","comparison_type","insight_type","source_role","source_category",
        "source_concept","target_role","target_category","target_concept",
        "relationship_type","relationship_strength","pair_confidence",
        "audience_support_rate","evidence_strength"
    ], "source_insights")
    for c in [
        "content_id","comparison_type","insight_type","source_role","source_category",
        "source_concept","target_role","target_category","target_concept",
        "relationship_type","relationship_source"
    ]:
        if c in df.columns:
            df[c] = df[c].map(txt)
    return df


def load_audience_signals(path):
    df = pd.read_csv(path)
    require_columns(df, [
        "content_id","comparison_type","signal_type","target_category",
        "target_concept","audience_support_rate","target_avg_confidence","evidence_strength"
    ], "audience_signals")
    for c in [
        "content_id","comparison_type","signal_type","source_role","target_role",
        "target_node_id","target_category","target_concept"
    ]:
        if c in df.columns:
            df[c] = df[c].map(txt)
    return df


def risk_candidate(concept):
    words = set(re.findall(r"[a-z]+", key(concept)))
    return bool(words & RISK_TERMS)


def rank_source(df):
    if df.empty:
        return df
    out = df.copy()
    out["_relationship_strength"] = out["relationship_strength"].map(lambda x: safe_float(x, 0.0))
    out["_evidence_strength"] = out["evidence_strength"].map(lambda x: safe_float(x, 0.0))
    out["_pair_confidence"] = out["pair_confidence"].map(lambda x: safe_float(x, 0.0))
    return out.sort_values(
        ["_evidence_strength","_relationship_strength","_pair_confidence"],
        ascending=[False,False,False]
    )


def rank_audience(df):
    if df.empty:
        return df
    out = df.copy()
    out["_support"] = out["audience_support_rate"].map(lambda x: safe_float(x, 0.0))
    out["_evidence"] = out["evidence_strength"].map(lambda x: safe_float(x, 0.0))
    out["_confidence"] = out["target_avg_confidence"].map(lambda x: safe_float(x, 0.0))
    return out.sort_values(["_support","_evidence","_confidence"], ascending=[False,False,False])


def unique_concept_list(df, concept_col, top_k):
    vals = []
    for v in df[concept_col].tolist() if not df.empty else []:
        t = txt(v)
        if t and t not in vals:
            vals.append(t)
        if len(vals) >= top_k:
            break
    return " | ".join(vals)


def unique_pair_list(df, top_k):
    vals = []
    for _, r in df.iterrows():
        s, t = txt(r.get("source_concept")), txt(r.get("target_concept"))
        if not s:
            continue
        pair = f"{s} -> {t}" if t else s
        if pair not in vals:
            vals.append(pair)
        if len(vals) >= top_k:
            break
    return " | ".join(vals)


def preferred_audience_view(audience, signal_type):
    sub = audience[audience["signal_type"] == signal_type].copy()
    if sub.empty:
        return sub
    ca = sub[sub["comparison_type"] == "content_to_audience"]
    sub = ca if not ca.empty else sub[sub["comparison_type"] == "intent_to_audience"]
    sub = rank_audience(sub)
    dedupe_cols = ["target_node_id"] if "target_node_id" in sub.columns else ["target_concept"]
    return sub.drop_duplicates(dedupe_cols)


def choose_key_finding_type(content_id, diagnostic, source):
    d = diagnostic[diagnostic["content_id"] == content_id]
    if len(d[~d["alignment_computable"]]) >= 2:
        return "limited_by_normalized_signal"

    def score(comp):
        row = d[d["comparison_type"] == comp]
        return np.nan if row.empty else safe_float(row.iloc[0]["alignment_score"])

    ic, ca, ia = score("intent_to_content"), score("content_to_audience"), score("intent_to_audience")

    if not np.isnan(ic) and ic >= 0.50 and not np.isnan(ca) and ca <= 0.10 and not np.isnan(ia) and ia <= 0.10:
        return "strong_realization_limited_audience_carrythrough"
    if not np.isnan(ia) and ia > 0:
        return "direct_intent_audience_carrythrough_detected"

    aligned = source[(source["content_id"] == content_id) & (source["insight_type"] == "aligned_concept")]
    return "partial_governed_alignment_detected" if not aligned.empty else "no_supported_alignment_detected"


def make_evidence_row(content_id, insight_field, rank, row, evidence_source):
    return {
        "content_id": content_id,
        "insight_field": insight_field,
        "rank": rank,
        "evidence_source": evidence_source,
        "comparison_type": txt(row.get("comparison_type")),
        "signal_or_insight_type": txt(row.get("signal_type", row.get("insight_type", ""))),
        "source_role": txt(row.get("source_role")),
        "source_category": txt(row.get("source_category")),
        "source_concept": txt(row.get("source_concept")),
        "target_role": txt(row.get("target_role")),
        "target_category": txt(row.get("target_category")),
        "target_concept": txt(row.get("target_concept")),
        "relationship_type": txt(row.get("relationship_type")),
        "relationship_strength": safe_float(row.get("relationship_strength")),
        "pair_confidence": safe_float(row.get("pair_confidence", row.get("target_avg_confidence"))),
        "audience_support_rate": safe_float(row.get("audience_support_rate")),
        "evidence_strength": safe_float(row.get("evidence_strength")),
        "interpretation_scope": INTERPRETATION_SCOPE,
    }


def add_evidence_rows(store, cid, field, df, top_k, source_name):
    for rank, (_, r) in enumerate(df.head(top_k).iterrows(), 1):
        store.append(make_evidence_row(cid, field, rank, r, source_name))


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    diagnostic = load_diagnostic(args.diagnostic)
    source = load_source_insights(args.source_insights)
    audience = load_audience_signals(args.audience_signals)

    content_ids = sorted(set(diagnostic["content_id"]) | set(source["content_id"]) | set(audience["content_id"]))
    decision_rows, evidence_rows = [], []

    for cid in content_ids:
        d = diagnostic[diagnostic["content_id"] == cid].copy()
        s = source[source["content_id"] == cid].copy()
        a = audience[audience["content_id"] == cid].copy()

        realization = rank_source(s[(s["comparison_type"]=="intent_to_content") & s["insight_type"].isin(["aligned_concept","partial_alignment"])])
        carry = rank_source(s[(s["comparison_type"]=="content_to_audience") & s["insight_type"].isin(["aligned_concept","partial_alignment"])])
        e2e = rank_source(s[(s["comparison_type"]=="intent_to_audience") & s["insight_type"].isin(["aligned_concept","partial_alignment"])])

        realization_gap_df = rank_source(s[(s["comparison_type"]=="intent_to_content") & (s["insight_type"]=="missing_resonance")])
        audience_gap_df = rank_source(s[(s["comparison_type"]=="content_to_audience") & (s["insight_type"]=="missing_resonance")])
        e2e_gap_df = rank_source(s[(s["comparison_type"]=="intent_to_audience") & (s["insight_type"]=="missing_resonance")])

        content_realization = unique_pair_list(realization, args.top_k)
        audience_carrythrough = unique_pair_list(carry, args.top_k)
        end_to_end_carrythrough = unique_pair_list(e2e, args.top_k)
        realization_gap = unique_concept_list(realization_gap_df, "source_concept", args.top_k)
        audience_carrythrough_gap = unique_concept_list(audience_gap_df, "source_concept", args.top_k)
        end_to_end_gap = unique_concept_list(e2e_gap_df, "source_concept", args.top_k)

        add_evidence_rows(evidence_rows, cid, "content_realization", realization, args.top_k, "source_insights")
        add_evidence_rows(evidence_rows, cid, "audience_carrythrough", carry, args.top_k, "source_insights")
        add_evidence_rows(evidence_rows, cid, "end_to_end_carrythrough", e2e, args.top_k, "source_insights")
        add_evidence_rows(evidence_rows, cid, "realization_gap", realization_gap_df, args.top_k, "source_insights")
        add_evidence_rows(evidence_rows, cid, "audience_carrythrough_gap", audience_gap_df, args.top_k, "source_insights")
        add_evidence_rows(evidence_rows, cid, "end_to_end_gap", e2e_gap_df, args.top_k, "source_insights")

        unexpected = preferred_audience_view(a, "unexpected_perception")
        unexpected_perception = unique_concept_list(unexpected, "target_concept", args.top_k)
        add_evidence_rows(evidence_rows, cid, "unexpected_perception", unexpected, args.top_k, "audience_signals")

        emergent_pool = a[a["signal_type"].isin(EMERGENT_SIGNAL_TYPES)].copy()
        if not emergent_pool.empty:
            ca = emergent_pool[emergent_pool["comparison_type"]=="content_to_audience"]
            emergent_pool = ca if not ca.empty else emergent_pool[emergent_pool["comparison_type"]=="intent_to_audience"]
            emergent_pool = emergent_pool[emergent_pool["audience_support_rate"].map(lambda x: safe_float(x,0.0)) >= args.emergent_min_support]
            emergent_pool = emergent_pool[~emergent_pool["target_concept"].map(risk_candidate)]
            emergent_pool = rank_audience(emergent_pool)
            dedupe_cols = ["target_node_id"] if "target_node_id" in emergent_pool.columns else ["target_concept"]
            emergent_pool = emergent_pool.drop_duplicates(dedupe_cols)

        emergent_audience_signal = unique_concept_list(emergent_pool, "target_concept", args.top_k) if not emergent_pool.empty else ""
        add_evidence_rows(evidence_rows, cid, "emergent_audience_signal", emergent_pool, args.top_k, "audience_signals")

        action = preferred_audience_view(a, "audience_action_signal")
        discussion = preferred_audience_view(a, "audience_discussion_signal")
        community = preferred_audience_view(a, "community_signal")

        audience_demand = unique_concept_list(action, "target_concept", args.top_k)
        audience_discussion = unique_concept_list(discussion, "target_concept", args.top_k)
        community_signal = unique_concept_list(community, "target_concept", args.top_k)

        add_evidence_rows(evidence_rows, cid, "audience_demand", action, args.top_k, "audience_signals")
        add_evidence_rows(evidence_rows, cid, "audience_discussion", discussion, args.top_k, "audience_signals")
        add_evidence_rows(evidence_rows, cid, "community_signal", community, args.top_k, "audience_signals")

        risk_pool = a[a["audience_support_rate"].map(lambda x: safe_float(x,0.0)) >= args.risk_min_support].copy()
        risk_pool = risk_pool[risk_pool["target_concept"].map(risk_candidate)]
        if not risk_pool.empty:
            ca = risk_pool[risk_pool["comparison_type"]=="content_to_audience"]
            risk_pool = ca if not ca.empty else risk_pool[risk_pool["comparison_type"]=="intent_to_audience"]
            risk_pool = rank_audience(risk_pool)
            dedupe_cols = ["target_node_id"] if "target_node_id" in risk_pool.columns else ["target_concept"]
            risk_pool = risk_pool.drop_duplicates(dedupe_cols)

        potential_risk_candidate = unique_concept_list(risk_pool, "target_concept", args.top_k) if not risk_pool.empty else ""
        add_evidence_rows(evidence_rows, cid, "potential_risk_candidate", risk_pool, args.top_k, "audience_signals")

        comp_metrics = {}
        for comp in ["intent_to_content","content_to_audience","intent_to_audience"]:
            row = d[d["comparison_type"]==comp]
            if row.empty:
                comp_metrics[f"{comp}_status"] = "missing"
                comp_metrics[f"{comp}_score"] = np.nan
                comp_metrics[f"{comp}_coverage"] = np.nan
                comp_metrics[f"{comp}_relationship_strength"] = np.nan
            else:
                rr = row.iloc[0]
                comp_metrics[f"{comp}_status"] = txt(rr["alignment_status"])
                comp_metrics[f"{comp}_score"] = safe_float(rr["alignment_score"])
                comp_metrics[f"{comp}_coverage"] = safe_float(rr["source_concept_coverage"])
                comp_metrics[f"{comp}_relationship_strength"] = safe_float(rr["weighted_relationship_strength"])

        decision_rows.append({
            "content_id": cid,
            "key_finding_type": choose_key_finding_type(cid, diagnostic, source),
            "content_realization": content_realization,
            "audience_carrythrough": audience_carrythrough,
            "end_to_end_carrythrough": end_to_end_carrythrough,
            "realization_gap": realization_gap,
            "audience_carrythrough_gap": audience_carrythrough_gap,
            "end_to_end_gap": end_to_end_gap,
            "unexpected_perception": unexpected_perception,
            "emergent_audience_signal": emergent_audience_signal,
            "audience_demand": audience_demand,
            "audience_discussion": audience_discussion,
            "community_signal": community_signal,
            "potential_risk_candidate": potential_risk_candidate,
            "potential_risk_requires_validation": bool(potential_risk_candidate),
            "interpretation_scope": INTERPRETATION_SCOPE,
            **comp_metrics
        })

    decisions = pd.DataFrame(decision_rows)
    evidence = pd.DataFrame(evidence_rows)

    decisions.to_csv(args.output_dir/"decision_insights_5d2_v2.csv", index=False)
    evidence.to_csv(args.output_dir/"decision_insight_evidence_5d2_v2.csv", index=False)

    summary = pd.DataFrame([
        {"metric":"content_count","value":len(decisions)},
        {"metric":"decision_rows","value":len(decisions)},
        {"metric":"evidence_rows","value":len(evidence)},
        {"metric":"contents_with_content_realization","value":int(decisions["content_realization"].map(bool).sum())},
        {"metric":"contents_with_audience_carrythrough","value":int(decisions["audience_carrythrough"].map(bool).sum())},
        {"metric":"contents_with_end_to_end_carrythrough","value":int(decisions["end_to_end_carrythrough"].map(bool).sum())},
        {"metric":"contents_with_realization_gap","value":int(decisions["realization_gap"].map(bool).sum())},
        {"metric":"contents_with_audience_carrythrough_gap","value":int(decisions["audience_carrythrough_gap"].map(bool).sum())},
        {"metric":"contents_with_end_to_end_gap","value":int(decisions["end_to_end_gap"].map(bool).sum())},
        {"metric":"contents_with_unexpected_perception","value":int(decisions["unexpected_perception"].map(bool).sum())},
        {"metric":"contents_with_emergent_audience_signal","value":int(decisions["emergent_audience_signal"].map(bool).sum())},
        {"metric":"contents_with_audience_demand","value":int(decisions["audience_demand"].map(bool).sum())},
        {"metric":"contents_with_audience_discussion","value":int(decisions["audience_discussion"].map(bool).sum())},
        {"metric":"contents_with_community_signal","value":int(decisions["community_signal"].map(bool).sum())},
        {"metric":"contents_with_potential_risk_candidate","value":int(decisions["potential_risk_candidate"].map(bool).sum())},
        {"metric":"emergent_min_support","value":args.emergent_min_support},
        {"metric":"risk_min_support","value":args.risk_min_support},
        {"metric":"top_k","value":args.top_k},
        {"metric":"overall_score_defined","value":False},
    ])
    summary.to_csv(args.output_dir/"build_summary_5d2_v2.csv", index=False)

    print("\n=== 5D.2 v2 Semantic Cleanup complete ===")
    print(summary.to_string(index=False))

    print("\nKey finding types:")
    print(decisions["key_finding_type"].value_counts().to_string())

    preview_cols = [
        "content_id","key_finding_type",
        "content_realization","audience_carrythrough","end_to_end_carrythrough",
        "realization_gap","audience_carrythrough_gap","end_to_end_gap",
        "unexpected_perception","emergent_audience_signal",
        "audience_demand","audience_discussion","community_signal",
        "potential_risk_candidate"
    ]
    print("\nDecision insight preview:")
    print(decisions[preview_cols].to_string(index=False))

    focus = decisions[decisions["content_id"].isin(["youtube_gaw2OJ5PoDc","youtube_w0wF-O0EGsQ"])]
    if not focus.empty:
        print("\nFocus-case decision insights:")
        print(focus[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
