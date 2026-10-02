#!/usr/bin/env python3
"""
P0-A — Build the recruiter-facing demo dataset.

Reads the outputs of the full pipeline (3A/3B extraction, 4x ontology,
5A-5D alignment engine) and writes a small, self-contained, static dataset
for the 3 curated demo cases. The Streamlit app reads ONLY data/demo/, so the
deployed demo needs no API keys, no live LLM, and no local paths.

Usage
-----
# from the repo root, pointing at the workspace that holds pipeline outputs
python scripts/build_demo_dataset.py --source-root ..            # workspace = parent folder
python scripts/build_demo_dataset.py --source-root . --output-dir data/demo

Outputs (data/demo/)
--------------------
demo_content.csv             1 row per case: metadata, curated story, 3 alignment scores
demo_decision_insights.csv   long format: 5D.2 decision insights (aligned / gaps / audience intel)
demo_alignment_pairs.csv     concept-level alignment journey (aligned / partial / missing)
demo_alignment_evidence.csv  quote-level evidence: transcript spans (3A) + viewer comments (3B)
demo_concept_profiles.csv    normalized concept profiles per layer (intent / content / audience)
demo_relationship_rules.csv  governed cross-ontology relationship rules (v0.1)
demo_system_metrics.csv      pipeline funnel, 3B human-QC evaluation, scoring config, findings
demo_manifest.json           provenance: source files (relative), row counts, checksums
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Curated cases — edit the narrative here, not in the app.
# ---------------------------------------------------------------------------

CASES = [
    {
        "content_id": "youtube_w0wF-O0EGsQ",
        "demo_order": 1,
        "demo_role": "hero",
        "demo_label": "Creative Craft",
        "display_title_en": "EP.47 'Love Language' Recording Behind the Scene | TXT",
        "headline": "Vocal craftsmanship reaches the audience, under a different name",
        "story": (
            "The inferred content framing emphasizes vocal production craftsmanship, "
            "and that signal is realized in the content. Viewers respond by praising "
            "the members' vocal ability. Because the creator-side and audience-side "
            "concepts use different labels, exact matching would miss this connection. "
            "The alignment engine captures it through a human-approved 'contributes_to' "
            "relationship."
        ),
        "what_it_proves": (
            "Cross-ontology alignment: creator-side and audience-side concepts do not "
            "need identical labels to represent a governed relationship."
        ),
        "focus_path": (
            "Vocal Production Craftsmanship → "
            "Vocal Production Craftsmanship → Vocal Ability"
        ),
        "focus_concepts": [
            "Vocal Production Craftsmanship",
            "Vocal Ability",
        ],
        "checks": [
            (
                "content_realization",
                "Vocal Production Craftsmanship -> Vocal Production Craftsmanship",
            ),
            (
                "audience_carrythrough",
                "Vocal Production Craftsmanship -> Vocal Ability",
            ),
            (
                "end_to_end_carrythrough",
                "Vocal Production Craftsmanship -> Vocal Ability",
            ),
        ],
    },
    {
        "content_id": "youtube_gaw2OJ5PoDc",
        "demo_order": 2,
        "demo_role": "gap",
        "demo_label": "Values / Care",
        "display_title_en": "One Day, a Baby Appeared | TXT's Parenting Diary EP.01",
        "headline": (
            "Caring is clear in the governed content profile, "
            "but the system does not overclaim audience perception"
        ),
        "story": (
            "Caring is consistent across the inferred framing and governed content "
            "representation. No governed audience counterpart surfaces for Caring in "
            "the current audience profile. Rather than stretching a weak semantic match, "
            "the engine preserves the gap while separately surfacing what viewers did "
            "express, including gratitude and requests for subtitles or more content. "
            "The result is treated as a limitation of the current governed evidence—not "
            "proof that viewers failed to perceive care."
        ),
        "what_it_proves": (
            "Evaluation discipline: absence of a governed counterpart is preserved as "
            "uncertainty, not converted into a claim about audience perception."
        ),
        "focus_path": "Caring → Caring → (no governed audience counterpart)",
        "focus_concepts": [
            "Caring",
            "Gratitude",
            "Subtitle / Translation Request",
        ],
        "checks": [
            ("content_realization", "Caring -> Caring"),
            ("end_to_end_gap", "Caring"),
        ],
        "score_checks": [
            ("content_to_audience_score", 0.0),
            ("intent_to_content_score", 1.0),
        ],
    },
    {
        "content_id": "youtube_v5DXpxUAO3w",
        "demo_order": 3,
        "demo_role": "relationship",
        "demo_label": "Relationship Identity",
        "display_title_en": "Our Beach Day Escape | Summer Diary with TXT EP.4",
        "headline": (
            "Relationship identity carries through in multiple forms"
        ),
        "story": (
            "Playful camaraderie in the inferred framing carries through to viewers as "
            "a playful teasing dynamic through a governed 'manifests_as' relationship, "
            "while friendship carries through directly as friendship. The audience also "
            "surfaces additional perceptions—including chemistry, cuteness, and love—that "
            "were not part of the matched framing path. These are retained as emergent "
            "audience signals rather than automatically treated as product opportunities."
        ),
        "what_it_proves": (
            "The relationship engine supports both direct and cross-ontology "
            "carry-through while keeping emergent audience signals analytically separate."
        ),
        "focus_path": (
            "Playful Camaraderie → Playful Teasing Dynamic · "
            "Friendship → Friendship"
        ),
        "focus_concepts": [
            "Playful Camaraderie",
            "Playful Teasing Dynamic",
            "Friendship",
        ],
        "checks": [
            (
                "audience_carrythrough",
                "Playful Camaraderie -> Playful Teasing Dynamic",
            ),
            (
                "end_to_end_carrythrough",
                "Playful Camaraderie -> Playful Teasing Dynamic",
            ),
            (
                "end_to_end_carrythrough",
                "Friendship -> Friendship",
            ),
        ],
    },
]

# ---------------------------------------------------------------------------
# 2. Canonical source files (relative to --source-root). One place to bump versions.
# ---------------------------------------------------------------------------

SRC = {
    # raw (1. ingestion)
    "content": "data/raw/content.csv",
    "comments": "data/raw/audience_comment.csv",
    "transcripts": "data/raw/transcript_segment.csv",
    # 3A / 3B extraction
    "raw_3a": "data/processed/taxonomy_annotation_raw_3a.csv",
    "raw_3b": "data/processed/taxonomy_annotation_raw_3b.csv",
    # 3B.1 human evaluation
    "qc_metrics": "data/evaluation/audience_human_qc_metrics.csv",
    "qc_errors": "data/evaluation/audience_human_qc_error_analysis.csv",
    "qc_groups": "data/evaluation/audience_human_qc_group_analysis.csv",
    # 4A-4D ontology
    "inventory_summary": "data/processed/taxonomy_label_inventory_summary.csv",
    "candidate_summary": "data/processed/taxonomy_normalization_candidate_summary.csv",
    "consolidation_summary": "data/processed/ontology_v02_consolidation_summary.csv",
    "ontology_build": "data/ontology/v0.2/ontology_v02_build_summary.csv",
    # 5A normalized profiles
    "build_5a": "data/alignment/5a/build_summary_5a.csv",
    "normalized": "data/alignment/5a/normalized_annotations_5a.csv",
    "content_profile": "data/alignment/5a/content_concept_profile_5a.csv",
    "audience_profile": "data/alignment/5a/audience_concept_profile_5a.csv",
    # 5B relationship rules (v0.1, human-governed)
    "rules": "data/alignment/5b/rules_v0.1/relationship_rules_v0.1.csv",
    "rules_summary": "data/alignment/5b/rules_v0.1/relationship_rules_v0.1_summary.csv",
    # 5C scores
    "build_5c": "data/alignment/5c/build_summary_5c.csv",
    "content_summary_5c": "data/alignment/5c/alignment_content_summary_5c.csv",
    # 5D.1 diagnostics (v2)
    "insights_5d": "data/alignment/5d_v2/alignment_insights_5d_v2.csv",
    "diagnostic_5d": "data/alignment/5d_v2/content_alignment_diagnostic_5d_v2.csv",
    "build_5d": "data/alignment/5d_v2/build_summary_5d_v2.csv",
    # 5D.2 decision insights (v2)
    "decisions": "data/alignment/5d2_v2/decision_insights_5d2_v2.csv",
    "decision_evidence": "data/alignment/5d2_v2/decision_insight_evidence_5d2_v2.csv",
}

# 5D.2 insight field → how the app should present it
INSIGHT_DISPLAY = {
    "content_realization":       ("alignment", "aligned", "Framing realized in content"),
    "audience_carrythrough":     ("alignment", "aligned", "Content carried through to audience"),
    "end_to_end_carrythrough":   ("alignment", "aligned", "Framing carried through end-to-end"),
    "realization_gap":           ("gap", "gap", "Framing not realized in content"),
    "audience_carrythrough_gap": ("gap", "gap", "Content with weak/missing audience resonance"),
    "end_to_end_gap":            ("gap", "gap", "Framing with weak/missing audience resonance"),
    "unexpected_perception":     ("audience_intelligence", "signal", "Unexpected audience perception"),
    "emergent_audience_signal":  ("audience_intelligence", "signal", "Emergent audience response"),
    "audience_demand":           ("audience_intelligence", "signal", "Audience demand"),
    "audience_discussion":       ("audience_intelligence", "signal", "What viewers discuss"),
    "community_signal":          ("audience_intelligence", "signal", "Community signal"),
    "potential_risk_candidate":  ("audience_intelligence", "risk", "Potential risk (needs validation)"),
}
INSIGHT_ORDER = list(INSIGHT_DISPLAY)

COMPARISON_LABEL = {
    "intent_to_content": "Framing → Content",
    "content_to_audience": "Content → Audience",
    "intent_to_audience": "Framing → Audience",
}
ROLE_TO_LAYER = {
    "inferred_content_intent": "intent",
    "ai_content_understanding": "content",
    "audience_perception": "audience",
}

MAX_QUOTES_PER_CONCEPT = 5     # viewer comments kept per (case, audience concept)
MAX_QUOTE_CHARS = 280

PROMPT_ONTOLOGY_FINDINGS = [
    ("generic_interaction", "Generic interaction ≠ Community Signal",
     "Generic interaction alone should not be labeled as a community signal."),
    ("emoji_only", "Emoji-only ≠ sufficient signal",
     "Emoji-only comments are not sufficient evidence for an audience label."),
    ("content_callback", "Content callback → Community candidate",
     "Comments that call back to the content are candidates for community signals."),
    ("member_fandom", "Member/fandom references can be under-detected",
     "Member and fandom references are a source of under-interpretation."),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class BuildError(RuntimeError):
    pass


def load(root: Path, key: str) -> pd.DataFrame:
    path = root / SRC[key]
    if not path.exists():
        raise BuildError(f"Missing source file for '{key}': {SRC[key]} (source root: {root})")
    return pd.read_csv(path, low_memory=False)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def metric_lookup(df: pd.DataFrame, key_col: str = "metric", val_col: str = "value") -> dict:
    return dict(zip(df[key_col].astype(str), df[val_col]))


def clean_text(x, limit: int | None = None) -> str:
    if pd.isna(x):
        return ""
    s = re.sub(r"\s+", " ", str(x)).strip()
    if limit and len(s) > limit:
        s = s[: limit - 1].rstrip() + "…"
    return s


def short_id(*parts) -> str:
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:10]


def video_id(content_id: str) -> str:
    return content_id.split("_", 1)[1] if content_id.startswith("youtube_") else content_id


def num(x, nd=4):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return np.nan
    return np.nan if np.isnan(f) else round(f, nd)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def build_content(root, ids, case_df) -> pd.DataFrame:
    content = load(root, "content")
    decisions = load(root, "decisions")
    summary_5c = load(root, "content_summary_5c")
    raw_3b = load(root, "raw_3b")
    normalized = load(root, "normalized")

    c = content[content.content_id.isin(ids)].copy()
    keep = ["content_id", "creator_name", "title", "url", "publish_time", "duration_seconds",
            "language", "view_count", "like_count", "comment_count", "collected_comment_rows",
            "transcript_status", "transcript_segment_count"]
    c = c[[k for k in keep if k in c.columns]].rename(columns={"title": "original_title"})
    c["youtube_video_id"] = c.content_id.map(video_id)
    c["thumbnail_url"] = "https://i.ytimg.com/vi/" + c.youtube_video_id + "/hqdefault.jpg"
    c["embed_url"] = "https://www.youtube.com/embed/" + c.youtube_video_id
    c["publish_date"] = pd.to_datetime(c.publish_time, errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    c = c.drop(columns=["publish_time"])

    d = decisions[decisions.content_id.isin(ids)][[
        "content_id", "key_finding_type", "interpretation_scope",
        "intent_to_content_score", "intent_to_content_coverage", "intent_to_content_relationship_strength",
        "content_to_audience_score", "content_to_audience_coverage", "content_to_audience_relationship_strength",
        "intent_to_audience_score", "intent_to_audience_coverage", "intent_to_audience_relationship_strength",
    ]]
    s = summary_5c[summary_5c.content_id.isin(ids)][["content_id", "overall_score_defined", "overall_score_note"]]

    # annotation volume per case
    ann_comments = (raw_3b[raw_3b.content_id.isin(ids)]
                    .groupby("content_id").comment_id.nunique().rename("annotated_comment_count"))
    n = normalized[normalized.content_id.isin(ids)]
    concept_counts = (n.dropna(subset=["canonical_concept"])
                      .groupby(["content_id", "annotation_role"]).canonical_concept.nunique()
                      .unstack(fill_value=0)
                      .rename(columns=lambda r: f"{ROLE_TO_LAYER.get(r, r)}_concept_count"))
    mapping_rate = (n.assign(m=n.normalization_status.eq("mapped"))
                    .groupby("content_id").m.mean().rename("normalization_mapping_rate"))

    out = (case_df.merge(c, on="content_id", how="left")
                  .merge(d, on="content_id", how="left")
                  .merge(s, on="content_id", how="left")
                  .merge(ann_comments, on="content_id", how="left")
                  .merge(concept_counts, on="content_id", how="left")
                  .merge(mapping_rate, on="content_id", how="left"))
    for col in [x for x in out.columns if x.endswith(("_score", "_coverage", "_strength", "_rate"))]:
        out[col] = out[col].map(num)
    return out.sort_values("demo_order").reset_index(drop=True)


def build_decision_insights(root, ids) -> pd.DataFrame:
    ev = load(root, "decision_evidence")
    ev = ev[ev.content_id.isin(ids)].copy()
    ev["display_section"] = ev.insight_field.map(lambda f: INSIGHT_DISPLAY.get(f, ("other",) * 3)[0])
    ev["polarity"] = ev.insight_field.map(lambda f: INSIGHT_DISPLAY.get(f, ("other",) * 3)[1])
    ev["display_label"] = ev.insight_field.map(lambda f: INSIGHT_DISPLAY.get(f, (None, None, f))[2])
    ev["comparison_label"] = ev.comparison_type.map(COMPARISON_LABEL)
    ev["field_order"] = ev.insight_field.map(lambda f: INSIGHT_ORDER.index(f) if f in INSIGHT_ORDER else 99)

    def statement(r):
        src, tgt = clean_text(r.source_concept), clean_text(r.target_concept)
        if src and tgt:
            return f"{src} → {tgt}"
        return src or tgt
    ev["statement"] = ev.apply(statement, axis=1)

    for col in ["relationship_strength", "pair_confidence", "audience_support_rate", "evidence_strength"]:
        ev[col] = ev[col].map(num)
    cols = ["content_id", "display_section", "insight_field", "display_label", "polarity", "field_order",
            "rank", "statement", "comparison_type", "comparison_label", "evidence_source",
            "signal_or_insight_type", "source_role", "source_category", "source_concept",
            "target_role", "target_category", "target_concept", "relationship_type",
            "relationship_strength", "pair_confidence", "audience_support_rate", "evidence_strength"]
    return ev[cols].sort_values(["content_id", "field_order", "rank"]).reset_index(drop=True)


def build_alignment_pairs(root, ids, focus) -> pd.DataFrame:
    ins = load(root, "insights_5d")
    diag = load(root, "diagnostic_5d")
    p = ins[ins.content_id.isin(ids)].copy()
    p["comparison_label"] = p.comparison_type.map(COMPARISON_LABEL)
    p["source_layer"] = p.source_role.map(ROLE_TO_LAYER)
    p["target_layer"] = p.target_role.map(ROLE_TO_LAYER)
    p["is_focus_pair"] = p.apply(
        lambda r: r.source_concept in focus[r.content_id]
        and (pd.isna(r.target_concept) or r.target_concept in focus[r.content_id]), axis=1)
    p["comparison_order"] = p.comparison_type.map({k: i for i, k in enumerate(COMPARISON_LABEL)})
    p["status_order"] = p.insight_type.map({"aligned_concept": 0, "partial_alignment": 1, "missing_resonance": 2})
    for col in ["relationship_strength", "pair_confidence", "audience_support_rate",
                "audience_support_factor", "evidence_strength"]:
        p[col] = p[col].map(num)

    d = diag[diag.content_id.isin(ids)][[
        "content_id", "comparison_type", "alignment_computable", "alignment_status", "alignment_score",
        "aligned_concept_count", "partial_alignment_count", "missing_resonance_count"]].copy()
    d = d.rename(columns={"alignment_score": "comparison_alignment_score"})
    d["comparison_alignment_score"] = d.comparison_alignment_score.map(num)
    p = p.merge(d, on=["content_id", "comparison_type"], how="left")

    cols = ["content_id", "comparison_type", "comparison_label", "comparison_order", "insight_type",
            "status_order", "is_focus_pair", "source_layer", "source_category", "source_concept",
            "target_layer", "target_category", "target_concept", "relationship_type", "relationship_source",
            "relationship_strength", "pair_confidence", "audience_support_rate", "audience_support_factor",
            "evidence_strength", "comparison_alignment_score", "alignment_status", "alignment_computable",
            "aligned_concept_count", "partial_alignment_count", "missing_resonance_count", "interpretation"]
    return p[cols].sort_values(["content_id", "comparison_order", "status_order", "evidence_strength"],
                               ascending=[True, True, True, False]).reset_index(drop=True)


def build_evidence(root, ids, focus) -> pd.DataFrame:
    n = load(root, "normalized")
    raw_3a = load(root, "raw_3a")[["annotation_id", "explanation", "source_field"]]
    raw_3b = load(root, "raw_3b")[["annotation_id", "explanation"]]
    comments = load(root, "comments")[["comment_id", "comment_text", "like_count", "is_reply",
                                       "engagement_stage", "hours_after_publish"]]

    n = n[n.content_id.isin(ids)].copy()
    n["layer"] = n.annotation_role.map(ROLE_TO_LAYER)

    # 3A — creator-side: every annotation (mapped or not), with its transcript/metadata span
    a = n[n.dataset.eq("3a")].merge(raw_3a, on="annotation_id", how="left")
    a["evidence_type"] = "transcript_or_metadata"
    a["evidence_text"] = a.evidence_span.map(lambda x: clean_text(x, MAX_QUOTE_CHARS))
    a["evidence_likes"] = np.nan
    a["engagement_stage"] = ""

    # 3B — audience-side: mapped concepts only, top-N most-liked comments per concept
    b = (n[n.dataset.eq("3b") & n.canonical_concept.notna()]
         .merge(raw_3b, on="annotation_id", how="left")
         .merge(comments, on="comment_id", how="left"))
    b["evidence_type"] = "viewer_comment"
    b["evidence_text"] = b.comment_text.map(lambda x: clean_text(x, MAX_QUOTE_CHARS))
    b["evidence_likes"] = pd.to_numeric(b.like_count, errors="coerce")
    b = b.sort_values(["content_id", "canonical_concept", "evidence_likes"], ascending=[True, True, False])
    b = b.drop_duplicates(["content_id", "canonical_concept", "comment_id"])
    b = b.groupby(["content_id", "canonical_concept"], group_keys=False).head(MAX_QUOTES_PER_CONCEPT)

    e = pd.concat([a, b], ignore_index=True, sort=False)
    e["evidence_id"] = e.apply(lambda r: short_id(r.content_id, r.annotation_id), axis=1)  # no raw comment ids
    e["evidence_span"] = e.evidence_span.map(lambda x: clean_text(x, MAX_QUOTE_CHARS))
    e["model_explanation"] = e.explanation.map(lambda x: clean_text(x, 400))
    e["is_mapped"] = e.normalization_status.eq("mapped")
    e["is_focus_concept"] = e.apply(lambda r: r.canonical_concept in focus[r.content_id], axis=1)
    e["evidence_rank"] = (e.sort_values("evidence_likes", ascending=False)
                           .groupby(["content_id", "layer", "canonical_concept"], dropna=False)
                           .cumcount() + 1)
    e["layer_order"] = e.layer.map({"intent": 0, "content": 1, "audience": 2})

    cols = ["evidence_id", "content_id", "layer", "layer_order", "evidence_type", "raw_category", "raw_label",
            "normalization_status", "is_mapped", "canonical_category", "canonical_concept", "ontology_node_id",
            "is_focus_concept", "evidence_rank", "evidence_span", "evidence_text", "evidence_likes",
            "engagement_stage", "model_explanation"]
    return e[cols].sort_values(["content_id", "layer_order", "is_focus_concept", "canonical_concept", "evidence_rank"],
                               ascending=[True, True, False, True, True]).reset_index(drop=True)


def build_profiles(root, ids, focus) -> pd.DataFrame:
    cp = load(root, "content_profile")
    ap = load(root, "audience_profile")
    cp = cp[cp.content_id.isin(ids)].copy()
    cp["layer"] = cp.annotation_role.map(ROLE_TO_LAYER)
    cp = cp.rename(columns={"concept_rank_within_role": "rank_within_layer",
                            "audience_mapped_signal_support_rate": "share_of_layer"})
    ap = ap[ap.content_id.isin(ids)].copy()
    ap["layer"] = "audience"
    ap = ap.rename(columns={"rank_within_content": "rank_within_layer"})
    ap["share_of_layer"] = ap.comment_prevalence
    p = pd.concat([cp, ap], ignore_index=True, sort=False)
    p["is_focus_concept"] = p.apply(lambda r: r.canonical_concept in focus[r.content_id], axis=1)
    for col in ["share_of_layer", "comment_prevalence", "avg_confidence"]:
        p[col] = p[col].map(num)
    cols = ["content_id", "layer", "canonical_category", "canonical_concept", "ontology_node_id",
            "is_focus_concept", "annotation_count", "unique_comment_count", "total_content_comment_count",
            "comment_prevalence", "share_of_layer", "avg_confidence", "rank_within_layer"]
    order = {"intent": 0, "content": 1, "audience": 2}
    return (p[cols].assign(_o=p.layer.map(order))
            .sort_values(["content_id", "_o", "rank_within_layer"]).drop(columns="_o").reset_index(drop=True))


def build_rules(root, used_concepts: set) -> pd.DataFrame:
    r = load(root, "rules").copy()
    r["source_layer"] = r.source_role.map(ROLE_TO_LAYER)
    r["target_layer"] = r.target_role.map(ROLE_TO_LAYER)
    r["used_in_demo"] = r.source_concept.isin(used_concepts) | r.target_concept.isin(used_concepts)
    cols = ["source_layer", "source_category", "source_concept", "target_layer", "target_category",
            "target_concept", "relationship_type", "relationship_strength", "directional", "rule_status",
            "rule_source", "used_in_demo", "notes"]
    return r[[c for c in cols if c in r.columns]]


def build_system_metrics(root) -> pd.DataFrame:
    rows = []

    def add(section, key, label, value, fmt="int", note="", source=""):
        # summary CSVs mix text and numbers in one column, so numbers can arrive as strings
        v = value if fmt == "text" else num(value, 6)
        if fmt != "text":
            value = v
        if isinstance(v, float) and np.isnan(v):
            display = "—"
        elif fmt == "pct":
            display = f"{v:.0%}" if v <= 1 else f"{v:.0f}%"
        elif fmt == "pct1":
            display = f"{v:.1%}"
        elif fmt == "dec":
            display = f"{v:.2f}"
        elif fmt == "text":
            display = str(value)
        else:
            display = f"{int(round(v)):,}"
        rows.append(dict(section=section, metric_key=key, label=label, value=value, display_value=display,
                         note=note, source_file=SRC.get(source, source)))

    content = load(root, "content")
    comments = load(root, "comments")
    transcripts = load(root, "transcripts")
    b5a = metric_lookup(load(root, "build_5a"))
    cand = load(root, "candidate_summary")
    cand_all = cand[cand.summary_level.eq("overall")].iloc[0]
    cons = load(root, "consolidation_summary")
    cons_all = cons[cons.summary_level.eq("overall")].iloc[0]
    onto = metric_lookup(load(root, "ontology_build"))
    rules = load(root, "rules")
    rsum = metric_lookup(load(root, "rules_summary"))
    b5c = metric_lookup(load(root, "build_5c"))
    b5d = metric_lookup(load(root, "build_5d"))

    # --- funnel: what the system processed
    S = "pipeline_funnel"
    add(S, "videos", "Videos analyzed", len(content), source="content")
    add(S, "comments", "Viewer comments collected", len(comments), source="comments")
    add(S, "transcript_segments", "Transcript segments", len(transcripts), source="transcripts")
    add(S, "annotations_3a", "Creator-side annotations (3A)", b5a.get("3a_annotation_rows"), source="build_5a")
    add(S, "annotations_3b", "Audience-side annotations (3B)", b5a.get("3b_annotation_rows"), source="build_5a")
    add(S, "raw_labels", "Free-form raw labels", cand_all.unique_raw_label_count,
        note="Vocabulary fragmentation before normalization (4A)", source="candidate_summary")
    add(S, "semantic_groups", "Semantic candidate groups (4B.1)", cand_all.candidate_cluster_count,
        source="candidate_summary")
    add(S, "new_node_proposals", "New-concept proposals (4B.2)", cons_all.new_node_proposal_count,
        source="consolidation_summary")
    add(S, "consolidated_concepts", "Consolidated proposals", cons_all.consolidated_concept_count,
        source="consolidation_summary")
    add(S, "human_review_rows", "Ontology rows human-reviewed (4C)", onto.get("review_completed_rows"),
        source="ontology_build")
    add(S, "ontology_v01_nodes", "Ontology v0.1 nodes", onto.get("seed_v01_node_count"), source="ontology_build")
    add(S, "ontology_v02_nodes", "Ontology v0.2 nodes", onto.get("final_v02_node_count"), source="ontology_build")
    add(S, "normalization_mapping_rate", "Annotations mapped to governed ontology (5A)",
        b5a.get("overall_mapping_rate"), fmt="pct",
        note="Unmapped labels are kept but excluded from alignment scoring", source="build_5a")
    add(S, "relationship_rules_reviewed", "Relationship rules human-reviewed (5B.1)",
        rsum.get("review_completed_rows"), source="rules_summary")
    add(S, "relationship_rules_approved", "Approved relationship rules", (rules.rule_status == "approved").sum(),
        source="rules")
    add(S, "relationship_rules_rejected", "Rejected relationship rules", (rules.rule_status == "rejected").sum(),
        source="rules")
    add(S, "comparisons", "Alignment comparisons (videos × 3)", b5c.get("comparison_rows"), source="build_5c")
    add(S, "comparisons_computable", "Computable comparisons", b5c.get("comparison_rows_computable"),
        source="build_5c")

    # --- 3B.1 human evaluation
    q = metric_lookup(load(root, "qc_metrics"))
    S = "eval_signal_detection"
    add(S, "qc_sample_size", "Human-QC sample size", q.get("qc_sample_size"), source="qc_metrics")
    add(S, "signal_precision", "Precision", q.get("signal_precision"), "pct1", source="qc_metrics")
    add(S, "signal_recall", "Recall", q.get("signal_recall"), "pct1", source="qc_metrics")
    add(S, "signal_f1", "F1", q.get("signal_f1"), "pct1", source="qc_metrics")
    add(S, "signal_accuracy", "Accuracy", q.get("signal_accuracy"), "pct1", source="qc_metrics")
    for k in ["true_positive", "false_positive", "false_negative", "true_negative"]:
        add(S, k, k.replace("_", " ").title(), q.get(k), source="qc_metrics")
    S = "eval_interpretation_quality"
    add(S, "overall_exact_accuracy", "Overall correct", q.get("overall_exact_accuracy"), "pct", source="qc_metrics")
    add(S, "overall_correct_or_partial_rate", "Correct + partial", q.get("overall_correct_or_partial_rate"), "pct",
        source="qc_metrics")
    add(S, "category_exact_accuracy", "Category validity", q.get("category_exact_accuracy"), "pct",
        source="qc_metrics")
    add(S, "label_exact_accuracy", "Label validity", q.get("label_exact_accuracy"), "pct", source="qc_metrics")
    S = "eval_error_analysis"
    err = load(root, "qc_errors")
    for _, r in err.iterrows():
        add(S, str(r.error_type), str(r.error_type).replace("_", "-").capitalize(), r["count"],
            note=f"{float(r.percent_of_qc_sample):.0%} of QC sample", source="qc_errors")
    S = "eval_findings"
    for key, label, note in PROMPT_ONTOLOGY_FINDINGS:
        add(S, key, label, label, "text", note=note, source="3B.1 human evaluation")

    # --- scoring configuration (transparency)
    S = "scoring_config"
    add(S, "audience_support_transform", "Audience support transform", str(b5c.get("audience_support_transform")),
        "text", source="build_5c")
    add(S, "confidence_floor", "Pair confidence floor", b5c.get("confidence_floor"), "dec", source="build_5c")
    add(S, "aligned_threshold", "Aligned threshold", b5d.get("aligned_threshold"), "dec", source="build_5d")
    add(S, "partial_threshold", "Partial threshold", b5d.get("partial_threshold"), "dec", source="build_5d")
    add(S, "overall_score_defined", "Overall score defined", "No (V1 keeps the 3 comparisons separate)", "text",
        source="build_5c")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def validate(outputs: dict, cases: list) -> list[str]:
    errors = []
    content = outputs["demo_content.csv"]
    decisions = outputs["demo_decision_insights.csv"]
    ids = [c["content_id"] for c in cases]

    for name, df in outputs.items():
        if not isinstance(df, pd.DataFrame):
            continue
        if df.empty:
            errors.append(f"{name}: empty")
        if "content_id" in df.columns:
            missing = set(ids) - set(df.content_id)
            if missing and name not in ("demo_relationship_rules.csv", "demo_system_metrics.csv"):
                errors.append(f"{name}: missing cases {sorted(missing)}")
        blob = df.astype(str).to_csv(index=False)
        for pat, why in [(r"/Users/|/home/|[A-Za-z]:\\\\", "absolute local path"),
                         (r"sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_\-]{30,}", "API-key-like string")]:
            if re.search(pat, blob):
                errors.append(f"{name}: contains {why}")

    for col in ["url", "original_title", "intent_to_content_score", "content_to_audience_score",
                "intent_to_audience_score"]:
        if content[col].isna().any():
            errors.append(f"demo_content.csv: null {col} for {content[content[col].isna()].content_id.tolist()}")

    # story guards: the curated narrative must still be true in the data
    raw_dec = outputs["_raw_decisions"]
    for case in cases:
        row = raw_dec[raw_dec.content_id == case["content_id"]]
        if row.empty:
            errors.append(f"{case['content_id']}: not in 5D.2 decision insights")
            continue
        row = row.iloc[0]
        for field, needle in case.get("checks", []):
            if needle not in str(row.get(field, "")):
                errors.append(f"{case['demo_role']}: expected '{needle}' in {field}, got '{row.get(field)}'")
        for field, expected in case.get("score_checks", []):
            if not np.isclose(float(row[field]), expected):
                errors.append(f"{case['demo_role']}: expected {field}={expected}, got {row[field]}")
    if decisions.statement.eq("").any():
        errors.append("demo_decision_insights.csv: empty statements")
    return errors


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="P0-A: build static demo dataset for the 3 curated cases")
    ap.add_argument("--source-root", type=Path, default=Path("."),
                    help="Folder that contains the pipeline's data/ directory")
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "demo")
    ap.add_argument("--dry-run", action="store_true", help="Validate only, do not write files")
    args = ap.parse_args()
    root = args.source_root.resolve()

    ids = [c["content_id"] for c in CASES]
    focus = {c["content_id"]: set(c["focus_concepts"]) for c in CASES}
    case_df = pd.DataFrame([{k: v for k, v in c.items() if k not in ("checks", "score_checks", "focus_concepts")}
                            | {"focus_concepts": " | ".join(c["focus_concepts"])} for c in CASES])

    print(f"[P0-A] source root: {root}")
    outputs = {
        "demo_content.csv": build_content(root, ids, case_df),
        "demo_decision_insights.csv": build_decision_insights(root, ids),
        "demo_alignment_pairs.csv": build_alignment_pairs(root, ids, focus),
        "demo_alignment_evidence.csv": build_evidence(root, ids, focus),
        "demo_concept_profiles.csv": build_profiles(root, ids, focus),
        "demo_system_metrics.csv": build_system_metrics(root),
    }
    used = set().union(*focus.values()) | set(outputs["demo_alignment_pairs.csv"].source_concept.dropna())
    outputs["demo_relationship_rules.csv"] = build_rules(root, used)
    outputs["_raw_decisions"] = load(root, "decisions")

    errors = validate(outputs, CASES)
    if errors:
        print("\n[P0-A] VALIDATION FAILED:")
        for e in errors:
            print("  ✗", e)
        return 1
    print("[P0-A] validation passed ✓")

    outputs.pop("_raw_decisions")
    if args.dry_run:
        for name, df in outputs.items():
            print(f"  (dry-run) {name:32s} {len(df):5d} rows")
        return 0

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    file_meta = {}
    for name, df in outputs.items():
        path = out / name
        df.to_csv(path, index=False, encoding="utf-8")
        size = path.stat().st_size
        total += size
        file_meta[name] = {"rows": int(len(df)), "columns": list(df.columns), "bytes": size}
        print(f"  ✓ {name:32s} {len(df):5d} rows  {size / 1024:7.1f} KB")

    manifest = {
        "dataset": "alignment-engine-demo",
        "milestone": "P0-A",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": [{k: c[k] for k in ("content_id", "demo_order", "demo_role", "demo_label", "focus_path")}
                  for c in CASES],
        "versions": {"ontology": "v0.2", "relationship_rules": "v0.1", "alignment_scores": "5C",
                     "diagnostics": "5D.1 v2", "decision_insights": "5D.2 v2"},
        "sources": {k: {"path": v, "sha256_16": sha256(root / v)} for k, v in SRC.items()},
        "outputs": file_meta,
        "notes": [
            "Static snapshot: the demo app needs no API keys, no live LLM, and no local paths.",
            "Viewer comments are public YouTube comments; author identifiers and comment IDs are excluded.",
            "Absence of a governed audience counterpart does not prove absence of human perception.",
        ],
    }
    (out / "demo_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓ demo_manifest.json\n[P0-A] wrote {len(outputs) + 1} files, {total / 1024:.0f} KB → {out}")
    if total > 5 * 1024 * 1024:
        print("[P0-A] WARNING: demo dataset > 5 MB; consider lowering MAX_QUOTES_PER_CONCEPT")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BuildError as e:
        print(f"[P0-A] ERROR: {e}")
        sys.exit(2)
