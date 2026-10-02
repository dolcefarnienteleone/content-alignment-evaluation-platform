#!/usr/bin/env python3
"""
5A v2 — Build Normalized Alignment Profiles

Why v2 exists
-------------
The 4D file taxonomy_normalization_mapping_v02.csv is an audit/provenance
mapping for human-reviewed consolidations. It is NOT a complete raw-label
normalization table for all 3A/3B annotations.

5A therefore reconstructs the full normalization backbone from:
  1) 4B semantic candidates      raw labels -> candidate_id
  2) 4B.2 LLM proposals         candidate_id -> proposed ontology outcome
  3) 4D human mapping           reviewed candidate_id -> final v0.2 outcome
  4) ontology_v02_nodes.csv     validates/resolves canonical nodes
  5) 3A/3B raw annotations      rows to normalize

Resolution precedence
---------------------
A. Human-reviewed 4D outcome wins.
   - active_in_v02=True  -> map to human canonical node
   - defer/reject        -> intentionally unresolved
B. Otherwise, 4B.2 "existing_node" proposals map to retained ontology nodes.
C. Unreviewed new_node / ambiguous / reject remain unresolved for MVP.
D. No raw label is silently discarded.

Outputs
-------
normalized_annotations_5a.csv
content_concept_profile_5a.csv
content_role_profile_5a.csv
audience_concept_profile_5a.csv
alignment_input_profile_5a.jsonl
normalization_unmapped_5a.csv
normalization_qc_5a.csv
normalization_backbone_5a.csv
build_summary_5a.csv

Example
-------
python build_normalized_alignment_profiles_5a_v3.py \
  --annotations-3a data/processed/taxonomy_annotation_raw_3a.csv \
  --annotations-3b data/processed/taxonomy_annotation_raw_3b.csv \
  --candidates data/processed/taxonomy_normalization_candidates.csv \
  --proposals data/processed/taxonomy_normalization_candidates_proposed.csv \
  --mapping-4d data/ontology/v0.2/taxonomy_normalization_mapping_v02.csv \
  --ontology-nodes data/ontology/v0.2/ontology_v02_nodes.csv \
  --content data/raw/content.csv \
  --audience data/raw/audience_comment.csv \
  --output-dir data/alignment/5a
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def txt(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def key(x: Any) -> str:
    return txt(x).casefold()


def first_col(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    for c in names:
        if c in df.columns:
            return c
    return None


def boolish(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return key(x) in {"1", "true", "yes", "y", "t"}


def safe_float(x: Any) -> float:
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def split_multi(x: Any) -> List[str]:
    """
    Supports:
      a | b | c
      a ; b ; c
      JSON/Python list strings
      single value
    """
    s = txt(x)
    if not s:
        return []

    if s.startswith("[") and s.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(s)
                if isinstance(obj, list):
                    return [txt(v) for v in obj if txt(v)]
            except Exception:
                pass

    if " | " in s or "|" in s:
        return [txt(v) for v in s.split("|") if txt(v)]
    if " ; " in s:
        return [txt(v) for v in s.split(";") if txt(v)]
    return [s]


def infer_side(source: Any) -> str:
    s = key(source)
    if s.startswith("3a") or "content" in s or "creator" in s:
        return "3a"
    if s.startswith("3b") or "audience" in s:
        return "3b"
    return s


def normalize_role(raw: Any, dataset: str) -> str:
    r = key(raw)
    if "inferred_content_intent" in r or "creator_intent" in r:
        return "inferred_content_intent"
    if "ai_content_understanding" in r or "content_understanding" in r:
        return "ai_content_understanding"
    if "audience" in r or dataset == "3b":
        return "audience_perception"
    return txt(raw) or ("audience_perception" if dataset == "3b" else "unknown")


# ---------------------------------------------------------------------
# Ontology nodes
# ---------------------------------------------------------------------

def load_nodes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    node_col = first_col(df, ["ontology_node_id", "node_id", "id"])
    concept_col = first_col(df, ["canonical_concept", "concept", "node_label", "label"])
    cat_col = first_col(df, ["taxonomy_category", "canonical_category", "category", "ontology_category"])
    source_col = first_col(df, ["inventory_source", "source", "ontology_source", "side"])

    if not node_col or not concept_col or not cat_col:
        raise ValueError(
            "ontology_v02_nodes.csv must contain node id, concept, and category columns. "
            f"Columns={list(df.columns)}"
        )

    out = pd.DataFrame({
        "ontology_node_id": df[node_col].map(txt),
        "canonical_concept": df[concept_col].map(txt),
        "canonical_category": df[cat_col].map(txt),
        "inventory_source": df[source_col].map(txt) if source_col else "",
    })
    out["side"] = out["inventory_source"].map(infer_side)
    return out.drop_duplicates().reset_index(drop=True)


def resolve_node(
    nodes: pd.DataFrame,
    concept: str,
    category: str = "",
    source: str = "",
) -> Tuple[str, str, str, str]:
    """
    Returns:
      node_id, resolved_concept, resolved_category, status
    """
    ckey = key(concept)
    catkey = key(category)
    side = infer_side(source)

    if not ckey:
        return "", "", "", "missing_concept"

    cand = nodes[nodes["canonical_concept"].map(key) == ckey].copy()
    if side in {"3a", "3b"}:
        side_cand = cand[cand["side"] == side]
        if not side_cand.empty:
            cand = side_cand

    if catkey:
        cat_cand = cand[cand["canonical_category"].map(key) == catkey]
        if not cat_cand.empty:
            cand = cat_cand

    if len(cand) == 1:
        r = cand.iloc[0]
        return (
            txt(r["ontology_node_id"]),
            txt(r["canonical_concept"]),
            txt(r["canonical_category"]),
            "resolved",
        )

    if len(cand) == 0:
        return "", concept, category, "node_not_found"

    return "", concept, category, "node_ambiguous"


# ---------------------------------------------------------------------
# 4B candidate raw-label membership
# ---------------------------------------------------------------------

CANDIDATE_ID_NAMES = [
    "candidate_id",
    "normalization_candidate_id",
    "candidate_cluster_id",
    "cluster_id",
]

RAW_LABEL_LIST_NAMES = [
    "member_raw_labels",
    "raw_labels",
    "member_labels",
    "cluster_member_labels",
    "candidate_member_labels",
    "raw_label_members",
    "labels",
]

RAW_LABEL_SINGLE_NAMES = [
    "raw_label",
    "member_raw_label",
    "source_raw_label",
    "original_raw_label",
]

ROLE_NAMES = ["annotation_role", "role"]
CATEGORY_NAMES = ["taxonomy_category", "proposed_taxonomy_category", "category"]
SOURCE_NAMES = ["inventory_source", "source", "annotation_source"]


def load_candidate_membership(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cid_col = first_col(df, CANDIDATE_ID_NAMES)
    if not cid_col:
        raise ValueError(
            f"Could not detect candidate id in {path}. "
            f"Tried={CANDIDATE_ID_NAMES}; Columns={list(df.columns)}"
        )

    raw_single = first_col(df, RAW_LABEL_SINGLE_NAMES)
    raw_list = first_col(df, RAW_LABEL_LIST_NAMES)

    # Prefer explicit member list over representative/raw single label.
    raw_col = raw_list or raw_single
    if not raw_col:
        # Last-resort heuristic: find a column containing both "raw" and "label".
        heur = [c for c in df.columns if "raw" in c.casefold() and "label" in c.casefold()]
        if heur:
            raw_col = heur[0]

    if not raw_col:
        raise ValueError(
            "Could not detect raw-label/member-label column in candidate file. "
            f"Columns={list(df.columns)}"
        )

    role_col = first_col(df, ROLE_NAMES)
    cat_col = first_col(df, CATEGORY_NAMES)
    source_col = first_col(df, SOURCE_NAMES)

    rows = []
    for _, r in df.iterrows():
        cid = txt(r[cid_col])
        labels = split_multi(r[raw_col])

        for raw_label in labels:
            rows.append({
                "candidate_id": cid,
                "raw_label": raw_label,
                "raw_label_key": key(raw_label),
                "annotation_role": txt(r[role_col]) if role_col else "",
                "taxonomy_category": txt(r[cat_col]) if cat_col else "",
                "inventory_source": txt(r[source_col]) if source_col else "",
            })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Candidate membership expansion produced zero rows.")

    return out.drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------
# 4B.2 proposal backbone
# ---------------------------------------------------------------------

PROPOSAL_LABEL_NAMES = [
    "proposed_normalized_label_y",
    "proposed_normalized_label",
    "normalized_label",
    "proposed_label",
    "canonical_label",
    "proposed_normalized_label_x",
]

PROPOSAL_CATEGORY_NAMES = [
    "proposed_taxonomy_category_y",
    "proposed_taxonomy_category",
    "taxonomy_category",
    "proposed_category",
    "proposed_taxonomy_category_x",
]

PROPOSAL_DECISION_NAMES = [
    "ontology_decision_y",
    "ontology_decision",
    "proposed_ontology_decision",
    "decision",
    "ontology_decision_x",
]

EXISTING_NODE_NAMES = [
    "possible_existing_node",
    "matched_existing_node",
    "existing_node",
    "merge_target_existing_node",
    "matched_ontology_node",
]


def load_proposals(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    cid_col = first_col(df, CANDIDATE_ID_NAMES)
    label_col = first_col(df, PROPOSAL_LABEL_NAMES)
    cat_col = first_col(df, PROPOSAL_CATEGORY_NAMES)
    decision_col = first_col(df, PROPOSAL_DECISION_NAMES)
    existing_col = first_col(df, EXISTING_NODE_NAMES)
    source_col = first_col(df, SOURCE_NAMES)
    role_col = first_col(df, ROLE_NAMES)

    missing = [
        name for name, col in {
            "candidate_id": cid_col,
            "proposed label": label_col,
            "ontology decision": decision_col,
        }.items() if not col
    ]
    if missing:
        raise ValueError(
            f"Proposal file missing {missing}. Columns={list(df.columns)}"
        )

    out = pd.DataFrame({
        "candidate_id": df[cid_col].map(txt),
        "proposal_label": df[label_col].map(txt),
        "proposal_category": df[cat_col].map(txt) if cat_col else "",
        "proposal_decision": df[decision_col].map(txt),
        "matched_existing_node": df[existing_col].map(txt) if existing_col else "",
        "proposal_inventory_source": df[source_col].map(txt) if source_col else "",
        "proposal_annotation_role": df[role_col].map(txt) if role_col else "",
    })

    if out["candidate_id"].duplicated().any():
        dup = out.loc[out["candidate_id"].duplicated(False), "candidate_id"].head(10).tolist()
        raise ValueError(f"Proposal file should have one row per candidate_id. Duplicates example={dup}")

    # Defensive checks: a proposal export can contain merge-suffix columns where
    # one side is entirely blank. Detect an accidentally selected blank column
    # early rather than failing later with 0% mapping.
    if out["proposal_decision"].eq("").all():
        raise ValueError(
            "Detected proposal decision column is entirely blank. "
            "For merged 4B.2 exports, the populated field is commonly ontology_decision_y."
        )
    if out["proposal_label"].eq("").mean() > 0.25:
        # Some reject/ambiguous rows may legitimately have no label, but most
        # proposal rows should still have one.
        nonblank = int(out["proposal_label"].ne("").sum())
        print(
            f"WARNING: proposal_label is populated for only {nonblank}/{len(out)} rows. "
            "Verify the selected proposal-label column if this is unexpected."
        )

    return out


# ---------------------------------------------------------------------
# 4D human override
# ---------------------------------------------------------------------

def load_human_mapping(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    cid_col = first_col(df, CANDIDATE_ID_NAMES)
    if not cid_col:
        raise ValueError(
            f"4D mapping missing candidate_id. Columns={list(df.columns)}"
        )

    concept_col = first_col(df, ["canonical_concept", "human_final_concept"])
    cat_col = first_col(df, ["canonical_category", "human_final_category"])
    node_col = first_col(df, ["ontology_node_id", "canonical_node_id"])
    active_col = first_col(df, ["active_in_v02", "is_active"])
    decision_col = first_col(df, ["human_decision", "human_ontology_decision", "ontology_decision"])

    if not concept_col or not active_col:
        raise ValueError(
            "4D mapping must include canonical_concept and active_in_v02. "
            f"Columns={list(df.columns)}"
        )

    out = pd.DataFrame({
        "candidate_id": df[cid_col].map(txt),
        "human_canonical_concept": df[concept_col].map(txt),
        "human_canonical_category": df[cat_col].map(txt) if cat_col else "",
        "human_ontology_node_id": df[node_col].map(txt) if node_col else "",
        "active_in_v02": df[active_col].map(boolish),
        "human_decision": df[decision_col].map(txt) if decision_col else "",
    })

    # 4D can have only one reviewed consolidation outcome per candidate.
    out = out.drop_duplicates(subset=["candidate_id"], keep="first")
    return out


# ---------------------------------------------------------------------
# Build complete raw-label normalization backbone
# ---------------------------------------------------------------------

def build_backbone(
    membership: pd.DataFrame,
    proposals: pd.DataFrame,
    human: pd.DataFrame,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    b = membership.merge(proposals, on="candidate_id", how="left", validate="m:1")
    b = b.merge(human, on="candidate_id", how="left", validate="m:1")

    rows = []

    for _, r in b.iterrows():
        source = txt(r.get("inventory_source")) or txt(r.get("proposal_inventory_source"))
        role = txt(r.get("annotation_role")) or txt(r.get("proposal_annotation_role"))
        original_category = txt(r.get("taxonomy_category"))
        proposal_category = txt(r.get("proposal_category"))
        pdecision = key(r.get("proposal_decision"))

        has_human = not pd.isna(r.get("active_in_v02"))
        active_human = bool(r.get("active_in_v02")) if has_human else False

        canonical_concept = ""
        canonical_category = ""
        ontology_node_id = ""
        resolution_status = ""
        resolution_source = ""

        if has_human:
            # Human review is authoritative.
            if active_human:
                canonical_concept = txt(r.get("human_canonical_concept"))
                canonical_category = txt(r.get("human_canonical_category"))
                ontology_node_id = txt(r.get("human_ontology_node_id"))

                if not ontology_node_id:
                    nid, cc, cat, status = resolve_node(
                        nodes, canonical_concept, canonical_category, source
                    )
                    ontology_node_id = nid
                    canonical_concept = cc
                    canonical_category = cat
                    resolution_status = status
                else:
                    resolution_status = "resolved"

                resolution_source = "4d_human_review"
            else:
                resolution_status = "human_defer_or_reject"
                resolution_source = "4d_human_review"

        elif pdecision == "existing_node":
            # Existing v0.1/v0.2 node — matched_existing_node is preferred.
            target = txt(r.get("matched_existing_node")) or txt(r.get("proposal_label"))
            nid, cc, cat, status = resolve_node(
                nodes,
                target,
                proposal_category or original_category,
                source,
            )
            ontology_node_id = nid
            canonical_concept = cc if nid else ""
            canonical_category = cat if nid else ""
            resolution_status = status if nid else f"existing_node_{status}"
            resolution_source = "4b2_existing_node"

        elif pdecision == "new_node":
            # Only reviewed + approved new nodes enter v0.2.
            resolution_status = "unreviewed_new_node"
            resolution_source = "4b2_proposal"

        elif pdecision == "ambiguous":
            resolution_status = "proposal_ambiguous"
            resolution_source = "4b2_proposal"

        elif pdecision == "reject":
            resolution_status = "proposal_reject"
            resolution_source = "4b2_proposal"

        elif not pdecision:
            resolution_status = "missing_proposal"
            resolution_source = "4b2_proposal"

        else:
            resolution_status = f"proposal_{pdecision}"
            resolution_source = "4b2_proposal"

        rows.append({
            "candidate_id": txt(r["candidate_id"]),
            "raw_label": txt(r["raw_label"]),
            "raw_label_key": txt(r["raw_label_key"]),
            "inventory_source": source,
            "annotation_role": role,
            "original_taxonomy_category": original_category,
            "proposal_label": txt(r.get("proposal_label")),
            "proposal_category": proposal_category,
            "proposal_decision": txt(r.get("proposal_decision")),
            "matched_existing_node": txt(r.get("matched_existing_node")),
            "canonical_concept": canonical_concept,
            "canonical_category": canonical_category,
            "ontology_node_id": ontology_node_id,
            "resolution_status": resolution_status,
            "resolution_source": resolution_source,
            "is_mappable": bool(ontology_node_id and canonical_concept),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Annotation standardization + matching
# ---------------------------------------------------------------------

def standardize_annotations(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    content_col = first_col(df, ["content_id", "video_id", "target_content_id"])
    label_col = first_col(df, ["raw_label", "label", "taxonomy_label", "extracted_label"])
    category_col = first_col(df, ["taxonomy_category", "category", "raw_category"])
    role_col = first_col(df, ["annotation_role", "role", "source_type"])
    conf_col = first_col(df, ["confidence", "annotation_confidence", "model_confidence"])
    ann_col = first_col(df, ["annotation_id", "id"])
    comment_col = first_col(df, ["comment_id", "target_comment_id"])
    evidence_col = first_col(df, ["evidence_span", "evidence", "supporting_evidence"])

    if not content_col or not label_col:
        raise ValueError(
            f"{dataset}: could not detect content_id/raw_label. Columns={list(df.columns)}"
        )

    out = pd.DataFrame({
        "annotation_id": df[ann_col].map(txt) if ann_col else "",
        "content_id": df[content_col].map(txt),
        "comment_id": df[comment_col].map(txt) if comment_col else "",
        "dataset": dataset,
        "raw_annotation_role": df[role_col].map(txt) if role_col else "",
        "raw_category": df[category_col].map(txt) if category_col else "",
        "raw_label": df[label_col].map(txt),
        "confidence": df[conf_col].map(safe_float) if conf_col else np.nan,
        "evidence_span": df[evidence_col].map(txt) if evidence_col else "",
    })
    out["annotation_role"] = out["raw_annotation_role"].map(
        lambda x: normalize_role(x, dataset)
    )
    if dataset == "3b":
        out["annotation_role"] = "audience_perception"

    out["raw_label_key"] = out["raw_label"].map(key)
    missing = out["annotation_id"].eq("")
    out.loc[missing, "annotation_id"] = [
        f"{dataset}_{i:06d}" for i in out.index[missing]
    ]
    return out


def match_annotation(row: pd.Series, backbone: pd.DataFrame) -> Dict[str, Any]:
    cand = backbone[backbone["raw_label_key"] == row["raw_label_key"]].copy()
    if cand.empty:
        return {
            "normalization_status": "raw_label_not_in_candidate_backbone",
            "mapping_match_type": "",
            "canonical_concept": "",
            "canonical_category": "",
            "ontology_node_id": "",
            "candidate_id": "",
            "resolution_source": "",
        }

    dataset = row["dataset"]
    role = key(row["annotation_role"])
    cat = key(row["raw_category"])

    # Source filter first.
    side = cand["inventory_source"].map(infer_side)
    c = cand[(side == dataset) | side.eq("")]
    if not c.empty:
        cand = c

    # Role filter where available.
    role_keys = cand["annotation_role"].map(key)
    c = cand[(role_keys == role) | role_keys.eq("")]
    if not c.empty:
        cand = c

    # Original category filter where available.
    cat_keys = cand["original_taxonomy_category"].map(key)
    c = cand[(cat_keys == cat) | cat_keys.eq("")]
    if not c.empty:
        cand = c

    # If duplicate candidate rows resolve to the same outcome, that's safe.
    outcome_cols = [
        "canonical_concept", "canonical_category", "ontology_node_id",
        "resolution_status", "resolution_source"
    ]
    unique_outcomes = cand[outcome_cols].drop_duplicates()

    if len(unique_outcomes) > 1:
        # Prefer mappable outcome if exactly one exists.
        mappable = cand[cand["is_mappable"]]
        mo = mappable[outcome_cols].drop_duplicates()
        if len(mo) == 1:
            cand = mappable
        else:
            return {
                "normalization_status": "ambiguous_raw_label_mapping",
                "mapping_match_type": "raw+source+role+category",
                "canonical_concept": "",
                "canonical_category": "",
                "ontology_node_id": "",
                "candidate_id": " | ".join(sorted(cand["candidate_id"].unique())),
                "resolution_source": "",
            }

    hit = cand.iloc[0]

    if not bool(hit["is_mappable"]):
        return {
            "normalization_status": txt(hit["resolution_status"]),
            "mapping_match_type": "raw+source+role+category",
            "canonical_concept": "",
            "canonical_category": "",
            "ontology_node_id": "",
            "candidate_id": txt(hit["candidate_id"]),
            "resolution_source": txt(hit["resolution_source"]),
        }

    return {
        "normalization_status": "mapped",
        "mapping_match_type": "raw+source+role+category",
        "canonical_concept": txt(hit["canonical_concept"]),
        "canonical_category": txt(hit["canonical_category"]),
        "ontology_node_id": txt(hit["ontology_node_id"]),
        "candidate_id": txt(hit["candidate_id"]),
        "resolution_source": txt(hit["resolution_source"]),
    }


def normalize_annotations(annotations: pd.DataFrame, backbone: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in annotations.iterrows():
        d = r.to_dict()
        d.update(match_annotation(r, backbone))
        rows.append(d)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------

def build_concept_profile(norm: pd.DataFrame) -> pd.DataFrame:
    m = norm[norm["normalization_status"] == "mapped"].copy()
    if m.empty:
        return pd.DataFrame()

    m["confidence_filled"] = m["confidence"].fillna(1.0)
    g = (
        m.groupby(
            [
                "content_id", "annotation_role", "canonical_category",
                "canonical_concept", "ontology_node_id"
            ],
            dropna=False,
        )
        .agg(
            annotation_count=("annotation_id", "count"),
            unique_comment_count=("comment_id", lambda s: s[s.map(txt).ne("")].nunique()),
            avg_confidence=("confidence_filled", "mean"),
            max_confidence=("confidence_filled", "max"),
        )
        .reset_index()
    )

    # Audience denominator here = comments that produced at least one mapped signal.
    aud_den = (
        m[m["annotation_role"] == "audience_perception"]
        .groupby("content_id")["comment_id"]
        .apply(lambda s: s[s.map(txt).ne("")].nunique())
        .to_dict()
    )

    g["audience_mapped_signal_support_rate"] = g.apply(
        lambda r: (
            r["unique_comment_count"] / aud_den.get(r["content_id"], 0)
            if r["annotation_role"] == "audience_perception"
            and aud_den.get(r["content_id"], 0) > 0
            else np.nan
        ),
        axis=1,
    )

    g["concept_rank_within_role"] = (
        g.groupby(["content_id", "annotation_role"])["annotation_count"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    return g.sort_values(
        ["content_id", "annotation_role", "annotation_count", "avg_confidence"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def build_role_profile(cp: pd.DataFrame) -> pd.DataFrame:
    if cp.empty:
        return pd.DataFrame()

    rows = []
    for (cid, role), sub in cp.groupby(["content_id", "annotation_role"]):
        sub = sub.sort_values(
            ["annotation_count", "avg_confidence"],
            ascending=[False, False],
        )
        concepts = []
        for _, r in sub.iterrows():
            concepts.append({
                "concept": r["canonical_concept"],
                "category": r["canonical_category"],
                "node_id": r["ontology_node_id"],
                "annotation_count": int(r["annotation_count"]),
                "unique_comment_count": int(r["unique_comment_count"]),
                "avg_confidence": round(float(r["avg_confidence"]), 4),
                "audience_mapped_signal_support_rate": (
                    None if pd.isna(r["audience_mapped_signal_support_rate"])
                    else round(float(r["audience_mapped_signal_support_rate"]), 4)
                ),
            })

        rows.append({
            "content_id": cid,
            "annotation_role": role,
            "unique_concept_count": sub["canonical_concept"].nunique(),
            "unique_category_count": sub["canonical_category"].nunique(),
            "total_annotation_count": int(sub["annotation_count"].sum()),
            "concept_list": " | ".join(sub["canonical_concept"].tolist()),
            "concepts_json": json.dumps(concepts, ensure_ascii=False),
        })

    return pd.DataFrame(rows).sort_values(
        ["content_id", "annotation_role"]
    ).reset_index(drop=True)


def build_audience_profile(norm: pd.DataFrame, audience_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    m = norm[
        (norm["normalization_status"] == "mapped")
        & (norm["annotation_role"] == "audience_perception")
    ].copy()

    if m.empty:
        return pd.DataFrame()

    totals: Dict[str, int] = {}
    if audience_df is not None and not audience_df.empty:
        ccol = first_col(audience_df, ["content_id", "video_id"])
        idcol = first_col(audience_df, ["comment_id", "id"])
        if ccol and idcol:
            totals = (
                audience_df.groupby(ccol)[idcol].nunique().to_dict()
            )

    rows = []
    for (cid, cat, concept, nid), sub in m.groupby(
        ["content_id", "canonical_category", "canonical_concept", "ontology_node_id"],
        dropna=False,
    ):
        uc = sub["comment_id"][sub["comment_id"].map(txt).ne("")].nunique()
        total = totals.get(cid, np.nan)
        rows.append({
            "content_id": cid,
            "canonical_category": cat,
            "canonical_concept": concept,
            "ontology_node_id": nid,
            "annotation_count": len(sub),
            "unique_comment_count": uc,
            "total_content_comment_count": total,
            "comment_prevalence": uc / total if not pd.isna(total) and total > 0 else np.nan,
            "avg_confidence": sub["confidence"].mean(),
            "max_confidence": sub["confidence"].max(),
        })

    out = pd.DataFrame(rows)
    out["rank_within_content"] = (
        out.groupby("content_id")["unique_comment_count"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    return out.sort_values(
        ["content_id", "unique_comment_count", "avg_confidence"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_jsonl(role_profile: pd.DataFrame, content_df: Optional[pd.DataFrame], path: Path) -> None:
    metadata = {}
    if content_df is not None and not content_df.empty:
        ccol = first_col(content_df, ["content_id", "video_id"])
        if ccol:
            for _, r in content_df.iterrows():
                cid = txt(r[ccol])
                metadata[cid] = {
                    k: (None if pd.isna(v) else v)
                    for k, v in r.to_dict().items()
                }

    profiles = defaultdict(dict)
    for _, r in role_profile.iterrows():
        profiles[r["content_id"]][r["annotation_role"]] = {
            "unique_concept_count": int(r["unique_concept_count"]),
            "unique_category_count": int(r["unique_category_count"]),
            "total_annotation_count": int(r["total_annotation_count"]),
            "concepts": json.loads(r["concepts_json"]),
        }

    empty = {
        "unique_concept_count": 0,
        "unique_category_count": 0,
        "total_annotation_count": 0,
        "concepts": [],
    }

    ids = sorted(set(profiles) | set(metadata))
    with path.open("w", encoding="utf-8") as f:
        for cid in ids:
            rec = {
                "content_id": cid,
                "metadata": metadata.get(cid, {}),
                "profiles": {
                    "inferred_content_intent": profiles.get(cid, {}).get("inferred_content_intent", empty),
                    "ai_content_understanding": profiles.get(cid, {}).get("ai_content_understanding", empty),
                    "audience_perception": profiles.get(cid, {}).get("audience_perception", empty),
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# CLI / run
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="5A v2: full ontology-v0.2 normalization backbone + content profiles")
    p.add_argument("--annotations-3a", type=Path, required=True)
    p.add_argument("--annotations-3b", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True,
                   help="Full 4B semantic candidate CSV (must contain candidate_id + member raw labels)")
    p.add_argument("--proposals", type=Path, required=True,
                   help="Full 4B.2 LLM proposal CSV, one row per candidate_id")
    p.add_argument("--mapping-4d", type=Path, required=True,
                   help="4D taxonomy_normalization_mapping_v02.csv")
    p.add_argument("--ontology-nodes", type=Path, required=True)
    p.add_argument("--content", type=Path)
    p.add_argument("--audience", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nodes = load_nodes(args.ontology_nodes)
    membership = load_candidate_membership(args.candidates)
    proposals = load_proposals(args.proposals)
    human = load_human_mapping(args.mapping_4d)

    backbone = build_backbone(membership, proposals, human, nodes)
    backbone.to_csv(args.output_dir / "normalization_backbone_5a.csv", index=False)

    a3 = standardize_annotations(pd.read_csv(args.annotations_3a), "3a")
    a3b = standardize_annotations(pd.read_csv(args.annotations_3b), "3b")
    annotations = pd.concat([a3, a3b], ignore_index=True)

    norm = normalize_annotations(annotations, backbone)
    norm.to_csv(args.output_dir / "normalized_annotations_5a.csv", index=False)

    cp = build_concept_profile(norm)
    rp = build_role_profile(cp)

    audience_df = pd.read_csv(args.audience) if args.audience else None
    ap = build_audience_profile(norm, audience_df)

    cp.to_csv(args.output_dir / "content_concept_profile_5a.csv", index=False)
    rp.to_csv(args.output_dir / "content_role_profile_5a.csv", index=False)
    ap.to_csv(args.output_dir / "audience_concept_profile_5a.csv", index=False)

    content_df = pd.read_csv(args.content) if args.content else None
    build_jsonl(rp, content_df, args.output_dir / "alignment_input_profile_5a.jsonl")

    unmapped = norm[norm["normalization_status"] != "mapped"].copy()
    unmapped.to_csv(args.output_dir / "normalization_unmapped_5a.csv", index=False)

    # QC by role + status.
    qc = (
        norm.groupby(["annotation_role", "normalization_status"], dropna=False)
        .size()
        .reset_index(name="row_count")
    )
    qc.to_csv(args.output_dir / "normalization_qc_5a.csv", index=False)

    mapped = norm["normalization_status"].eq("mapped")
    summary = pd.DataFrame([
        {"metric": "3a_annotation_rows", "value": len(a3)},
        {"metric": "3b_annotation_rows", "value": len(a3b)},
        {"metric": "total_annotation_rows", "value": len(norm)},
        {"metric": "mapped_annotation_rows", "value": int(mapped.sum())},
        {"metric": "unmapped_or_unapproved_rows", "value": int((~mapped).sum())},
        {"metric": "overall_mapping_rate", "value": round(float(mapped.mean()), 6)},
        {"metric": "content_count", "value": norm["content_id"].nunique()},
        {"metric": "canonical_concept_count", "value": norm.loc[mapped, "canonical_concept"].nunique()},
        {"metric": "normalization_backbone_raw_label_rows", "value": len(backbone)},
        {"metric": "normalization_backbone_mappable_rows", "value": int(backbone["is_mappable"].sum())},
        {"metric": "content_concept_profile_rows", "value": len(cp)},
        {"metric": "content_role_profile_rows", "value": len(rp)},
        {"metric": "audience_concept_profile_rows", "value": len(ap)},
    ])
    summary.to_csv(args.output_dir / "build_summary_5a.csv", index=False)

    print("\n=== 5A v2 normalized profile build complete ===")
    print(summary.to_string(index=False))

    print("\nNormalization status:")
    print(norm["normalization_status"].value_counts(dropna=False).to_string())

    print("\nResolution source among mapped rows:")
    if mapped.any():
        print(norm.loc[mapped, "resolution_source"].value_counts(dropna=False).to_string())
    else:
        print("(none)")

    print("\nBackbone resolution status:")
    print(backbone["resolution_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
