#!/usr/bin/env python3
"""
Deliverable 5B — Cross-Ontology Relationship & Concept Pairing Engine

Purpose
-------
Take the governed canonical profiles produced by 5A and build interpretable
concept-to-concept relationship candidates for the three alignment paths:

1) inferred_content_intent  <-> ai_content_understanding
2) ai_content_understanding <-> audience_perception
3) inferred_content_intent  <-> audience_perception

This script does NOT create the final 0–100 alignment score. That belongs to 5C.
5B answers a more fundamental question:

    "How are two governed canonical concepts related?"

Relationship evidence is intentionally separated into:
- exact concept identity
- curated cross-ontology/domain relationship
- semantic similarity
- no supported relationship

Important guardrail
-------------------
Missing profile != zero alignment.

If either side of a comparison is unavailable, the comparison is emitted as:
    alignment_computable = False
    alignment_status = insufficient_normalized_signal

Inputs
------
Required:
  --normalized-annotations  5A normalized_annotations_5a.csv
  --concept-profile         5A content_concept_profile_5a.csv
  --role-profile            5A content_role_profile_5a.csv
  --ontology-nodes          ontology_v02_nodes.csv

Optional:
  --relationship-rules      Curated relationship CSV.
                            If omitted, a starter file is generated automatically.
  --model                   sentence-transformers model name
  --semantic-threshold      minimum cosine similarity for semantic relation
  --strong-threshold        threshold for strong semantic relation

Outputs
-------
relationship_rules_5b.csv
concept_pair_candidates_5b.csv
content_comparison_availability_5b.csv
content_pair_summary_5b.csv
concept_relationship_catalog_5b.csv
relationship_review_queue_5b.csv
build_summary_5b.csv

Starter curated rules
---------------------
The default rules encode only relationships already established during
ontology design discussions. They are deliberately small and auditable:

Playful Camaraderie -> Playful Teasing Dynamic
    manifests_as

Vocal Production Craftsmanship -> Vocal Ability
    contributes_to

The user can extend relationship_rules_5b.csv later without changing code.

Example
-------
python build_cross_ontology_relationships_5b.py \
  --normalized-annotations data/alignment/5a/normalized_annotations_5a.csv \
  --concept-profile data/alignment/5a/content_concept_profile_5a.csv \
  --role-profile data/alignment/5a/content_role_profile_5a.csv \
  --ontology-nodes data/ontology/v0.2/ontology_v02_nodes.csv \
  --output-dir data/alignment/5b
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

COMPARISONS = [
    (
        "intent_to_content",
        "inferred_content_intent",
        "ai_content_understanding",
    ),
    (
        "content_to_audience",
        "ai_content_understanding",
        "audience_perception",
    ),
    (
        "intent_to_audience",
        "inferred_content_intent",
        "audience_perception",
    ),
]

DEFAULT_RELATIONSHIP_RULES = [
    {
        "source_concept": "Playful Camaraderie",
        "source_category": "Relationship Style",
        "source_role": "inferred_content_intent",
        "target_concept": "Playful Teasing Dynamic",
        "target_category": "Relationship Perception",
        "target_role": "audience_perception",
        "relationship_type": "manifests_as",
        "relationship_strength": 0.90,
        "directional": True,
        "rule_status": "approved",
        "rule_source": "ontology_v0.2_human_governance",
        "notes": "Creator-side playful camaraderie can manifest in audience perception as a playful teasing dynamic.",
    },
    {
        "source_concept": "Playful Camaraderie",
        "source_category": "Relationship Style",
        "source_role": "ai_content_understanding",
        "target_concept": "Playful Teasing Dynamic",
        "target_category": "Relationship Perception",
        "target_role": "audience_perception",
        "relationship_type": "manifests_as",
        "relationship_strength": 0.90,
        "directional": True,
        "rule_status": "approved",
        "rule_source": "ontology_v0.2_human_governance",
        "notes": "Content-level playful camaraderie can manifest in audience perception as a playful teasing dynamic.",
    },
    {
        "source_concept": "Vocal Production Craftsmanship",
        "source_category": "Creative Style",
        "source_role": "inferred_content_intent",
        "target_concept": "Vocal Ability",
        "target_category": "Creator Perception",
        "target_role": "audience_perception",
        "relationship_type": "contributes_to",
        "relationship_strength": 0.85,
        "directional": True,
        "rule_status": "approved",
        "rule_source": "ontology_v0.2_human_governance",
        "notes": "Creator-side vocal production craftsmanship can contribute to audience perception of vocal ability.",
    },
    {
        "source_concept": "Vocal Production Craftsmanship",
        "source_category": "Creative Style",
        "source_role": "ai_content_understanding",
        "target_concept": "Vocal Ability",
        "target_category": "Creator Perception",
        "target_role": "audience_perception",
        "relationship_type": "contributes_to",
        "relationship_strength": 0.85,
        "directional": True,
        "rule_status": "approved",
        "rule_source": "ontology_v0.2_human_governance",
        "notes": "Content-level vocal production craftsmanship can contribute to audience perception of vocal ability.",
    },
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

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


def first_col(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    for c in names:
        if c in df.columns:
            return c
    return None


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def boolish(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return key(x) in {"1", "true", "yes", "y", "t"}


def clipped(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def relation_priority(relation_type: str) -> int:
    order = {
        "exact": 4,
        "manifests_as": 3,
        "contributes_to": 3,
        "related": 3,
        "semantic_strong": 2,
        "semantic": 1,
        "none": 0,
    }
    return order.get(key(relation_type), 0)


# ---------------------------------------------------------------------
# Input standardization
# ---------------------------------------------------------------------

def load_concept_profile(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = [
        "content_id",
        "annotation_role",
        "canonical_category",
        "canonical_concept",
        "ontology_node_id",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Concept profile missing required columns {missing}. "
            f"Columns={list(df.columns)}"
        )

    out = df.copy()
    out["content_id"] = out["content_id"].map(txt)
    out["annotation_role"] = out["annotation_role"].map(txt)
    out["canonical_category"] = out["canonical_category"].map(txt)
    out["canonical_concept"] = out["canonical_concept"].map(txt)
    out["ontology_node_id"] = out["ontology_node_id"].map(txt)

    for c in [
        "annotation_count",
        "unique_comment_count",
        "avg_confidence",
        "max_confidence",
        "audience_mapped_signal_support_rate",
        "concept_rank_within_role",
    ]:
        if c not in out.columns:
            out[c] = np.nan

    return out


def load_role_profile(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["content_id", "annotation_role"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Role profile missing {missing}. Columns={list(df.columns)}")

    df["content_id"] = df["content_id"].map(txt)
    df["annotation_role"] = df["annotation_role"].map(txt)
    return df


def load_normalized_annotations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["content_id", "annotation_role", "normalization_status"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Normalized annotations missing {missing}. Columns={list(df.columns)}"
        )
    df["content_id"] = df["content_id"].map(txt)
    df["annotation_role"] = df["annotation_role"].map(txt)
    return df


def load_ontology_nodes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    node_col = first_col(df, ["ontology_node_id", "node_id", "id"])
    concept_col = first_col(df, ["canonical_concept", "concept", "node_label", "label"])
    category_col = first_col(df, ["canonical_category", "taxonomy_category", "category"])

    if not node_col or not concept_col or not category_col:
        raise ValueError(
            "Ontology nodes must contain node id, concept, and category. "
            f"Columns={list(df.columns)}"
        )

    return pd.DataFrame({
        "ontology_node_id": df[node_col].map(txt),
        "canonical_concept": df[concept_col].map(txt),
        "canonical_category": df[category_col].map(txt),
    }).drop_duplicates()


# ---------------------------------------------------------------------
# Relationship rules
# ---------------------------------------------------------------------

RULE_COLUMNS = [
    "source_concept",
    "source_category",
    "source_role",
    "target_concept",
    "target_category",
    "target_role",
    "relationship_type",
    "relationship_strength",
    "directional",
    "rule_status",
    "rule_source",
    "notes",
]


def write_starter_rules(path: Path) -> None:
    df = pd.DataFrame(DEFAULT_RELATIONSHIP_RULES, columns=RULE_COLUMNS)
    df.to_csv(path, index=False)


def load_rules(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    for c in RULE_COLUMNS:
        if c not in df.columns:
            if c == "relationship_strength":
                df[c] = 0.80
            elif c == "directional":
                df[c] = True
            elif c == "rule_status":
                df[c] = "approved"
            else:
                df[c] = ""

    df["relationship_strength"] = df["relationship_strength"].map(
        lambda x: clipped(safe_float(x, 0.80))
    )
    df["directional"] = df["directional"].map(boolish)
    df["rule_status"] = df["rule_status"].map(txt)

    # Only approved rules affect the engine.
    return df[df["rule_status"].map(key) == "approved"].copy()


def rule_matches(
    rule: pd.Series,
    source_role: str,
    source_concept: str,
    source_category: str,
    target_role: str,
    target_concept: str,
    target_category: str,
) -> bool:
    def eq_or_blank(rule_value: Any, actual: str) -> bool:
        rv = key(rule_value)
        return (not rv) or rv == key(actual)

    forward = (
        eq_or_blank(rule["source_role"], source_role)
        and key(rule["source_concept"]) == key(source_concept)
        and eq_or_blank(rule["source_category"], source_category)
        and eq_or_blank(rule["target_role"], target_role)
        and key(rule["target_concept"]) == key(target_concept)
        and eq_or_blank(rule["target_category"], target_category)
    )
    if forward:
        return True

    # Non-directional rules can match in reverse.
    if not bool(rule["directional"]):
        reverse = (
            eq_or_blank(rule["source_role"], target_role)
            and key(rule["source_concept"]) == key(target_concept)
            and eq_or_blank(rule["source_category"], target_category)
            and eq_or_blank(rule["target_role"], source_role)
            and key(rule["target_concept"]) == key(source_concept)
            and eq_or_blank(rule["target_category"], source_category)
        )
        return reverse

    return False


def find_curated_rule(
    rules: pd.DataFrame,
    source_role: str,
    source_concept: str,
    source_category: str,
    target_role: str,
    target_concept: str,
    target_category: str,
) -> Optional[pd.Series]:
    if rules.empty:
        return None

    hits = []
    for _, r in rules.iterrows():
        if rule_matches(
            r,
            source_role,
            source_concept,
            source_category,
            target_role,
            target_concept,
            target_category,
        ):
            hits.append(r)

    if not hits:
        return None

    # Strongest approved curated relationship wins.
    hits = sorted(
        hits,
        key=lambda r: (
            relation_priority(txt(r["relationship_type"])),
            safe_float(r["relationship_strength"], 0.0),
        ),
        reverse=True,
    )
    return hits[0]


# ---------------------------------------------------------------------
# Semantic embeddings
# ---------------------------------------------------------------------

def build_embedding_text(category: str, concept: str) -> str:
    # Category context helps disambiguate short labels such as "Trust".
    return f"{txt(category)}: {txt(concept)}"


def compute_embeddings(
    concepts: pd.DataFrame,
    model_name: str,
) -> Tuple[Dict[Tuple[str, str], np.ndarray], str]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for 5B semantic relationships. "
            "Install with: pip install sentence-transformers"
        ) from e

    model = SentenceTransformer(model_name)

    unique = (
        concepts[["canonical_category", "canonical_concept"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    texts = [
        build_embedding_text(r["canonical_category"], r["canonical_concept"])
        for _, r in unique.iterrows()
    ]

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    lookup = {}
    for i, r in unique.iterrows():
        lookup[(key(r["canonical_category"]), key(r["canonical_concept"]))] = np.asarray(
            vectors[i], dtype=float
        )

    return lookup, model_name


def cosine_from_lookup(
    lookup: Dict[Tuple[str, str], np.ndarray],
    category_a: str,
    concept_a: str,
    category_b: str,
    concept_b: str,
) -> float:
    a = lookup.get((key(category_a), key(concept_a)))
    b = lookup.get((key(category_b), key(concept_b)))
    if a is None or b is None:
        return np.nan
    return clipped(float(np.dot(a, b)))


# ---------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------

def build_availability(
    normalized: pd.DataFrame,
    role_profile: pd.DataFrame,
) -> pd.DataFrame:
    all_content = sorted(
        set(normalized["content_id"].dropna().map(txt))
        | set(role_profile["content_id"].dropna().map(txt))
    )

    available_keys = set(
        zip(role_profile["content_id"], role_profile["annotation_role"])
    )

    rows = []
    for content_id in all_content:
        for comparison_name, source_role, target_role in COMPARISONS:
            source_available = (content_id, source_role) in available_keys
            target_available = (content_id, target_role) in available_keys

            if source_available and target_available:
                status = "ready"
                computable = True
            else:
                status = "insufficient_normalized_signal"
                computable = False

            rows.append({
                "content_id": content_id,
                "comparison_type": comparison_name,
                "source_role": source_role,
                "target_role": target_role,
                "source_profile_available": source_available,
                "target_profile_available": target_available,
                "alignment_computable": computable,
                "alignment_status": status,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------

def pair_relationship(
    source: pd.Series,
    target: pd.Series,
    rules: pd.DataFrame,
    embeddings: Dict[Tuple[str, str], np.ndarray],
    semantic_threshold: float,
    strong_threshold: float,
) -> Dict[str, Any]:

    s_concept = txt(source["canonical_concept"])
    s_category = txt(source["canonical_category"])
    s_role = txt(source["annotation_role"])

    t_concept = txt(target["canonical_concept"])
    t_category = txt(target["canonical_category"])
    t_role = txt(target["annotation_role"])

    exact = key(s_concept) == key(t_concept)

    semantic_similarity = cosine_from_lookup(
        embeddings,
        s_category,
        s_concept,
        t_category,
        t_concept,
    )

    curated = find_curated_rule(
        rules,
        s_role,
        s_concept,
        s_category,
        t_role,
        t_concept,
        t_category,
    )

    curated_type = ""
    curated_strength = np.nan
    curated_source = ""

    if curated is not None:
        curated_type = txt(curated["relationship_type"])
        curated_strength = safe_float(curated["relationship_strength"], 0.80)
        curated_source = txt(curated["rule_source"])

    # Evidence priority:
    # exact identity > curated domain relation > semantic similarity > none
    if exact:
        relationship_type = "exact"
        relationship_strength = 1.0
        relationship_source = "exact_concept_identity"
        supported = True

    elif curated is not None:
        relationship_type = curated_type or "related"
        relationship_strength = clipped(curated_strength)
        relationship_source = curated_source or "curated_rule"
        supported = True

    elif not pd.isna(semantic_similarity) and semantic_similarity >= strong_threshold:
        relationship_type = "semantic_strong"
        relationship_strength = semantic_similarity
        relationship_source = "semantic_model"
        supported = True

    elif not pd.isna(semantic_similarity) and semantic_similarity >= semantic_threshold:
        relationship_type = "semantic"
        relationship_strength = semantic_similarity
        relationship_source = "semantic_model"
        supported = True

    else:
        relationship_type = "none"
        relationship_strength = 0.0
        relationship_source = "no_supported_relation"
        supported = False

    # Concept evidence/support metrics from 5A.
    source_confidence = safe_float(source.get("avg_confidence"))
    target_confidence = safe_float(target.get("avg_confidence"))

    target_prevalence = safe_float(
        target.get("audience_mapped_signal_support_rate")
    )
    source_prevalence = safe_float(
        source.get("audience_mapped_signal_support_rate")
    )

    return {
        "exact_concept_match": exact,
        "semantic_similarity": semantic_similarity,
        "curated_relationship_type": curated_type,
        "curated_relationship_strength": curated_strength,
        "relationship_type": relationship_type,
        "relationship_strength": relationship_strength,
        "relationship_source": relationship_source,
        "relationship_supported": supported,
        "source_avg_confidence": source_confidence,
        "target_avg_confidence": target_confidence,
        "source_audience_support_rate": source_prevalence,
        "target_audience_support_rate": target_prevalence,
    }


def build_pair_candidates(
    concept_profile: pd.DataFrame,
    availability: pd.DataFrame,
    rules: pd.DataFrame,
    embeddings: Dict[Tuple[str, str], np.ndarray],
    semantic_threshold: float,
    strong_threshold: float,
) -> pd.DataFrame:

    rows = []

    ready = availability[availability["alignment_computable"]].copy()

    for _, a in ready.iterrows():
        content_id = a["content_id"]
        comparison_type = a["comparison_type"]
        source_role = a["source_role"]
        target_role = a["target_role"]

        source_df = concept_profile[
            (concept_profile["content_id"] == content_id)
            & (concept_profile["annotation_role"] == source_role)
        ]
        target_df = concept_profile[
            (concept_profile["content_id"] == content_id)
            & (concept_profile["annotation_role"] == target_role)
        ]

        for _, s in source_df.iterrows():
            for _, t in target_df.iterrows():
                rel = pair_relationship(
                    s,
                    t,
                    rules,
                    embeddings,
                    semantic_threshold,
                    strong_threshold,
                )

                rows.append({
                    "content_id": content_id,
                    "comparison_type": comparison_type,
                    "source_role": source_role,
                    "source_node_id": txt(s["ontology_node_id"]),
                    "source_category": txt(s["canonical_category"]),
                    "source_concept": txt(s["canonical_concept"]),
                    "source_annotation_count": safe_float(s.get("annotation_count"), 0),
                    "source_unique_comment_count": safe_float(s.get("unique_comment_count"), 0),
                    "target_role": target_role,
                    "target_node_id": txt(t["ontology_node_id"]),
                    "target_category": txt(t["canonical_category"]),
                    "target_concept": txt(t["canonical_concept"]),
                    "target_annotation_count": safe_float(t.get("annotation_count"), 0),
                    "target_unique_comment_count": safe_float(t.get("unique_comment_count"), 0),
                    **rel,
                })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    # Ranking is useful for 5C and review:
    # strongest relationship first, then target audience support, then confidence.
    out["target_support_for_sort"] = out["target_audience_support_rate"].fillna(-1)
    out["mean_pair_confidence"] = out[
        ["source_avg_confidence", "target_avg_confidence"]
    ].mean(axis=1, skipna=True)

    out["pair_rank_for_source"] = (
        out.sort_values(
            [
                "content_id",
                "comparison_type",
                "source_node_id",
                "relationship_supported",
                "relationship_strength",
                "target_support_for_sort",
                "mean_pair_confidence",
            ],
            ascending=[True, True, True, False, False, False, False],
        )
        .groupby(["content_id", "comparison_type", "source_node_id"])
        .cumcount()
        + 1
    )

    return out.drop(columns=["target_support_for_sort"])


# ---------------------------------------------------------------------
# Summaries / catalog / review queue
# ---------------------------------------------------------------------

def build_pair_summary(
    pairs: pd.DataFrame,
    availability: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, a in availability.iterrows():
        cid = a["content_id"]
        ctype = a["comparison_type"]

        if not bool(a["alignment_computable"]):
            rows.append({
                "content_id": cid,
                "comparison_type": ctype,
                "source_role": a["source_role"],
                "target_role": a["target_role"],
                "alignment_computable": False,
                "alignment_status": a["alignment_status"],
                "source_concept_count": 0,
                "target_concept_count": 0,
                "pair_count": 0,
                "supported_pair_count": 0,
                "exact_pair_count": 0,
                "curated_pair_count": 0,
                "semantic_pair_count": 0,
                "unrelated_pair_count": 0,
                "source_concepts_with_supported_match": 0,
                "source_concept_match_coverage": np.nan,
                "target_concepts_with_supported_match": 0,
                "target_concept_match_coverage": np.nan,
            })
            continue

        sub = pairs[
            (pairs["content_id"] == cid)
            & (pairs["comparison_type"] == ctype)
        ].copy()

        source_count = sub["source_node_id"].nunique()
        target_count = sub["target_node_id"].nunique()
        supported = sub[sub["relationship_supported"]]

        source_matched = supported["source_node_id"].nunique()
        target_matched = supported["target_node_id"].nunique()

        rows.append({
            "content_id": cid,
            "comparison_type": ctype,
            "source_role": a["source_role"],
            "target_role": a["target_role"],
            "alignment_computable": True,
            "alignment_status": "ready",
            "source_concept_count": source_count,
            "target_concept_count": target_count,
            "pair_count": len(sub),
            "supported_pair_count": len(supported),
            "exact_pair_count": int((sub["relationship_type"] == "exact").sum()),
            "curated_pair_count": int(
                sub["relationship_type"].isin(
                    ["manifests_as", "contributes_to", "related"]
                ).sum()
            ),
            "semantic_pair_count": int(
                sub["relationship_type"].isin(
                    ["semantic", "semantic_strong"]
                ).sum()
            ),
            "unrelated_pair_count": int((sub["relationship_type"] == "none").sum()),
            "source_concepts_with_supported_match": source_matched,
            "source_concept_match_coverage": (
                source_matched / source_count if source_count else np.nan
            ),
            "target_concepts_with_supported_match": target_matched,
            "target_concept_match_coverage": (
                target_matched / target_count if target_count else np.nan
            ),
        })

    return pd.DataFrame(rows)


def build_relationship_catalog(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()

    g = (
        pairs.groupby(
            [
                "source_role",
                "source_category",
                "source_concept",
                "source_node_id",
                "target_role",
                "target_category",
                "target_concept",
                "target_node_id",
                "relationship_type",
                "relationship_source",
                "relationship_supported",
            ],
            dropna=False,
        )
        .agg(
            content_support_count=("content_id", "nunique"),
            pair_occurrence_count=("content_id", "size"),
            avg_semantic_similarity=("semantic_similarity", "mean"),
            avg_relationship_strength=("relationship_strength", "mean"),
            avg_pair_confidence=("mean_pair_confidence", "mean"),
        )
        .reset_index()
    )

    return g.sort_values(
        [
            "relationship_supported",
            "content_support_count",
            "avg_relationship_strength",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_review_queue(
    catalog: pd.DataFrame,
    semantic_threshold: float,
    strong_threshold: float,
) -> pd.DataFrame:
    if catalog.empty:
        return pd.DataFrame()

    # Human review is most useful for recurring semantic-only relations.
    q = catalog[
        catalog["relationship_type"].isin(["semantic", "semantic_strong"])
    ].copy()

    q["review_priority"] = np.select(
        [
            (q["content_support_count"] >= 2)
            & (q["avg_semantic_similarity"] >= strong_threshold),
            (q["content_support_count"] >= 2),
            (q["avg_semantic_similarity"] >= strong_threshold),
        ],
        ["high", "medium", "medium"],
        default="low",
    )

    q["human_relationship_decision"] = ""
    q["human_relationship_type"] = ""
    q["human_relationship_strength"] = np.nan
    q["human_review_notes"] = ""
    q["human_review_status"] = "pending"

    priority_order = {"high": 0, "medium": 1, "low": 2}
    q["_p"] = q["review_priority"].map(priority_order)

    return q.sort_values(
        ["_p", "content_support_count", "avg_semantic_similarity"],
        ascending=[True, False, False],
    ).drop(columns="_p").reset_index(drop=True)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="5B: Cross-ontology relationship & concept pairing engine"
    )
    p.add_argument("--normalized-annotations", type=Path, required=True)
    p.add_argument("--concept-profile", type=Path, required=True)
    p.add_argument("--role-profile", type=Path, required=True)
    p.add_argument("--ontology-nodes", type=Path, required=True)
    p.add_argument("--relationship-rules", type=Path)
    p.add_argument(
        "--model",
        default="sentence-transformers/distiluse-base-multilingual-cased-v1",
    )
    p.add_argument("--semantic-threshold", type=float, default=0.60)
    p.add_argument("--strong-threshold", type=float, default=0.72)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not 0 <= args.semantic_threshold <= 1:
        raise ValueError("--semantic-threshold must be between 0 and 1")
    if not 0 <= args.strong_threshold <= 1:
        raise ValueError("--strong-threshold must be between 0 and 1")
    if args.strong_threshold < args.semantic_threshold:
        raise ValueError("--strong-threshold must be >= --semantic-threshold")

    normalized = load_normalized_annotations(args.normalized_annotations)
    concept_profile = load_concept_profile(args.concept_profile)
    role_profile = load_role_profile(args.role_profile)
    ontology_nodes = load_ontology_nodes(args.ontology_nodes)

    # Starter relationship rules are always written to the 5B output folder.
    starter_path = args.output_dir / "relationship_rules_5b.csv"

    if args.relationship_rules:
        rules = load_rules(args.relationship_rules)
        # Preserve a copy in the output directory for reproducibility.
        pd.read_csv(args.relationship_rules).to_csv(starter_path, index=False)
        rule_source_used = str(args.relationship_rules)
    else:
        write_starter_rules(starter_path)
        rules = load_rules(starter_path)
        rule_source_used = str(starter_path)

    # Only concepts actually observed in 5A need embeddings for MVP.
    observed = concept_profile[
        ["canonical_category", "canonical_concept"]
    ].drop_duplicates()

    embeddings, model_name = compute_embeddings(
        observed,
        args.model,
    )

    availability = build_availability(normalized, role_profile)
    availability.to_csv(
        args.output_dir / "content_comparison_availability_5b.csv",
        index=False,
    )

    pairs = build_pair_candidates(
        concept_profile,
        availability,
        rules,
        embeddings,
        args.semantic_threshold,
        args.strong_threshold,
    )
    pairs.to_csv(
        args.output_dir / "concept_pair_candidates_5b.csv",
        index=False,
    )

    summary = build_pair_summary(pairs, availability)
    summary.to_csv(
        args.output_dir / "content_pair_summary_5b.csv",
        index=False,
    )

    catalog = build_relationship_catalog(pairs)
    catalog.to_csv(
        args.output_dir / "concept_relationship_catalog_5b.csv",
        index=False,
    )

    review = build_review_queue(
        catalog,
        args.semantic_threshold,
        args.strong_threshold,
    )
    review.to_csv(
        args.output_dir / "relationship_review_queue_5b.csv",
        index=False,
    )

    computable = availability["alignment_computable"].sum()
    unavailable = (~availability["alignment_computable"]).sum()

    supported_pairs = (
        int(pairs["relationship_supported"].sum())
        if not pairs.empty else 0
    )
    exact_pairs = (
        int((pairs["relationship_type"] == "exact").sum())
        if not pairs.empty else 0
    )
    curated_pairs = (
        int(
            pairs["relationship_type"].isin(
                ["manifests_as", "contributes_to", "related"]
            ).sum()
        )
        if not pairs.empty else 0
    )
    semantic_pairs = (
        int(
            pairs["relationship_type"].isin(
                ["semantic", "semantic_strong"]
            ).sum()
        )
        if not pairs.empty else 0
    )

    build_summary = pd.DataFrame([
        {"metric": "content_count", "value": normalized["content_id"].nunique()},
        {"metric": "observed_canonical_concepts", "value": observed["canonical_concept"].nunique()},
        {"metric": "comparison_rows_expected", "value": len(availability)},
        {"metric": "comparison_rows_computable", "value": int(computable)},
        {"metric": "comparison_rows_insufficient_signal", "value": int(unavailable)},
        {"metric": "concept_pair_rows", "value": len(pairs)},
        {"metric": "supported_pair_rows", "value": supported_pairs},
        {"metric": "exact_pair_rows", "value": exact_pairs},
        {"metric": "curated_pair_rows", "value": curated_pairs},
        {"metric": "semantic_pair_rows", "value": semantic_pairs},
        {"metric": "relationship_catalog_rows", "value": len(catalog)},
        {"metric": "relationship_review_queue_rows", "value": len(review)},
        {"metric": "approved_curated_rule_count", "value": len(rules)},
        {"metric": "semantic_threshold", "value": args.semantic_threshold},
        {"metric": "strong_threshold", "value": args.strong_threshold},
        {"metric": "semantic_model", "value": model_name},
        {"metric": "relationship_rules_used", "value": rule_source_used},
    ])

    build_summary.to_csv(
        args.output_dir / "build_summary_5b.csv",
        index=False,
    )

    print("\n=== 5B relationship & concept pairing build complete ===")
    print(build_summary.to_string(index=False))

    print("\nComparison availability:")
    print(
        availability.groupby(
            ["comparison_type", "alignment_status"]
        ).size().to_string()
    )

    if not pairs.empty:
        print("\nRelationship types:")
        print(pairs["relationship_type"].value_counts().to_string())

        print("\nSupported relationship source:")
        print(
            pairs.loc[
                pairs["relationship_supported"],
                "relationship_source",
            ].value_counts().to_string()
        )

        print("\nTop semantic-only relationship candidates:")
        sem = catalog[
            catalog["relationship_type"].isin(
                ["semantic", "semantic_strong"]
            )
        ].head(20)

        if sem.empty:
            print("(none)")
        else:
            cols = [
                "source_role",
                "source_concept",
                "target_role",
                "target_concept",
                "relationship_type",
                "avg_semantic_similarity",
                "content_support_count",
            ]
            print(sem[cols].to_string(index=False))

    print(
        "\nNOTE: 5B creates relationship evidence and pairing candidates only. "
        "Do not interpret relationship_strength as the final Alignment Score."
    )


if __name__ == "__main__":
    main()
