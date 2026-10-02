#!/usr/bin/env python3
"""
4D — Finalize Ontology v0.2

Purpose
-------
Turn the completed ontology human-review table into a versioned ontology and
normalization/provenance artifacts.

Key behaviors
-------------
1. Validate human review completion and decision consistency.
2. Keep v0.1 seed ontology as the source-of-truth base.
3. Add approved new concepts only.
4. Deduplicate peer-approved concepts using:
      inventory_source + human_final_category + human_final_concept
5. Exclude defer/reject from v0.2 additions, but retain them in the change log.
6. Support merge_existing_node when present.
7. Produce review-level and candidate-level canonical mappings.
8. Optionally enrich candidate mappings with raw-label information if the
   4B candidate file is supplied.

Expected review columns
-----------------------
consolidation_id
inventory_source
supporting_candidate_ids
human_final_concept
human_ontology_decision
human_final_category
human_merge_target_existing_node
human_review_notes
human_review_status

Seed ontology
-------------
Preferred: point --seed-python to the existing script that defines the exact
CREATOR_ONTOLOGY and AUDIENCE_ONTOLOGY dictionaries used in 4B.2.
This avoids silently recreating or changing v0.1 by hand.

Example
-------
python finalize_ontology_v02.py \
  --review data/evaluation/ontology_v02_human_review_complete.csv \
  --seed-python propose_taxonomy_normalization_candidates.py \
  --candidates data/processed/taxonomy_normalization_candidates.csv \
  --output-dir data/ontology/v0.2
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


VALID_DECISIONS = {
    "approve_new_node",
    "merge_existing_node",
    "reject",
    "defer",
}

APPROVED_DECISIONS = {"approve_new_node", "merge_existing_node"}

REQUIRED_REVIEW_COLUMNS = {
    "consolidation_id",
    "inventory_source",
    "supporting_candidate_ids",
    "human_final_concept",
    "human_ontology_decision",
    "human_final_category",
    "human_merge_target_existing_node",
    "human_review_notes",
    "human_review_status",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def norm_key(value: Any) -> str:
    """Case/space-insensitive comparison key; does not change display text."""
    return re.sub(r"\s+", " ", clean_text(value)).casefold()


def slugify(value: str) -> str:
    s = clean_text(value).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "unknown"


def split_pipe(value: Any) -> List[str]:
    text = clean_text(value)
    if not text:
        return []
    return [x.strip() for x in text.split("|") if x.strip()]


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        v = clean_text(value)
        if not v:
            continue
        k = norm_key(v)
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out


def infer_side(inventory_source: str) -> str:
    s = norm_key(inventory_source)
    if s.startswith("3a") or "content" in s:
        return "creator_content"
    if s.startswith("3b") or "audience" in s:
        return "audience"
    return slugify(inventory_source)


def canonical_node_id(inventory_source: str, category: str, concept: str) -> str:
    """Deterministic semantic ID; intentionally not version-numbered."""
    side = infer_side(inventory_source)
    return f"{slugify(side)}.{slugify(category)}.{slugify(concept)}"


def warn(messages: List[Dict[str, str]], code: str, message: str, row_id: str = "") -> None:
    messages.append({"level": "warning", "code": code, "row_id": row_id, "message": message})


def error(messages: List[Dict[str, str]], code: str, message: str, row_id: str = "") -> None:
    messages.append({"level": "error", "code": code, "row_id": row_id, "message": message})


# ---------------------------------------------------------------------------
# Load v0.1 seed ontology from exact project source-of-truth
# ---------------------------------------------------------------------------

def _literal_assignment_from_python(path: Path, variable_name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = []
            value_node = None
            if isinstance(node, ast.Assign):
                targets = node.targets
                value_node = node.value
            else:
                targets = [node.target]
                value_node = node.value

            for target in targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    try:
                        return ast.literal_eval(value_node)
                    except Exception as exc:
                        raise ValueError(
                            f"{variable_name} exists in {path}, but is not a literal "
                            f"Python structure that can be safely parsed: {exc}"
                        ) from exc
    raise KeyError(f"Could not find {variable_name} in {path}")


def load_seed_from_python(path: Path) -> Dict[str, Any]:
    creator = _literal_assignment_from_python(path, "CREATOR_ONTOLOGY")
    audience = _literal_assignment_from_python(path, "AUDIENCE_ONTOLOGY")
    return {
        "version": "0.1",
        "creator_content": creator,
        "audience": audience,
    }


def load_seed_from_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def normalize_seed_structure(seed: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    """
    Normalize several plausible seed JSON/dict shapes into:
      {"creator_content": {category: [concepts]}, "audience": {...}}
    """
    creator = (
        seed.get("creator_content")
        or seed.get("creator")
        or seed.get("CREATOR_ONTOLOGY")
        or {}
    )
    audience = seed.get("audience") or seed.get("AUDIENCE_ONTOLOGY") or {}

    def normalize_side(side: Any) -> Dict[str, List[str]]:
        if not isinstance(side, dict):
            raise ValueError("Ontology side must be a dictionary of category -> concepts")
        out: Dict[str, List[str]] = {}
        for category, concepts in side.items():
            cat = clean_text(category)
            if isinstance(concepts, dict):
                # Support {node_id: label} or {label: metadata} shapes.
                vals = []
                for k, v in concepts.items():
                    if isinstance(v, str):
                        vals.append(v)
                    elif isinstance(v, dict) and clean_text(v.get("label")):
                        vals.append(clean_text(v.get("label")))
                    else:
                        vals.append(clean_text(k))
                out[cat] = unique_preserve_order(vals)
            elif isinstance(concepts, list):
                vals = []
                for item in concepts:
                    if isinstance(item, str):
                        vals.append(item)
                    elif isinstance(item, dict):
                        vals.append(clean_text(item.get("label") or item.get("concept") or item.get("name")))
                out[cat] = unique_preserve_order(vals)
            else:
                raise ValueError(f"Unsupported concept structure for category {cat!r}")
        return out

    normalized = {
        "creator_content": normalize_side(creator),
        "audience": normalize_side(audience),
    }
    if not normalized["creator_content"] or not normalized["audience"]:
        raise ValueError("Seed ontology must contain both creator/content and audience sides")
    return normalized


def seed_rows(seed: Dict[str, Dict[str, List[str]]]) -> pd.DataFrame:
    rows = []
    for side, categories in seed.items():
        inventory_source = "3A_content" if side == "creator_content" else "3B_audience"
        for category, concepts in categories.items():
            for concept in concepts:
                rows.append({
                    "ontology_node_id": canonical_node_id(inventory_source, category, concept),
                    "inventory_source": inventory_source,
                    "ontology_side": side,
                    "taxonomy_category": category,
                    "canonical_concept": concept,
                    "introduced_version": "0.1",
                    "source": "seed_v0.1",
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Review validation
# ---------------------------------------------------------------------------

def validate_review(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    messages: List[Dict[str, str]] = []

    missing = REQUIRED_REVIEW_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Review file is missing required columns: {sorted(missing)}")

    out = df.copy()
    for col in REQUIRED_REVIEW_COLUMNS:
        out[col] = out[col].map(clean_text)

    for idx, row in out.iterrows():
        row_id = row["consolidation_id"] or f"row_{idx}"
        status = norm_key(row["human_review_status"])
        decision = norm_key(row["human_ontology_decision"])
        concept = row["human_final_concept"]
        category = row["human_final_category"]
        target = row["human_merge_target_existing_node"]

        if status != "completed":
            error(messages, "review_not_completed", f"human_review_status={row['human_review_status']!r}", row_id)

        if decision not in VALID_DECISIONS:
            error(messages, "invalid_decision", f"Unsupported decision {row['human_ontology_decision']!r}", row_id)
            continue

        if decision == "approve_new_node":
            if not concept:
                error(messages, "missing_final_concept", "approve_new_node requires human_final_concept", row_id)
            if not category:
                error(messages, "missing_final_category", "approve_new_node requires human_final_category", row_id)
            if target:
                # This catches the Vocal Ability row where explanatory text landed in merge_target.
                warn(
                    messages,
                    "unexpected_merge_target_for_approve",
                    "approve_new_node has a nonblank human_merge_target_existing_node; "
                    "the value will be ignored. Move explanatory prose to human_review_notes if desired.",
                    row_id,
                )

        elif decision == "merge_existing_node":
            if not target:
                error(messages, "missing_merge_target", "merge_existing_node requires human_merge_target_existing_node", row_id)
            # final concept/category can be filled from target later if target is machine-resolvable.

        elif decision in {"defer", "reject"}:
            # Category may be intentionally blank for unresolved ontology placement.
            pass

    return out, messages


# ---------------------------------------------------------------------------
# Build approved additions and v0.2 nodes
# ---------------------------------------------------------------------------

def build_approved_additions(review: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    approved = review[review["human_ontology_decision"].map(norm_key) == "approve_new_node"].copy()

    if approved.empty:
        columns = [
            "ontology_node_id", "inventory_source", "ontology_side", "taxonomy_category",
            "canonical_concept", "introduced_version", "source", "supporting_consolidation_ids",
            "supporting_candidate_ids", "review_row_count", "review_notes"
        ]
        return pd.DataFrame(columns=columns), approved

    approved["_dedupe_key"] = approved.apply(
        lambda r: (
            norm_key(r["inventory_source"]),
            norm_key(r["human_final_category"]),
            norm_key(r["human_final_concept"]),
        ),
        axis=1,
    )

    addition_rows = []
    for _, group in approved.groupby("_dedupe_key", sort=False):
        first = group.iloc[0]
        inventory_source = first["inventory_source"]
        category = first["human_final_category"]
        concept = first["human_final_concept"]
        side = infer_side(inventory_source)

        candidate_ids = unique_preserve_order(
            cid for value in group["supporting_candidate_ids"] for cid in split_pipe(value)
        )
        notes = unique_preserve_order(group["human_review_notes"].tolist())

        addition_rows.append({
            "ontology_node_id": canonical_node_id(inventory_source, category, concept),
            "inventory_source": inventory_source,
            "ontology_side": side,
            "taxonomy_category": category,
            "canonical_concept": concept,
            "introduced_version": "0.2",
            "source": "human_review_approved",
            "supporting_consolidation_ids": " | ".join(unique_preserve_order(group["consolidation_id"].tolist())),
            "supporting_candidate_ids": " | ".join(candidate_ids),
            "review_row_count": len(group),
            "review_notes": " || ".join(notes),
        })

    additions = pd.DataFrame(addition_rows)
    return additions, approved


def check_seed_collisions(
    seed_df: pd.DataFrame,
    additions: pd.DataFrame,
    messages: List[Dict[str, str]],
) -> pd.DataFrame:
    """If an approved new node already exists in v0.1, don't duplicate it."""
    if additions.empty:
        return additions

    seed_keys = {
        (norm_key(r.inventory_source), norm_key(r.taxonomy_category), norm_key(r.canonical_concept))
        for r in seed_df.itertuples(index=False)
    }

    keep = []
    for row in additions.itertuples(index=False):
        key = (norm_key(row.inventory_source), norm_key(row.taxonomy_category), norm_key(row.canonical_concept))
        if key in seed_keys:
            warn(
                messages,
                "approved_node_already_exists_in_v01",
                f"{row.canonical_concept!r} already exists in v0.1 under {row.taxonomy_category!r}; "
                "it will not be duplicated in v0.2 additions.",
                clean_text(row.supporting_consolidation_ids),
            )
            keep.append(False)
        else:
            keep.append(True)

    return additions.loc[keep].reset_index(drop=True)


def build_final_nodes(seed_df: pd.DataFrame, additions: pd.DataFrame) -> pd.DataFrame:
    all_cols = sorted(set(seed_df.columns) | set(additions.columns))
    seed_aligned = seed_df.reindex(columns=all_cols)
    additions_aligned = additions.reindex(columns=all_cols)
    final = pd.concat([seed_aligned, additions_aligned], ignore_index=True)

    # Defensive dedupe on semantic identity.
    final["_key"] = final.apply(
        lambda r: (
            norm_key(r["inventory_source"]),
            norm_key(r["taxonomy_category"]),
            norm_key(r["canonical_concept"]),
        ),
        axis=1,
    )
    final = final.drop_duplicates("_key", keep="first").drop(columns="_key")
    return final.sort_values(
        ["ontology_side", "taxonomy_category", "canonical_concept"],
        key=lambda s: s.astype(str).str.casefold(),
    ).reset_index(drop=True)


def nodes_to_json(final_nodes: pd.DataFrame) -> Dict[str, Any]:
    ontology: Dict[str, Dict[str, List[Dict[str, str]]]] = {
        "creator_content": defaultdict(list),
        "audience": defaultdict(list),
    }

    for row in final_nodes.itertuples(index=False):
        side = clean_text(row.ontology_side)
        category = clean_text(row.taxonomy_category)
        ontology.setdefault(side, defaultdict(list))
        ontology[side][category].append({
            "id": clean_text(row.ontology_node_id),
            "label": clean_text(row.canonical_concept),
            "introduced_version": clean_text(row.introduced_version),
        })

    # Convert defaultdicts and sort labels for deterministic output.
    result: Dict[str, Any] = {
        "version": "0.2",
        "creator_content": {},
        "audience": {},
    }
    for side in ["creator_content", "audience"]:
        for category in sorted(ontology.get(side, {}).keys(), key=str.casefold):
            items = ontology[side][category]
            result[side][category] = sorted(items, key=lambda x: x["label"].casefold())
    return result


# ---------------------------------------------------------------------------
# Mappings / provenance
# ---------------------------------------------------------------------------

def resolve_review_mapping(
    review: pd.DataFrame,
    final_nodes: pd.DataFrame,
    messages: List[Dict[str, str]],
) -> pd.DataFrame:
    """
    Build one canonical outcome per reviewed consolidation.

    approve_new_node -> final human concept/category
    merge_existing_node -> merge target (best-effort target resolution)
    defer/reject -> no active v0.2 node
    """
    node_lookup = {}
    concept_lookup = defaultdict(list)
    for row in final_nodes.itertuples(index=False):
        key = (norm_key(row.inventory_source), norm_key(row.taxonomy_category), norm_key(row.canonical_concept))
        node_lookup[key] = row
        concept_lookup[(norm_key(row.inventory_source), norm_key(row.canonical_concept))].append(row)

    rows = []
    for r in review.itertuples(index=False):
        decision = norm_key(r.human_ontology_decision)
        inventory_source = clean_text(r.inventory_source)
        category = clean_text(r.human_final_category)
        concept = clean_text(r.human_final_concept)
        node_id = ""
        active = False
        mapping_status = decision

        if decision == "approve_new_node":
            key = (norm_key(inventory_source), norm_key(category), norm_key(concept))
            node = node_lookup.get(key)
            if node is not None:
                node_id = clean_text(node.ontology_node_id)
                active = True
                # If it collided with a seed node, this still resolves correctly.
                mapping_status = "approved_or_existing_canonical"
            else:
                error(messages, "approved_node_not_found", f"Could not resolve approved node {concept!r}", r.consolidation_id)

        elif decision == "merge_existing_node":
            target = clean_text(r.human_merge_target_existing_node)
            # Best effort: target may be a label or a node id.
            id_match = final_nodes[final_nodes["ontology_node_id"].map(norm_key) == norm_key(target)]
            if len(id_match) == 1:
                node = id_match.iloc[0]
                concept = clean_text(node["canonical_concept"])
                category = clean_text(node["taxonomy_category"])
                node_id = clean_text(node["ontology_node_id"])
                active = True
            else:
                matches = concept_lookup.get((norm_key(inventory_source), norm_key(target)), [])
                if len(matches) == 1:
                    node = matches[0]
                    concept = clean_text(node.canonical_concept)
                    category = clean_text(node.taxonomy_category)
                    node_id = clean_text(node.ontology_node_id)
                    active = True
                else:
                    error(
                        messages,
                        "unresolved_existing_merge_target",
                        f"Could not uniquely resolve merge target {target!r} against final ontology.",
                        r.consolidation_id,
                    )

        rows.append({
            "consolidation_id": clean_text(r.consolidation_id),
            "inventory_source": inventory_source,
            "original_proposed_category": clean_text(getattr(r, "proposed_taxonomy_category", "")),
            "representative_proposed_label": clean_text(getattr(r, "representative_proposed_label", "")),
            "human_decision": clean_text(r.human_ontology_decision),
            "canonical_concept": concept if active else "",
            "canonical_category": category if active else "",
            "ontology_node_id": node_id,
            "active_in_v02": active,
            "mapping_status": mapping_status,
            "supporting_candidate_ids": clean_text(r.supporting_candidate_ids),
            "human_review_notes": clean_text(r.human_review_notes),
        })

    return pd.DataFrame(rows)


def build_candidate_mapping(review_mapping: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in review_mapping.itertuples(index=False):
        for candidate_id in split_pipe(r.supporting_candidate_ids):
            rows.append({
                "candidate_id": candidate_id,
                "consolidation_id": r.consolidation_id,
                "inventory_source": r.inventory_source,
                "human_decision": r.human_decision,
                "canonical_concept": r.canonical_concept,
                "canonical_category": r.canonical_category,
                "ontology_node_id": r.ontology_node_id,
                "active_in_v02": r.active_in_v02,
            })
    return pd.DataFrame(rows)


def enrich_with_candidate_file(candidate_mapping: pd.DataFrame, candidate_path: Optional[Path]) -> pd.DataFrame:
    """
    Optional enrichment. The 4B file schema can vary, so keep all candidate
    columns and join on a detected candidate-id column.
    """
    if candidate_path is None:
        return candidate_mapping

    candidates = pd.read_csv(candidate_path)
    id_candidates = [
        "candidate_id",
        "normalization_candidate_id",
        "cluster_id",
        "candidate_cluster_id",
    ]
    id_col = next((c for c in id_candidates if c in candidates.columns), None)
    if id_col is None:
        raise ValueError(
            f"Could not find a candidate id column in {candidate_path}. "
            f"Tried {id_candidates}; available columns={list(candidates.columns)}"
        )

    candidates = candidates.copy()
    candidates[id_col] = candidates[id_col].map(clean_text)
    enriched = candidate_mapping.merge(
        candidates,
        left_on="candidate_id",
        right_on=id_col,
        how="left",
        suffixes=("_canonical", "_4b"),
        validate="m:1",
    )
    return enriched


def build_change_log(review: pd.DataFrame, review_mapping: pd.DataFrame) -> pd.DataFrame:
    mapping_cols = review_mapping[[
        "consolidation_id", "canonical_concept", "canonical_category",
        "ontology_node_id", "active_in_v02", "mapping_status"
    ]]
    change = review.merge(mapping_cols, on="consolidation_id", how="left", validate="1:1")

    preferred = [
        "consolidation_id",
        "inventory_source",
        "proposed_taxonomy_category",
        "representative_proposed_label",
        "member_proposed_labels",
        "human_final_concept",
        "human_ontology_decision",
        "human_final_category",
        "human_merge_target_existing_node",
        "canonical_concept",
        "canonical_category",
        "ontology_node_id",
        "active_in_v02",
        "mapping_status",
        "proposal_cluster_count",
        "supporting_raw_label_count",
        "raw_annotation_count",
        "avg_proposal_confidence",
        "mean_pairwise_similarity",
        "consolidation_cohesion",
        "review_priority",
        "selection_reason",
        "supporting_candidate_ids",
        "human_review_notes",
        "human_review_status",
    ]
    cols = [c for c in preferred if c in change.columns]
    remaining = [c for c in change.columns if c not in cols]
    return change[cols + remaining]


# ---------------------------------------------------------------------------
# Summary / output
# ---------------------------------------------------------------------------

def build_summary(
    review: pd.DataFrame,
    seed_df: pd.DataFrame,
    additions: pd.DataFrame,
    final_nodes: pd.DataFrame,
    candidate_mapping: pd.DataFrame,
    messages: List[Dict[str, str]],
) -> pd.DataFrame:
    decision_counts = review["human_ontology_decision"].map(norm_key).value_counts().to_dict()
    warnings = sum(1 for m in messages if m["level"] == "warning")
    errors = sum(1 for m in messages if m["level"] == "error")

    rows = [
        ("ontology_version", "0.2"),
        ("review_rows", len(review)),
        ("review_completed_rows", int((review["human_review_status"].map(norm_key) == "completed").sum())),
        ("approve_new_node_rows", decision_counts.get("approve_new_node", 0)),
        ("merge_existing_node_rows", decision_counts.get("merge_existing_node", 0)),
        ("defer_rows", decision_counts.get("defer", 0)),
        ("reject_rows", decision_counts.get("reject", 0)),
        ("seed_v01_node_count", len(seed_df)),
        ("unique_v02_added_node_count", len(additions)),
        ("final_v02_node_count", len(final_nodes)),
        ("candidate_mapping_rows", len(candidate_mapping)),
        ("validation_warning_count", warnings),
        ("validation_error_count", errors),
    ]

    # Peer consolidation count = approved review rows minus unique approved additions
    approved_rows = decision_counts.get("approve_new_node", 0)
    rows.append(("peer_deduped_review_rows", max(0, approved_rows - len(additions))))

    return pd.DataFrame(rows, columns=["metric", "value"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize human-reviewed taxonomy into ontology v0.2")
    parser.add_argument(
        "--review",
        type=Path,
        default=Path("data/evaluation/ontology_v02_human_review_complete.csv"),
        help="Completed 4C human review CSV",
    )
    seed_group = parser.add_mutually_exclusive_group(required=False)
    seed_group.add_argument(
        "--seed-python",
        type=Path,
        default=None,
        help="Python file containing literal CREATOR_ONTOLOGY and AUDIENCE_ONTOLOGY dictionaries",
    )
    seed_group.add_argument(
        "--seed-json",
        type=Path,
        default=None,
        help="Existing v0.1 ontology JSON",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="Optional 4B candidate CSV; joined to candidate-level mapping for raw-label provenance",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ontology/v0.2"),
        help="Output directory",
    )
    parser.add_argument(
        "--allow-warnings",
        action="store_true",
        help="Do not fail because of validation warnings. Errors always fail.",
    )
    args = parser.parse_args()

    if not args.review.exists():
        raise FileNotFoundError(f"Review file not found: {args.review}")

    # Sensible project default: reuse exact dictionaries from 4B.2 script.
    if args.seed_python is None and args.seed_json is None:
        default_seed = Path("src/propose_taxonomy_normalization_candidates.py")
        if default_seed.exists():
            args.seed_python = default_seed
        else:
            raise FileNotFoundError(
                "No v0.1 ontology source was provided. Pass --seed-python pointing to the script "
                "that contains CREATOR_ONTOLOGY and AUDIENCE_ONTOLOGY, or pass --seed-json. "
                "This script intentionally does not recreate v0.1 from memory."
            )

    if args.seed_python is not None:
        if not args.seed_python.exists():
            raise FileNotFoundError(f"Seed Python file not found: {args.seed_python}")
        seed_raw = load_seed_from_python(args.seed_python)
    else:
        if not args.seed_json.exists():
            raise FileNotFoundError(f"Seed JSON file not found: {args.seed_json}")
        seed_raw = load_seed_from_json(args.seed_json)

    seed = normalize_seed_structure(seed_raw)
    seed_df = seed_rows(seed)

    review_raw = pd.read_csv(args.review)
    review, messages = validate_review(review_raw)

    # Fatal review errors stop before ontology generation.
    initial_errors = [m for m in messages if m["level"] == "error"]
    if initial_errors:
        print("\nValidation errors:")
        for m in initial_errors:
            print(f"  - [{m['code']}] {m['row_id']}: {m['message']}")
        raise SystemExit(2)

    additions, _ = build_approved_additions(review)
    additions = check_seed_collisions(seed_df, additions, messages)
    final_nodes = build_final_nodes(seed_df, additions)

    review_mapping = resolve_review_mapping(review, final_nodes, messages)
    candidate_mapping = build_candidate_mapping(review_mapping)
    candidate_mapping_enriched = enrich_with_candidate_file(candidate_mapping, args.candidates)
    change_log = build_change_log(review, review_mapping)

    # Errors created during canonical resolution are also fatal.
    final_errors = [m for m in messages if m["level"] == "error"]
    if final_errors:
        print("\nResolution errors:")
        for m in final_errors:
            print(f"  - [{m['code']}] {m['row_id']}: {m['message']}")
        raise SystemExit(2)

    summary = build_summary(review, seed_df, additions, final_nodes, candidate_mapping, messages)
    validation = pd.DataFrame(messages, columns=["level", "code", "row_id", "message"])

    ontology_json = nodes_to_json(final_nodes)

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir / "ontology_v02.json").open("w", encoding="utf-8") as f:
        json.dump(ontology_json, f, ensure_ascii=False, indent=2)

    final_nodes.to_csv(outdir / "ontology_v02_nodes.csv", index=False)
    additions.to_csv(outdir / "ontology_v02_added_nodes.csv", index=False)
    review_mapping.to_csv(outdir / "ontology_v02_review_to_canonical_mapping.csv", index=False)
    candidate_mapping_enriched.to_csv(outdir / "taxonomy_normalization_mapping_v02.csv", index=False)
    change_log.to_csv(outdir / "ontology_v02_change_log.csv", index=False)
    summary.to_csv(outdir / "ontology_v02_build_summary.csv", index=False)
    validation.to_csv(outdir / "ontology_v02_validation.csv", index=False)

    deferred = change_log[
        change_log["human_ontology_decision"].map(norm_key).isin({"defer", "reject"})
    ].copy()
    deferred.to_csv(outdir / "ontology_v02_deferred_rejected.csv", index=False)

    print("\n=== Ontology v0.2 build complete ===")
    print(summary.to_string(index=False))

    if not validation.empty:
        print("\nValidation messages:")
        print(validation.to_string(index=False))

    print("\nOutputs:")
    for name in [
        "ontology_v02.json",
        "ontology_v02_nodes.csv",
        "ontology_v02_added_nodes.csv",
        "ontology_v02_review_to_canonical_mapping.csv",
        "taxonomy_normalization_mapping_v02.csv",
        "ontology_v02_change_log.csv",
        "ontology_v02_build_summary.csv",
        "ontology_v02_validation.csv",
        "ontology_v02_deferred_rejected.csv",
    ]:
        print(f"  - {outdir / name}")


if __name__ == "__main__":
    main()
