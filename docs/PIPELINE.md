# Pipeline Reference

_Content Alignment Evaluation Platform. UI terminology is explained in [DEMO_DECISIONS.md](DEMO_DECISIONS.md). The pipeline keeps internal `intent_*` names for the inferred-framing layer._

Each stage, the script that runs it, what it reads, what it writes, and where a human signs off.
Every script reads and writes relative paths under `data/`. `run_pipeline.sh` runs them in order from the workspace root.

```
 1  INGEST            YouTube API + transcripts
      │               data/raw/{content, transcript_segment, audience_comment}.csv
      ▼
 3A CONTENT ─────────┐            3B AUDIENCE
    framing +        │               comment-level
    content concepts │               perception
      │              │                  │
      │              │            3B.1 HUMAN QC ⏸  precision / recall / interpretation quality
      ▼              ▼                  ▼
            FREE-FORM RAW LABELS (894 unique)
      │
 4A  label inventory            → measure vocabulary fragmentation
 4B.1 semantic grouping         → 894 → 658 candidate groups
 4B.2 ontology proposal (LLM)   → existing / new / reject / ambiguous
 4B.3 consolidation             → 290 → 169 new-concept proposals
 4C  human review ⏸             → approve / merge / defer / reject
 4D  finalize                   → Ontology v0.2 (88 → 123 nodes) + raw→canonical mapping
      │
 5A  normalized profiles        → per-video concept profiles for inferred framing / content / audience
 5B  relationship candidates    → cross-ontology pairs (exact, seed rules, semantic)
 5B.1 human rule review ⏸       → relationship_rules_v0.1 (32 approved, 7 rejected, 1 deferred)
 5B v2 governed pairs           → pairs supported only by exact match or approved rules
 5C  alignment diagnostics      → Framing→Content, Content→Audience, Framing→Audience
 5D.1 diagnostics               → aligned / partial / missing resonance / unexpected perceptions
 5D.2 decision insights         → realization, carry-through, gaps, demand, community, opportunities
      │
 P0-A demo dataset              → data/demo/ (static, for the Streamlit app)
```

## Stage table

| # | Stage | Script (`pipeline/…`) | Reads | Writes | Needs |
|---|---|---|---|---|---|
| 1 | Ingestion | `stage1_ingestion/collect_yt_v2.py` | `selected_videos.csv` | `data/raw/content.csv`, `transcript_segment.csv`, `audience_comment.csv`, `youtube_ingestion_run_log.csv` | `YOUTUBE_API_KEY` |
| 3A | Content understanding | `stage3_extraction/run_3a_content_extraction.py` | `data/raw/content.csv`, `transcript_segment.csv` | `data/processed/taxonomy_annotation_raw_3a.csv`, `intent_content_semantic_match_3a.csv`, run log/summary | `OPENAI_API_KEY` |
| 3A.1 | Extraction ablation *(optional)* | `stage3_extraction/run_3a_1_content_extraction_check.py` | same as 3A | `data/processed_test/…` | `OPENAI_API_KEY` |
| 3B | Audience understanding | `stage3_extraction/run_3b_audience_extraction.py` | `data/raw/audience_comment.csv` | `data/processed/taxonomy_annotation_raw_3b.csv`, `audience_comment_annotation_status_3b.csv` | `OPENAI_API_KEY` |
| 3B.1 | QC sample | `stage3_extraction/sample_3b_human_qc.py` | 3B outputs, raw comments | `data/evaluation/audience_human_qc_sample.csv` | — |
| ⏸ | Human review | — | `audience_human_qc_sample.csv` | `audience_human_qc_reviewed.csv` | human |
| 3B.1 | QC evaluation | `stage3_extraction/evaluate_3b_human_qc.py` | `audience_human_qc_reviewed.csv` | `audience_human_qc_{metrics, evaluated, group_analysis, error_analysis}.csv`, `audience_signal_confusion_matrix.csv` | — |
| 4A | Label inventory | `stage4_ontology/build_taxonomy_label_inventory.py` | `taxonomy_annotation_raw_3a/3b.csv` | `taxonomy_label_inventory*.csv`, `taxonomy_label_examples.csv` | — |
| 4B.1 | Semantic grouping | `stage4_ontology/build_taxonomy_normalization_candidates.py` | `taxonomy_label_inventory.csv` | `taxonomy_normalization_candidates.csv`, `…_candidate_members.csv`, `…_candidate_summary.csv` | sentence-transformers |
| 4B.2 | Ontology proposal | `stage4_ontology/propose_taxonomy_normalization_candidates.py` | candidates + members | `taxonomy_normalization_candidates_proposed.csv` | `OPENAI_API_KEY` |
| 4B.3 | Consolidation | `stage4_ontology/consolidate_ontology_proposals.py` | proposals + members | `ontology_v02_consolidation_{candidates, members, summary}.csv` | sentence-transformers |
| 4C | Review sheet | `stage4_ontology/prepare_ontology_v02_human_review.py` | consolidation candidates | `data/evaluation/ontology_v02_human_review.csv` | — |
| ⏸ | Human review | — | `ontology_v02_human_review.csv` | `ontology_v02_human_review_complete.csv` | human |
| 4D | Finalize ontology | `stage4_ontology/finalize_ontology_v02.py` | completed review, seed v0.1, candidates | `data/ontology/v0.2/ontology_v02_nodes.csv`, `taxonomy_normalization_mapping_v02.csv`, `ontology_v02.json`, change log | — |
| 5A | Normalized profiles | `stage5_alignment/build_normalized_alignment_profiles_5a_v3.py` | 3A/3B annotations, 4B/4D mapping, ontology nodes | `data/alignment/5a/normalized_annotations_5a.csv`, `content_concept_profile_5a.csv`, `audience_concept_profile_5a.csv`, `content_role_profile_5a.csv` | — |
| 5B | Relationship candidates | `stage5_alignment/build_cross_ontology_relationships_5b.py` | 5A profiles, ontology nodes | `data/alignment/5b/relationship_review_queue_5b.csv`, `relationship_rules_5b.csv`, catalog | sentence-transformers |
| ⏸ | Human review | — | `relationship_review_queue_5b.csv` | `relationship_review_queue_5b_reviewed.csv` | human |
| 5B.1 | Governed rules | `stage5_alignment/finalize_relationship_rules_5b1.py` | reviewed queue, base rules | `data/alignment/5b/rules_v0.1/relationship_rules_v0.1.csv` | — |
| 5B v2 | Governed pairs | `stage5_alignment/build_cross_ontology_relationships_5b_v2.py` | 5A profiles, rules v0.1 | `data/alignment/5b_v0.1/concept_pair_candidates_5b.csv`, availability, pair summary | — |
| 5C | Scores | `stage5_alignment/compute_alignment_scores_5c.py` | 5B v2 outputs | `data/alignment/5c/alignment_{content_summary, comparison_score, source_concept_score, pair_best_match}_5c.csv` | — |
| 5D.1 | Diagnostics | `stage5_alignment/build_alignment_diagnostics_5d1_v2.py` | 5C scores, 5B v2 pairs | `data/alignment/5d_v2/{alignment_insights, content_alignment_diagnostic, audience_unmatched_signals}_5d_v2.csv` | — |
| 5D.2 | Decision insights | `stage5_alignment/build_decision_insights_5d2_v2.py` | 5D.1 outputs | `data/alignment/5d2_v2/decision_insights_5d2_v2.csv`, `decision_insight_evidence_5d2_v2.csv` | — |
| P0-A | Demo dataset | `scripts/build_demo_dataset.py` | outputs below | `data/demo/*` | — |

> The 5B (v1) command in `run_pipeline.sh` was reconstructed from the script's CLI. All other commands match the documented examples in each script's docstring.

## What feeds the demo (P0-A)

| Demo file | Built from | Used in app section |
|---|---|---|
| `demo_content.csv` | `raw/content.csv`, `5d2_v2/decision_insights`, `5c/alignment_content_summary`, `5a/normalized_annotations` + curated story in `CASES` | Header, content selector, Overview |
| `demo_decision_insights.csv` | `5d2_v2/decision_insight_evidence_5d2_v2.csv` | Overview (key findings), Audience Intelligence |
| `demo_alignment_pairs.csv` | `5d_v2/alignment_insights`, `5d_v2/content_alignment_diagnostic` | Alignment Journey |
| `demo_alignment_evidence.csv` | `5a/normalized_annotations`, `processed/taxonomy_annotation_raw_3a/3b`, `raw/audience_comment` | Alignment Journey (evidence drawer) |
| `demo_concept_profiles.csv` | `5a/content_concept_profile`, `5a/audience_concept_profile` | Audience Intelligence |
| `demo_relationship_rules.csv` | `5b/rules_v0.1/relationship_rules_v0.1.csv` | Evaluation & System |
| `demo_system_metrics.csv` | build summaries (4A–5D), `evaluation/audience_human_qc_*` | Evaluation & System |
| `demo_manifest.json` | checksums of all sources above | provenance |

## Canonical versions

When several versions of a script exist in the original workspace, these are the ones in the chain:

| Step | Canonical | Superseded |
|---|---|---|
| Ingestion | `collect_yt_v2.py` | `collect_yt.py` |
| 5A | `build_normalized_alignment_profiles_5a_v3.py` | — |
| 5B | `…_5b.py` (candidates) → `…_5b1.py` (rules) → `…_5b_v2.py` (governed pairs) | output dir `5b_v0.1/` is the v2 output |
| 5D.1 | `build_alignment_diagnostics_5d1_v2.py` → `5d_v2/` | `…_5d1.py` → `5d/` |
| 5D.2 | `build_decision_insights_5d2_v2.py` → `5d2_v2/` | `…_5d2.py` → `5d2/` |
