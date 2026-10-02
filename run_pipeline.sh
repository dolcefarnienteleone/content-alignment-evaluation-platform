#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Content Alignment Evaluation Platform — end-to-end pipeline runner
#
#   ./run_pipeline.sh <stage>
#
# Scripts read/write relative paths under data/, so everything runs from the
# WORKSPACE folder (default: this repo). Point it elsewhere with:
#   WORKSPACE=/path/to/workspace ./run_pipeline.sh <stage>
#
# Stages, in order. ⏸ = human-in-the-loop checkpoint (the runner stops and
# tells you which file to review before the next stage can run).
#
#   ingest              1   YouTube metadata, transcripts, comments      [YOUTUBE_API_KEY]
#   extract             3A/3B LLM taxonomy extraction                    [OPENAI_API_KEY]
#   qc-sample           3B.1 draw stratified human-QC sample             ⏸ review sample
#   qc-eval             3B.1 precision / recall / interpretation quality
#   ontology            4A inventory → 4B.1 grouping → 4B.2 proposals
#                       → consolidation → 4C review sheet                ⏸ review ontology
#   ontology-finalize   4D build ontology v0.2 + normalization mapping
#   align-candidates    5A normalized profiles → 5B relationship candidates  ⏸ review rules
#   align               5B.1 rules v0.1 → 5B v2 → 5C diagnostics → 5D.1 diagnostics → 5D.2 insights
#   demo                P0-A static demo dataset (data/demo/)
#   offline             qc-eval + ontology-finalize + align + demo (no API calls)
# -----------------------------------------------------------------------------
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "${WORKSPACE:-$REPO}" && pwd)"   # absolute, so a relative WORKSPACE works
P="$REPO/pipeline"
PY="${PYTHON:-python}"
cd "$WS"

say()  { printf '\n\033[1m▶ %s\033[0m\n' "$*"; }
need() {  # need <file> <what to do>
  if [ ! -f "$1" ]; then
    printf '\n⏸  Human checkpoint — missing: %s\n   %s\n' "$1" "$2"
    exit 3
  fi
}

stage_ingest() {
  say "1 · Ingest YouTube content, transcripts, comments"
  [ -f selected_videos.csv ] || cp "$P/config/selected_videos.csv" .
  $PY "$P/stage1_ingestion/collect_yt_v2.py"
}

stage_extract() {
  say "3A · Content extraction (inferred framing + content understanding)"
  $PY "$P/stage3_extraction/run_3a_content_extraction.py"
  say "3B · Audience understanding (comment-level perception)"
  $PY "$P/stage3_extraction/run_3b_audience_extraction.py"
}

stage_qc_sample() {
  say "3B.1 · Draw human-QC sample"
  $PY "$P/stage3_extraction/sample_3b_human_qc.py"
  echo "⏸  Review data/evaluation/audience_human_qc_sample.csv and save it as"
  echo "   data/evaluation/audience_human_qc_reviewed.csv, then run: ./run_pipeline.sh qc-eval"
}

stage_qc_eval() {
  need data/evaluation/audience_human_qc_reviewed.csv \
       "Fill in the human judgements on the QC sample (see qc-sample)."
  say "3B.1 · Evaluate 3B against human review"
  $PY "$P/stage3_extraction/evaluate_3b_human_qc.py"
}

stage_ontology() {
  say "4A · Label inventory"
  $PY "$P/stage4_ontology/build_taxonomy_label_inventory.py"
  say "4B.1 · Semantic grouping"
  $PY "$P/stage4_ontology/build_taxonomy_normalization_candidates.py"
  say "4B.2 · Ontology proposals (LLM)"
  $PY "$P/stage4_ontology/propose_taxonomy_normalization_candidates.py"
  say "4B.3 · Consolidate proposals"
  $PY "$P/stage4_ontology/consolidate_ontology_proposals.py"
  say "4C · Prepare human review sheet"
  $PY "$P/stage4_ontology/prepare_ontology_v02_human_review.py"
  echo "⏸  Review data/evaluation/ontology_v02_human_review.csv and save it as"
  echo "   data/evaluation/ontology_v02_human_review_complete.csv, then run: ./run_pipeline.sh ontology-finalize"
}

stage_ontology_finalize() {
  need data/evaluation/ontology_v02_human_review_complete.csv \
       "Complete the 4C ontology review (see ontology)."
  say "4D · Finalize ontology v0.2"
  $PY "$P/stage4_ontology/finalize_ontology_v02.py" \
    --review data/evaluation/ontology_v02_human_review_complete.csv \
    --seed-python "$P/stage4_ontology/propose_taxonomy_normalization_candidates.py" \
    --candidates data/processed/taxonomy_normalization_candidates.csv \
    --output-dir data/ontology/v0.2
}

stage_align_candidates() {
  say "5A · Normalized content / audience concept profiles"
  $PY "$P/stage5_alignment/build_normalized_alignment_profiles_5a_v3.py" \
    --annotations-3a data/processed/taxonomy_annotation_raw_3a.csv \
    --annotations-3b data/processed/taxonomy_annotation_raw_3b.csv \
    --candidates data/processed/taxonomy_normalization_candidates.csv \
    --proposals data/processed/taxonomy_normalization_candidates_proposed.csv \
    --mapping-4d data/ontology/v0.2/taxonomy_normalization_mapping_v02.csv \
    --ontology-nodes data/ontology/v0.2/ontology_v02_nodes.csv \
    --content data/raw/content.csv \
    --audience data/raw/audience_comment.csv \
    --output-dir data/alignment/5a
  say "5B · Cross-ontology relationship candidates (semantic + seed rules)"
  $PY "$P/stage5_alignment/build_cross_ontology_relationships_5b.py" \
    --normalized-annotations data/alignment/5a/normalized_annotations_5a.csv \
    --concept-profile data/alignment/5a/content_concept_profile_5a.csv \
    --role-profile data/alignment/5a/content_role_profile_5a.csv \
    --ontology-nodes data/ontology/v0.2/ontology_v02_nodes.csv \
    --output-dir data/alignment/5b
  echo "⏸  Review data/alignment/5b/relationship_review_queue_5b.csv and save it as"
  echo "   data/alignment/5b/relationship_review_queue_5b_reviewed.csv, then run: ./run_pipeline.sh align"
}

stage_align() {
  need data/alignment/5b/relationship_review_queue_5b_reviewed.csv \
       "Approve / reject / defer the proposed relationships (see align-candidates)."
  say "5B.1 · Finalize governed relationship rules v0.1"
  $PY "$P/stage5_alignment/finalize_relationship_rules_5b1.py" \
    --review-queue data/alignment/5b/relationship_review_queue_5b_reviewed.csv \
    --base-rules data/alignment/5b/relationship_rules_5b.csv \
    --output-dir data/alignment/5b/rules_v0.1
  say "5B v2 · Rebuild concept pairs with governed rules only"
  $PY "$P/stage5_alignment/build_cross_ontology_relationships_5b_v2.py" \
    --normalized-annotations data/alignment/5a/normalized_annotations_5a.csv \
    --concept-profile data/alignment/5a/content_concept_profile_5a.csv \
    --role-profile data/alignment/5a/content_role_profile_5a.csv \
    --ontology-nodes data/ontology/v0.2/ontology_v02_nodes.csv \
    --relationship-rules data/alignment/5b/rules_v0.1/relationship_rules_v0.1.csv \
    --output-dir data/alignment/5b_v0.1
  say "5C · Alignment diagnostics (Framing→Content, Content→Audience, Framing→Audience)"
  $PY "$P/stage5_alignment/compute_alignment_scores_5c.py" \
    --pairs data/alignment/5b_v0.1/concept_pair_candidates_5b.csv \
    --availability data/alignment/5b_v0.1/content_comparison_availability_5b.csv \
    --pair-summary data/alignment/5b_v0.1/content_pair_summary_5b.csv \
    --output-dir data/alignment/5c
  say "5D.1 · Alignment diagnostics"
  $PY "$P/stage5_alignment/build_alignment_diagnostics_5d1_v2.py" \
    --source-scores data/alignment/5c/alignment_source_concept_score_5c.csv \
    --comparison-scores data/alignment/5c/alignment_comparison_score_5c.csv \
    --pairs data/alignment/5b_v0.1/concept_pair_candidates_5b.csv \
    --availability data/alignment/5b_v0.1/content_comparison_availability_5b.csv \
    --output-dir data/alignment/5d_v2
  say "5D.2 · Decision insights"
  $PY "$P/stage5_alignment/build_decision_insights_5d2_v2.py" \
    --diagnostic data/alignment/5d_v2/content_alignment_diagnostic_5d_v2.csv \
    --source-insights data/alignment/5d_v2/alignment_insights_5d_v2.csv \
    --audience-signals data/alignment/5d_v2/audience_unmatched_signals_5d_v2.csv \
    --output-dir data/alignment/5d2_v2
}

stage_demo() {
  say "P0-A · Build static demo dataset"
  $PY "$REPO/scripts/build_demo_dataset.py" --source-root "$WS" --output-dir "$REPO/data/demo"
}

case "${1:-}" in
  ingest)            stage_ingest ;;
  extract)           stage_extract ;;
  qc-sample)         stage_qc_sample ;;
  qc-eval)           stage_qc_eval ;;
  ontology)          stage_ontology ;;
  ontology-finalize) stage_ontology_finalize ;;
  align-candidates)  stage_align_candidates ;;
  align)             stage_align ;;
  demo)              stage_demo ;;
  offline)           stage_qc_eval; stage_ontology_finalize; stage_align; stage_demo ;;
  *) sed -n '2,27p' "$0"; exit 1 ;;
esac
