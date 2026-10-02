from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


# ============================================================
# CONFIG
# ============================================================

CONTENT_CSV = Path("data/raw/content.csv")
TRANSCRIPT_CSV = Path("data/raw/transcript_segment.csv")

OUTPUT_DIR = Path("data/processed_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_ANNOTATIONS = OUTPUT_DIR / "taxonomy_annotation_3a1_ablation.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "taxonomy_extraction_3a1_ablation_summary.csv"

MODEL_NAME = os.getenv("CONTENT_EXTRACTION_MODEL", "gpt-5.6-terra")

TAXONOMY_VERSION = "creator_identity_ontology_v0.1"

OLD_INTENT_PROMPT_VERSION = "inferred_creator_intent_v0.1"
NEW_INTENT_PROMPT_VERSION = "inferred_content_intent_v0.2"
CONTENT_UNDERSTANDING_PROMPT_VERSION = "ai_content_understanding_v0.1"

MAX_CONTENT_CHARS = 40_000

# Recommended two contrast videos:
SELECTED_CONTENT_IDS = [
    "youtube_w0wF-O0EGsQ",  # recording behind
    "youtube_v5DXpxUAO3w",  # summer vacation
]


# ============================================================
# OPENAI SETUP
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. Add it to your .env file."
    )

client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================
# ONTOLOGY
# ============================================================

CREATOR_CATEGORIES = {
    "Core Values": (
        "Underlying values, principles, or recurring messages conveyed "
        "through the content, such as hope, growth, comfort, authenticity, "
        "friendship, or courage."
    ),
    "Personality": (
        "Observable personality characteristics presented through the content, "
        "such as passionate, humble, calm, funny, playful, resilient, "
        "melancholic, or sensitive."
    ),
    "Relationship Style": (
        "How the creator relates to members or other people, such as caring, "
        "supportive, team-oriented, leadership-oriented, loving, brotherhood, "
        "family-like, or inspirational relationships."
    ),
    "Creative Style": (
        "Characteristics of creative expression, production, or performance, "
        "such as artistic, experimental, performance-focused, professional, "
        "spontaneous, surprising, distinctive, or unconventional."
    ),
    "Communication Style": (
        "How ideas or emotions are communicated, such as casual, emotional, "
        "reflective, educational, logical, informative, or storytelling-oriented."
    ),
    "Lifestyle": (
        "Lifestyle-oriented content signals, such as daily life, "
        "behind-the-scenes activities, travel, hobbies, or games."
    ),
    "Visual Identity": (
        "Visual image or aesthetic characteristics, such as cute, cool, "
        "fashionable, luxurious, handsome/beautiful, distinctive, "
        "or animal-like visual associations."
    ),
}

CategoryLiteral = Literal[
    "Core Values",
    "Personality",
    "Relationship Style",
    "Creative Style",
    "Communication Style",
    "Lifestyle",
    "Visual Identity",
]

SourceFieldLiteral = Literal[
    "title",
    "description",
    "hashtags",
    "transcript",
    "multiple",
]


# ============================================================
# STRUCTURED OUTPUT
# ============================================================

class ExtractedCreatorLabel(BaseModel):
    raw_label: str
    taxonomy_category: CategoryLiteral

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    source_field: SourceFieldLiteral
    evidence_span: str
    explanation: str


class CreatorTaxonomyExtraction(BaseModel):
    labels: list[ExtractedCreatorLabel]


# ============================================================
# UTILS
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_annotation_id() -> str:
    return f"ann_{uuid.uuid4().hex[:16]}"


def clean_string(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_jsonish_list(value) -> list[str]:
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass

    return [
        x.strip()
        for x in text.split(",")
        if x.strip()
    ]


def normalize_for_evidence_check(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def validate_evidence(
    evidence_span: str,
    source_text: str,
) -> tuple[bool, str]:

    if not evidence_span:
        return False, "empty"

    if evidence_span in source_text:
        return True, "exact"

    normalized_evidence = normalize_for_evidence_check(
        evidence_span
    )
    normalized_source = normalize_for_evidence_check(
        source_text
    )

    if (
        normalized_evidence
        and normalized_evidence in normalized_source
    ):
        return True, "normalized_whitespace"

    return False, "failed"


# ============================================================
# LOAD DATA
# ============================================================

def load_source_data() -> tuple[pd.DataFrame, pd.DataFrame]:

    content_df = pd.read_csv(CONTENT_CSV)
    transcript_df = pd.read_csv(TRANSCRIPT_CSV)

    content_df = content_df[
        content_df["content_id"].isin(SELECTED_CONTENT_IDS)
    ].copy()

    if content_df.empty:
        raise ValueError(
            "None of the SELECTED_CONTENT_IDS were found in content.csv."
        )

    return content_df, transcript_df


def build_transcript(
    transcript_df: pd.DataFrame,
    content_id: str,
) -> str:

    subset = transcript_df[
        transcript_df["content_id"] == content_id
    ].copy()

    if subset.empty:
        return ""

    if "segment_index" in subset.columns:
        subset = subset.sort_values("segment_index")
    elif "start_seconds" in subset.columns:
        subset = subset.sort_values("start_seconds")

    texts = (
        subset["text"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    return " ".join(
        x for x in texts
        if x
    )


def build_inputs(
    row: pd.Series,
    transcript_df: pd.DataFrame,
) -> dict:

    content_id = row["content_id"]

    title = clean_string(
        row.get("title", "")
    )

    description = clean_string(
        row.get("description", "")
    )

    hashtags = parse_jsonish_list(
        row.get("hashtags", "")
    )

    transcript = build_transcript(
        transcript_df,
        content_id,
    )

    framing_text = (
        f"TITLE\n"
        f"{title}\n\n"
        f"DESCRIPTION\n"
        f"{description}\n\n"
        f"HASHTAGS\n"
        f"{' '.join('#' + x.lstrip('#') for x in hashtags)}"
    )

    full_content_text = (
        f"{framing_text}\n\n"
        f"TRANSCRIPT\n"
        f"{transcript}"
    )

    return {
        "content_id": content_id,
        "title": title,
        "description": description,
        "hashtags": hashtags,
        "transcript": transcript,
        "framing_text": framing_text,
        "full_content_text": full_content_text[:MAX_CONTENT_CHARS],
        "full_content_char_count": len(full_content_text),
        "framing_char_count": len(framing_text),
    }


# ============================================================
# PROMPT HELPERS
# ============================================================

def taxonomy_definition_text() -> str:

    return "\n".join(
        f"- {category}: {definition}"
        for category, definition
        in CREATOR_CATEGORIES.items()
    )


def common_rules() -> str:

    return f"""
Taxonomy categories:

{taxonomy_definition_text()}

Rules:

1. Produce only labels supported by the supplied evidence.

2. raw_label should be concise natural language.
   Do not force raw_label to match existing ontology labels.

3. taxonomy_category must use one of the supplied categories.

4. evidence_span must be copied exactly from the supplied input.

5. Do not paraphrase evidence.

6. Avoid redundant labels.

7. Do not infer private facts or hidden mental states.

8. confidence is an evidence-strength heuristic:
   0.90-1.00 = directly supported
   0.75-0.89 = strongly supported
   0.60-0.74 = plausible but interpretive
   below 0.60 = weak or ambiguous

9. Return no more than 8 labels.

10. Focus on the strongest signals rather than exhaustively labeling
    every detail.
""".strip()


# ============================================================
# VERSION A — OLD INTENT
# Full content
# ============================================================

def extract_old_intent(
    content_input: dict,
) -> CreatorTaxonomyExtraction:

    system_prompt = f"""
You are inferring creator-side intent from publicly available content.

Ask:
"What does this content appear designed to communicate or emphasize?"

You may use title, description, hashtags, and transcript.

This is inferred intent only and must not be treated as confirmed
private creator intent.

{common_rules()}
""".strip()

    user_prompt = f"""
CONTENT_ID:
{content_input["content_id"]}

CONTENT:
{content_input["full_content_text"]}
""".strip()

    response = client.responses.parse(
        model=MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        text_format=CreatorTaxonomyExtraction,
    )

    if response.output_parsed is None:
        raise RuntimeError(
            "No parsed output returned for old intent extraction."
        )

    return response.output_parsed


# ============================================================
# VERSION B — NEW INTENT
# Creator-side framing only
# ============================================================

def extract_new_intent(
    content_input: dict,
) -> CreatorTaxonomyExtraction:

    system_prompt = f"""
You are analyzing PUBLIC CREATOR-SIDE CONTENT FRAMING.

Your task is to infer what the content appears positioned or framed
to communicate BEFORE examining the detailed body of the content.

Use only:
- title
- description
- hashtags

Do NOT infer from transcript dialogue or detailed events inside the video.

Ask:
"Based on how the creator or content team publicly packaged this item,
what identity, message, emotional direction, relationship framing,
creative positioning, or lifestyle theme appears intentionally emphasized?"

This is an external proxy for content intent, not confirmed private intent.

Important:
If the title, description, and hashtags do not support a strong inference,
return fewer labels rather than inventing intent.

{common_rules()}
""".strip()

    user_prompt = f"""
CONTENT_ID:
{content_input["content_id"]}

CREATOR-SIDE FRAMING:
{content_input["framing_text"]}
""".strip()

    response = client.responses.parse(
        model=MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        text_format=CreatorTaxonomyExtraction,
    )

    if response.output_parsed is None:
        raise RuntimeError(
            "No parsed output returned for new intent extraction."
        )

    return response.output_parsed


# ============================================================
# AI CONTENT UNDERSTANDING
# Full content, unchanged
# ============================================================

def extract_content_understanding(
    content_input: dict,
) -> CreatorTaxonomyExtraction:

    system_prompt = f"""
You are analyzing what identity and content characteristics are
OBSERVABLY PRESENT in creator content.

Ask:
"What characteristics can actually be detected in the content itself,
regardless of whether they were intentionally planned?"

Use title, description, hashtags, and full transcript.

Do not speculate about private creator intent.

{common_rules()}
""".strip()

    user_prompt = f"""
CONTENT_ID:
{content_input["content_id"]}

FULL CONTENT:
{content_input["full_content_text"]}
""".strip()

    response = client.responses.parse(
        model=MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        text_format=CreatorTaxonomyExtraction,
    )

    if response.output_parsed is None:
        raise RuntimeError(
            "No parsed output returned for content understanding."
        )

    return response.output_parsed


# ============================================================
# FLATTEN
# ============================================================

def extraction_to_rows(
    *,
    extraction: CreatorTaxonomyExtraction,
    content_input: dict,
    annotation_role: str,
    prompt_version: str,
    evidence_source_text: str,
    pipeline_run_id: str,
) -> list[dict]:

    rows = []

    for label in extraction.labels:

        evidence_valid, evidence_validation_type = (
            validate_evidence(
                label.evidence_span,
                evidence_source_text,
            )
        )

        rows.append(
            {
                "annotation_id":
                    make_annotation_id(),

                "pipeline_run_id":
                    pipeline_run_id,

                "content_id":
                    content_input["content_id"],

                "target_type":
                    "content",

                "target_id":
                    content_input["content_id"],

                "annotation_role":
                    annotation_role,

                "raw_label":
                    label.raw_label.strip(),

                "normalized_label":
                    None,

                "taxonomy_category":
                    label.taxonomy_category,

                "taxonomy_version":
                    TAXONOMY_VERSION,

                "label_confidence":
                    label.confidence,

                "source_field":
                    label.source_field,

                "evidence_span":
                    label.evidence_span,

                "evidence_valid":
                    evidence_valid,

                "evidence_validation_type":
                    evidence_validation_type,

                "explanation":
                    label.explanation,

                "annotation_source":
                    "llm",

                "model_name":
                    MODEL_NAME,

                "prompt_version":
                    prompt_version,

                "framing_char_count":
                    content_input["framing_char_count"],

                "full_content_char_count":
                    content_input["full_content_char_count"],

                "created_at":
                    utc_now_iso(),
            }
        )

    return rows


# ============================================================
# SIMPLE OVERLAP ANALYSIS
# ============================================================

def tokenize_label(text: str) -> set[str]:

    text = text.lower()

    words = re.findall(
        r"[a-zA-Z가-힣]+",
        text
    )

    stopwords = {
        "and",
        "the",
        "a",
        "an",
        "of",
        "to",
        "with",
        "among",
        "through",
        "in",
        "for",
    }

    return {
        word
        for word in words
        if word not in stopwords
    }


def lexical_similarity(
    a: str,
    b: str,
) -> float:

    set_a = tokenize_label(a)
    set_b = tokenize_label(b)

    if not set_a or not set_b:
        return 0.0

    intersection = len(
        set_a & set_b
    )

    union = len(
        set_a | set_b
    )

    return (
        intersection / union
        if union
        else 0.0
    )


def best_overlap_score(
    source_labels: list[str],
    target_labels: list[str],
) -> float:

    if not source_labels or not target_labels:
        return 0.0

    best_scores = []

    for source in source_labels:

        best = max(
            lexical_similarity(
                source,
                target,
            )
            for target in target_labels
        )

        best_scores.append(best)

    return (
        sum(best_scores)
        / len(best_scores)
    )


# ============================================================
# MAIN
# ============================================================

def run_ablation():

    pipeline_run_id = (
        "3a1_"
        + datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")
    )

    print(
        "=========================================="
    )
    print(
        "Sprint 2 Deliverable 3A.1"
    )
    print(
        "Intent Scope Ablation"
    )
    print(
        "=========================================="
    )

    print(
        f"Pipeline run : {pipeline_run_id}"
    )

    print(
        f"Model        : {MODEL_NAME}"
    )

    content_df, transcript_df = (
        load_source_data()
    )

    all_rows = []
    summary_rows = []

    for _, row in content_df.iterrows():

        content_input = build_inputs(
            row,
            transcript_df,
        )

        content_id = content_input[
            "content_id"
        ]

        print()
        print(
            f"[INFO] Processing {content_id}"
        )

        print(
            f"       Framing chars: "
            f"{content_input['framing_char_count']:,}"
        )

        print(
            f"       Full chars   : "
            f"{content_input['full_content_char_count']:,}"
        )

        # ------------------------------------------
        # OLD INTENT
        # ------------------------------------------

        print(
            "[INFO]   Running old intent..."
        )

        old_intent = extract_old_intent(
            content_input
        )

        old_rows = extraction_to_rows(
            extraction=old_intent,
            content_input=content_input,
            annotation_role=(
                "inferred_creator_intent_old"
            ),
            prompt_version=(
                OLD_INTENT_PROMPT_VERSION
            ),
            evidence_source_text=(
                content_input[
                    "full_content_text"
                ]
            ),
            pipeline_run_id=pipeline_run_id,
        )

        all_rows.extend(old_rows)

        # ------------------------------------------
        # NEW INTENT
        # ------------------------------------------

        print(
            "[INFO]   Running new framing-only intent..."
        )

        new_intent = extract_new_intent(
            content_input
        )

        new_rows = extraction_to_rows(
            extraction=new_intent,
            content_input=content_input,
            annotation_role=(
                "inferred_content_intent_new"
            ),
            prompt_version=(
                NEW_INTENT_PROMPT_VERSION
            ),
            evidence_source_text=(
                content_input[
                    "framing_text"
                ]
            ),
            pipeline_run_id=pipeline_run_id,
        )

        all_rows.extend(new_rows)

        # ------------------------------------------
        # CONTENT UNDERSTANDING
        # ------------------------------------------

        print(
            "[INFO]   Running content understanding..."
        )

        content_result = (
            extract_content_understanding(
                content_input
            )
        )

        content_rows = extraction_to_rows(
            extraction=content_result,
            content_input=content_input,
            annotation_role=(
                "ai_content_understanding"
            ),
            prompt_version=(
                CONTENT_UNDERSTANDING_PROMPT_VERSION
            ),
            evidence_source_text=(
                content_input[
                    "full_content_text"
                ]
            ),
            pipeline_run_id=pipeline_run_id,
        )

        all_rows.extend(content_rows)

        # ------------------------------------------
        # SIMPLE ABALATION SUMMARY
        # ------------------------------------------

        old_labels = [
            x.raw_label
            for x in old_intent.labels
        ]

        new_labels = [
            x.raw_label
            for x in new_intent.labels
        ]

        content_labels = [
            x.raw_label
            for x in content_result.labels
        ]

        old_overlap = (
            best_overlap_score(
                old_labels,
                content_labels,
            )
        )

        new_overlap = (
            best_overlap_score(
                new_labels,
                content_labels,
            )
        )

        summary_rows.append(
            {
                "content_id":
                    content_id,

                "title":
                    row.get("title"),

                "old_intent_label_count":
                    len(old_labels),

                "new_intent_label_count":
                    len(new_labels),

                "content_label_count":
                    len(content_labels),

                "old_intent_labels":
                    " | ".join(old_labels),

                "new_intent_labels":
                    " | ".join(new_labels),

                "content_understanding_labels":
                    " | ".join(content_labels),

                "old_vs_content_lexical_overlap":
                    round(old_overlap, 3),

                "new_vs_content_lexical_overlap":
                    round(new_overlap, 3),

                "overlap_change":
                    round(
                        new_overlap
                        - old_overlap,
                        3,
                    ),

                "created_at":
                    utc_now_iso(),
            }
        )

        print(
            f"[INFO]   Old intent labels : "
            f"{len(old_labels)}"
        )

        print(
            f"[INFO]   New intent labels : "
            f"{len(new_labels)}"
        )

        print(
            f"[INFO]   Content labels    : "
            f"{len(content_labels)}"
        )

        print(
            f"[INFO]   Old overlap       : "
            f"{old_overlap:.3f}"
        )

        print(
            f"[INFO]   New overlap       : "
            f"{new_overlap:.3f}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    annotation_df = pd.DataFrame(
        all_rows
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    annotation_df.to_csv(
        OUTPUT_ANNOTATIONS,
        index=False,
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    print()
    print(
        "=========================================="
    )
    print(
        "3A.1 Ablation Complete"
    )
    print(
        "=========================================="
    )

    print(
        f"Annotation rows: "
        f"{len(annotation_df)}"
    )

    print(
        f"Summary rows   : "
        f"{len(summary_df)}"
    )

    if not annotation_df.empty:

        evidence_pass = (
            annotation_df[
                "evidence_valid"
            ].mean()
        )

        print(
            f"Evidence pass  : "
            f"{evidence_pass:.1%}"
        )

    print()
    print(
        f"Saved: {OUTPUT_ANNOTATIONS}"
    )

    print(
        f"Saved: {OUTPUT_SUMMARY}"
    )


if __name__ == "__main__":
    run_ablation()