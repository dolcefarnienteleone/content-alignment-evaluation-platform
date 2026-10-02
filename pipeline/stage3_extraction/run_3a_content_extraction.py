from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

CONTENT_CSV = Path("data/raw/content.csv")
TRANSCRIPT_CSV = Path("data/raw/transcript_segment.csv")

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATION_OUTPUT = (
    OUTPUT_DIR / "taxonomy_annotation_raw_3a.csv"
)

RUN_LOG_OUTPUT = (
    OUTPUT_DIR / "taxonomy_extraction_3a_run_log.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR / "taxonomy_extraction_3a_summary.csv"
)

SEMANTIC_MATCH_OUTPUT = (
    OUTPUT_DIR / "intent_content_semantic_match_3a.csv"
)


# ------------------------------------------------------------
# MODEL / VERSION CONFIG
# ------------------------------------------------------------

MODEL_NAME = os.getenv(
    "CONTENT_EXTRACTION_MODEL",
    "gpt-5.6-terra",
)

SEMANTIC_MODEL_NAME = (
    "sentence-transformers/"
    "distiluse-base-multilingual-cased-v1"
)

TAXONOMY_VERSION = (
    "creator_identity_ontology_v0.1"
)

CONTENT_INTENT_PROMPT_VERSION = (
    "inferred_content_intent_v0.3"
)

CONTENT_UNDERSTANDING_PROMPT_VERSION = (
    "ai_content_understanding_v0.1"
)

PIPELINE_VERSION = "3a_v0.3"


# ------------------------------------------------------------
# INPUT SCOPE
# ------------------------------------------------------------

MAX_CONTENT_CHARS = 40_000

OPENING_WINDOW_SECONDS = 90
CLOSING_WINDOW_SECONDS = 90

MIN_LABELS = 1
MAX_LABELS = 8

SLEEP_BETWEEN_CALLS_SECONDS = 0.5


# ============================================================
# API SETUP
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. "
        "Add OPENAI_API_KEY=... to your .env file."
    )

client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ============================================================
# SEMANTIC MODEL
# ============================================================

print(
    f"[INFO] Loading semantic model: "
    f"{SEMANTIC_MODEL_NAME}"
)

semantic_model = SentenceTransformer(
    SEMANTIC_MODEL_NAME
)


# ============================================================
# ONTOLOGY CATEGORY DEFINITIONS
# ============================================================

CREATOR_CATEGORIES = {

    "Core Values": (
        "Underlying values, principles, or recurring messages "
        "conveyed by the creator, such as hope, growth, comfort, "
        "authenticity, friendship, or courage."
    ),

    "Personality": (
        "Observable personality characteristics expressed through "
        "the content, such as passionate, humble, calm, funny, "
        "playful, resilient, melancholic, or sensitive."
    ),

    "Relationship Style": (
        "How the creator relates to members or other people, "
        "such as caring, supportive, team-oriented, leadership, "
        "brotherhood, family-like, loving, or inspirational."
    ),

    "Creative Style": (
        "Characteristics of creative expression, production, "
        "or performance, such as artistic, experimental, "
        "performance-focused, professional, spontaneous, "
        "surprising, distinctive, or unconventional."
    ),

    "Communication Style": (
        "How ideas or emotions are communicated, such as casual, "
        "emotional, reflective, educational, logical, informative, "
        "or storytelling-oriented."
    ),

    "Lifestyle": (
        "Lifestyle-oriented signals such as daily life, "
        "behind-the-scenes activities, travel, hobbies, or games."
    ),

    "Visual Identity": (
        "Visual image or aesthetic characteristics, such as cute, "
        "cool, fashionable, luxurious, handsome/beautiful, "
        "distinctive, or animal-like associations."
    ),
}


CATEGORY_NAMES = list(
    CREATOR_CATEGORIES.keys()
)


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
    "opening_transcript",
    "closing_transcript",
    "transcript",
    "multiple",
]


# ============================================================
# STRUCTURED OUTPUT SCHEMA
# ============================================================

class ExtractedCreatorLabel(BaseModel):

    raw_label: str = Field(
        description=(
            "Concise natural-language description of the detected "
            "concept. Do not force wording to match the ontology."
        )
    )

    taxonomy_category: CategoryLiteral

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Heuristic evidence-strength score, "
            "not a calibrated probability."
        ),
    )

    source_field: SourceFieldLiteral

    evidence_span: str = Field(
        description=(
            "Exact substring copied from supplied source content."
        )
    )

    explanation: str = Field(
        description=(
            "Short explanation of why the evidence supports "
            "the extracted label."
        )
    )


class CreatorTaxonomyExtraction(BaseModel):

    labels: list[ExtractedCreatorLabel]


# ============================================================
# GENERAL UTILS
# ============================================================

def utc_now_iso() -> str:

    return datetime.now(
        timezone.utc
    ).isoformat()


def make_annotation_id() -> str:

    return (
        f"ann_{uuid.uuid4().hex[:16]}"
    )


def clean_string(value) -> str:

    if pd.isna(value):
        return ""

    return str(value).strip()


def parse_jsonish_list(
    value,
) -> list[str]:

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

            return [
                str(x)
                for x in parsed
            ]

    except Exception:
        pass

    return [
        x.strip()
        for x in text.split(",")
        if x.strip()
    ]


def normalize_for_evidence_check(
    text: str,
) -> str:

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def validate_evidence(
    evidence_span: str,
    source_text: str,
) -> tuple[bool, str]:

    if not evidence_span:

        return False, "empty"

    if evidence_span in source_text:

        return True, "exact"

    normalized_evidence = (
        normalize_for_evidence_check(
            evidence_span
        )
    )

    normalized_source = (
        normalize_for_evidence_check(
            source_text
        )
    )

    if (
        normalized_evidence
        and normalized_evidence
        in normalized_source
    ):

        return (
            True,
            "normalized_whitespace",
        )

    return False, "failed"


# ============================================================
# LOAD SOURCE DATA
# ============================================================

def load_source_data(
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if not CONTENT_CSV.exists():

        raise FileNotFoundError(
            f"Missing: {CONTENT_CSV}"
        )

    if not TRANSCRIPT_CSV.exists():

        raise FileNotFoundError(
            f"Missing: {TRANSCRIPT_CSV}"
        )

    content_df = pd.read_csv(
        CONTENT_CSV
    )

    transcript_df = pd.read_csv(
        TRANSCRIPT_CSV
    )

    content_df["content_id"] = (
        content_df["content_id"]
        .astype(str)
        .str.strip()
    )

    transcript_df["content_id"] = (
        transcript_df["content_id"]
        .astype(str)
        .str.strip()
    )

    return (
        content_df,
        transcript_df,
    )


# ============================================================
# TRANSCRIPT BUILDING
# ============================================================

def get_transcript_subset(
    transcript_df: pd.DataFrame,
    content_id: str,
) -> pd.DataFrame:

    subset = transcript_df[
        transcript_df["content_id"]
        == content_id
    ].copy()

    if subset.empty:
        return subset

    if (
        "segment_index"
        in subset.columns
    ):

        subset = subset.sort_values(
            "segment_index"
        )

    elif (
        "start_seconds"
        in subset.columns
    ):

        subset = subset.sort_values(
            "start_seconds"
        )

    return subset


def build_full_transcript(
    transcript_df: pd.DataFrame,
    content_id: str,
) -> str:

    subset = get_transcript_subset(
        transcript_df,
        content_id,
    )

    if subset.empty:
        return ""

    texts = (
        subset["text"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    return " ".join(
        x
        for x in texts
        if x
    )


def build_transcript_windows(
    transcript_df: pd.DataFrame,
    content_id: str,
    opening_seconds: int = OPENING_WINDOW_SECONDS,
    closing_seconds: int = CLOSING_WINDOW_SECONDS,
) -> tuple[str, str]:

    subset = get_transcript_subset(
        transcript_df,
        content_id,
    )

    if subset.empty:

        return "", ""

    required_cols = {
        "start_seconds",
        "end_seconds",
        "text",
    }

    if not required_cols.issubset(
        subset.columns
    ):

        raise ValueError(
            "transcript_segment.csv must contain "
            "start_seconds, end_seconds, and text "
            "for v0.3 intent extraction."
        )

    video_end = (
        subset["end_seconds"]
        .max()
    )

    opening_df = subset[
        subset["start_seconds"]
        <= opening_seconds
    ]

    closing_start = max(
        0,
        video_end
        - closing_seconds,
    )

    closing_df = subset[
        subset["end_seconds"]
        >= closing_start
    ]

    opening_text = " ".join(
        opening_df["text"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    closing_text = " ".join(
        closing_df["text"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    return (
        opening_text,
        closing_text,
    )


# ============================================================
# INPUT BUILDING
# ============================================================

def build_content_input(
    row: pd.Series,
    transcript_df: pd.DataFrame,
) -> dict:

    content_id = row[
        "content_id"
    ]

    title = clean_string(
        row.get(
            "title",
            "",
        )
    )

    description = clean_string(
        row.get(
            "description",
            "",
        )
    )

    hashtags = parse_jsonish_list(
        row.get(
            "hashtags",
            "",
        )
    )

    transcript = build_full_transcript(
        transcript_df,
        content_id,
    )

    (
        opening_text,
        closing_text,
    ) = build_transcript_windows(
        transcript_df,
        content_id,
    )

    hashtag_text = " ".join(
        "#"
        + x.lstrip("#")
        for x in hashtags
    )

    # --------------------------------------------------------
    # Intent input:
    # creator/editorial framing proxy
    # --------------------------------------------------------

    framing_text = (
        f"TITLE\n"
        f"{title}\n\n"

        f"DESCRIPTION\n"
        f"{description}\n\n"

        f"HASHTAGS\n"
        f"{hashtag_text}\n\n"

        f"OPENING TRANSCRIPT WINDOW\n"
        f"{opening_text}\n\n"

        f"CLOSING TRANSCRIPT WINDOW\n"
        f"{closing_text}"
    )

    # --------------------------------------------------------
    # Content understanding input:
    # full observable content
    # --------------------------------------------------------

    full_content_text = (
        f"TITLE\n"
        f"{title}\n\n"

        f"DESCRIPTION\n"
        f"{description}\n\n"

        f"HASHTAGS\n"
        f"{hashtag_text}\n\n"

        f"FULL TRANSCRIPT\n"
        f"{transcript}"
    )

    full_content_char_count = (
        len(full_content_text)
    )

    full_content_was_truncated = (
        full_content_char_count
        > MAX_CONTENT_CHARS
    )

    model_full_content_text = (
        full_content_text[
            :MAX_CONTENT_CHARS
        ]
    )

    return {

        "content_id":
            content_id,

        "title":
            title,

        "description":
            description,

        "hashtags":
            hashtags,

        "opening_text":
            opening_text,

        "closing_text":
            closing_text,

        "full_transcript":
            transcript,

        "framing_text":
            framing_text,

        "full_content_text":
            full_content_text,

        "model_full_content_text":
            model_full_content_text,

        "framing_char_count":
            len(framing_text),

        "full_content_char_count":
            full_content_char_count,

        "input_was_truncated":
            full_content_was_truncated,
    }


# ============================================================
# PROMPT HELPERS
# ============================================================

def taxonomy_definition_text(
) -> str:

    return "\n".join(
        f"- {category}: {definition}"
        for (
            category,
            definition,
        )
        in CREATOR_CATEGORIES.items()
    )


def common_extraction_rules(
) -> str:

    return f"""
Taxonomy categories:

{taxonomy_definition_text()}

Extraction rules:

1. Return between {MIN_LABELS} and {MAX_LABELS} labels when
   there is sufficient evidence.

2. raw_label should describe the concept naturally.
   Do NOT force raw_label to match an existing ontology label.

3. taxonomy_category MUST be one of the supplied categories.

4. Every label MUST contain evidence_span.

5. evidence_span MUST be an exact quotation copied from the
   supplied source. Never paraphrase evidence.

6. Prefer specific and defensible labels over generic labels.

7. Do not infer private facts, hidden intentions, or mental states.

8. Do not use audience reactions, popularity metrics, views,
   likes, or external knowledge.

9. Avoid redundant labels.

10. confidence represents evidence strength:
    0.90-1.00 = direct and explicit
    0.75-0.89 = strongly supported with minor interpretation
    0.60-0.74 = plausible but noticeably interpretive
    below 0.60 = weak or ambiguous

11. If evidence is weak, return fewer labels rather than
    inventing a signal.

12. Focus on characteristics communicated by the content as a
    whole unless one individual is explicitly central.
""".strip()


# ============================================================
# EXTRACTION 1:
# INFERRED CONTENT INTENT v0.3
# ============================================================

def infer_content_intent(
    content_input: dict,
) -> CreatorTaxonomyExtraction:

    system_prompt = f"""
You are analyzing PUBLIC CONTENT FRAMING.

Your task is to infer what this content appears positioned or framed
to communicate based only on externally observable creator-side
and editorial framing.

You may use only:

- title
- description
- hashtags
- opening transcript window
- closing transcript window

Do NOT use the full middle/body transcript.

Ask:

"Based on how this content is introduced, packaged, and concluded,
what identity, message, emotional direction, relationship framing,
creative positioning, or lifestyle theme appears intentionally
emphasized?"

This is inferred content intent, NOT confirmed private creator intent.

Important:

- Opening and closing transcript segments may contain program setup,
  editorial framing, reflections, or closing messages.

- Do not infer detailed characteristics that appear only in the
  middle of the video.

- If creator-side framing is sparse, return fewer labels rather than
  filling the result with guesses.

{common_extraction_rules()}
""".strip()

    user_prompt = f"""
CONTENT_ID:
{content_input["content_id"]}

PUBLIC CONTENT FRAMING:
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

        text_format=(
            CreatorTaxonomyExtraction
        ),
    )

    if response.output_parsed is None:

        raise RuntimeError(
            "No parsed structured output "
            "for inferred content intent."
        )

    return response.output_parsed


# ============================================================
# EXTRACTION 2:
# AI CONTENT UNDERSTANDING
# ============================================================

def extract_ai_content_understanding(
    content_input: dict,
) -> CreatorTaxonomyExtraction:

    system_prompt = f"""
You are analyzing what identity and content characteristics are
OBSERVABLY PRESENT in creator content.

Your task is different from inferred content intent.

Ask:

"What characteristics can actually be detected in the content itself,
regardless of whether they were intentionally planned or emphasized?"

Use:

- title
- description
- hashtags
- full transcript

Possible signals include:

- values
- personality presentation
- relationship dynamics
- creative style
- communication style
- lifestyle
- visual identity when explicitly supported

Do NOT speculate about private creator intent.

{common_extraction_rules()}
""".strip()

    user_prompt = f"""
CONTENT_ID:
{content_input["content_id"]}

FULL PUBLIC CONTENT:
{content_input["model_full_content_text"]}
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

        text_format=(
            CreatorTaxonomyExtraction
        ),
    )

    if response.output_parsed is None:

        raise RuntimeError(
            "No parsed structured output "
            "for AI content understanding."
        )

    return response.output_parsed


# ============================================================
# FLATTEN EXTRACTION
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

        (
            evidence_valid,
            evidence_validation_type,
        ) = validate_evidence(
            label.evidence_span,
            evidence_source_text,
        )

        rows.append(
            {
                "annotation_id":
                    make_annotation_id(),

                "pipeline_run_id":
                    pipeline_run_id,

                "pipeline_version":
                    PIPELINE_VERSION,

                "content_id":
                    content_input[
                        "content_id"
                    ],

                "target_type":
                    "content",

                "target_id":
                    content_input[
                        "content_id"
                    ],

                "annotation_role":
                    annotation_role,

                "raw_label":
                    label.raw_label.strip(),

                # Deliverable 4 will populate this.
                "normalized_label":
                    None,

                "taxonomy_category":
                    label.taxonomy_category,

                "taxonomy_subcategory":
                    None,

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

                "model_version":
                    MODEL_NAME,

                "prompt_version":
                    prompt_version,

                "annotator_id":
                    f"model:{MODEL_NAME}",

                "review_status":
                    "pending",

                "reviewed_by":
                    None,

                "reviewed_at":
                    None,

                "framing_char_count":
                    content_input[
                        "framing_char_count"
                    ],

                "full_content_char_count":
                    content_input[
                        "full_content_char_count"
                    ],

                "input_was_truncated":
                    content_input[
                        "input_was_truncated"
                    ],

                "created_at":
                    utc_now_iso(),
            }
        )

    return rows


# ============================================================
# MULTILINGUAL SEMANTIC SIMILARITY
# ============================================================

def compute_similarity_matrix(
    reference_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:

    if (
        reference_df.empty
        or comparison_df.empty
    ):

        return pd.DataFrame()

    reference_labels = (
        reference_df[
            "raw_label"
        ]
        .astype(str)
        .tolist()
    )

    comparison_labels = (
        comparison_df[
            "raw_label"
        ]
        .astype(str)
        .tolist()
    )

    all_labels = (
        reference_labels
        + comparison_labels
    )

    embeddings = semantic_model.encode(
        all_labels,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    reference_embeddings = embeddings[
        :len(reference_labels)
    ]

    comparison_embeddings = embeddings[
        len(reference_labels):
    ]

    rows = []

    for ref_idx, ref_row in (
        reference_df
        .reset_index(drop=True)
        .iterrows()
    ):

        for comp_idx, comp_row in (
            comparison_df
            .reset_index(drop=True)
            .iterrows()
        ):

            similarity = float(
                np.dot(
                    reference_embeddings[
                        ref_idx
                    ],
                    comparison_embeddings[
                        comp_idx
                    ],
                )
            )

            rows.append(
                {
                    "content_id":
                        ref_row[
                            "content_id"
                        ],

                    "reference_annotation_id":
                        ref_row[
                            "annotation_id"
                        ],

                    "reference_label":
                        ref_row[
                            "raw_label"
                        ],

                    "reference_category":
                        ref_row[
                            "taxonomy_category"
                        ],

                    "comparison_annotation_id":
                        comp_row[
                            "annotation_id"
                        ],

                    "comparison_label":
                        comp_row[
                            "raw_label"
                        ],

                    "comparison_category":
                        comp_row[
                            "taxonomy_category"
                        ],

                    "semantic_similarity":
                        round(
                            similarity,
                            4,
                        ),

                    "semantic_model":
                        SEMANTIC_MODEL_NAME,
                }
            )

    return pd.DataFrame(
        rows
    )


def get_best_semantic_matches(
    similarity_df: pd.DataFrame,
) -> pd.DataFrame:

    if similarity_df.empty:
        return pd.DataFrame()

    best_matches = (
        similarity_df
        .sort_values(
            "semantic_similarity",
            ascending=False,
        )
        .groupby(
            [
                "content_id",
                "reference_annotation_id",
            ],
            as_index=False,
        )
        .first()
    )

    return best_matches


# ============================================================
# BUILD CONTENT SUMMARY
# ============================================================

def build_summary(
    annotation_df: pd.DataFrame,
    semantic_match_df: pd.DataFrame,
    content_df: pd.DataFrame,
) -> pd.DataFrame:

    if annotation_df.empty:
        return pd.DataFrame()

    rows = []

    for content_id, group in (
        annotation_df.groupby(
            "content_id"
        )
    ):

        content_match = content_df[
            content_df["content_id"]
            == content_id
        ]

        if content_match.empty:

            title = None
            creator_name = None

        else:

            content_row = (
                content_match.iloc[0]
            )

            title = content_row.get(
                "title"
            )

            creator_name = (
                content_row.get(
                    "creator_name"
                )
            )

        intent = group[
            group[
                "annotation_role"
            ]
            == "inferred_content_intent"
        ]

        understanding = group[
            group[
                "annotation_role"
            ]
            == "ai_content_understanding"
        ]

        content_matches = (
            semantic_match_df[
                semantic_match_df[
                    "content_id"
                ]
                == content_id
            ]
            if not semantic_match_df.empty
            else pd.DataFrame()
        )

        mean_best_similarity = (
            round(
                content_matches[
                    "semantic_similarity"
                ].mean(),
                3,
            )
            if not content_matches.empty
            else None
        )

        min_best_similarity = (
            round(
                content_matches[
                    "semantic_similarity"
                ].min(),
                3,
            )
            if not content_matches.empty
            else None
        )

        max_best_similarity = (
            round(
                content_matches[
                    "semantic_similarity"
                ].max(),
                3,
            )
            if not content_matches.empty
            else None
        )

        rows.append(
            {
                "content_id":
                    content_id,

                "creator_name":
                    creator_name,

                "title":
                    title,

                "inferred_content_intent_label_count":
                    len(intent),

                "ai_content_understanding_label_count":
                    len(understanding),

                "inferred_content_intent_labels":
                    " | ".join(
                        intent[
                            "raw_label"
                        ].tolist()
                    ),

                "ai_content_understanding_labels":
                    " | ".join(
                        understanding[
                            "raw_label"
                        ].tolist()
                    ),

                "intent_avg_confidence":
                    (
                        round(
                            intent[
                                "label_confidence"
                            ].mean(),
                            3,
                        )
                        if not intent.empty
                        else None
                    ),

                "content_avg_confidence":
                    (
                        round(
                            understanding[
                                "label_confidence"
                            ].mean(),
                            3,
                        )
                        if not understanding.empty
                        else None
                    ),

                "evidence_pass_rate":
                    round(
                        group[
                            "evidence_valid"
                        ].mean(),
                        3,
                    ),

                "needs_evidence_review_count":
                    int(
                        (
                            ~group[
                                "evidence_valid"
                            ]
                        ).sum()
                    ),

                # Diagnostic only.
                "mean_best_semantic_similarity":
                    mean_best_similarity,

                "min_best_semantic_similarity":
                    min_best_similarity,

                "max_best_semantic_similarity":
                    max_best_similarity,

                "created_at":
                    utc_now_iso(),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PROCESS SINGLE CONTENT
# ============================================================

def process_content(
    row: pd.Series,
    transcript_df: pd.DataFrame,
    pipeline_run_id: str,
) -> tuple[
    list[dict],
    list[dict],
]:

    content_input = (
        build_content_input(
            row,
            transcript_df,
        )
    )

    content_id = (
        content_input[
            "content_id"
        ]
    )

    print()
    print(
        f"[INFO] Processing "
        f"{content_id}"
    )

    print(
        f"       Framing chars: "
        f"{content_input['framing_char_count']:,}"
    )

    print(
        f"       Full chars   : "
        f"{content_input['full_content_char_count']:,}"
    )

    if (
        content_input[
            "input_was_truncated"
        ]
    ):

        print(
            f"[WARNING] Full content truncated "
            f"to {MAX_CONTENT_CHARS:,} chars."
        )

    annotation_rows = []
    log_rows = []

    # ========================================================
    # A. INFERRED CONTENT INTENT
    # ========================================================

    try:

        print(
            "[INFO]   Extracting "
            "inferred content intent..."
        )

        result = (
            infer_content_intent(
                content_input
            )
        )

        rows = extraction_to_rows(
            extraction=result,

            content_input=(
                content_input
            ),

            annotation_role=(
                "inferred_content_intent"
            ),

            prompt_version=(
                CONTENT_INTENT_PROMPT_VERSION
            ),

            evidence_source_text=(
                content_input[
                    "framing_text"
                ]
            ),

            pipeline_run_id=(
                pipeline_run_id
            ),
        )

        annotation_rows.extend(
            rows
        )

        log_rows.append(
            {
                "pipeline_run_id":
                    pipeline_run_id,

                "content_id":
                    content_id,

                "annotation_role":
                    "inferred_content_intent",

                "status":
                    "success",

                "label_count":
                    len(rows),

                "error_type":
                    None,

                "error_message":
                    None,

                "processed_at":
                    utc_now_iso(),
            }
        )

        print(
            f"[INFO]   → "
            f"{len(rows)} labels"
        )

    except Exception as exc:

        print(
            "[ERROR]  Inferred content "
            f"intent failed: {exc}"
        )

        log_rows.append(
            {
                "pipeline_run_id":
                    pipeline_run_id,

                "content_id":
                    content_id,

                "annotation_role":
                    "inferred_content_intent",

                "status":
                    "failed",

                "label_count":
                    0,

                "error_type":
                    type(exc).__name__,

                "error_message":
                    str(exc),

                "processed_at":
                    utc_now_iso(),
            }
        )

    time.sleep(
        SLEEP_BETWEEN_CALLS_SECONDS
    )

    # ========================================================
    # B. AI CONTENT UNDERSTANDING
    # ========================================================

    try:

        print(
            "[INFO]   Extracting "
            "AI content understanding..."
        )

        result = (
            extract_ai_content_understanding(
                content_input
            )
        )

        rows = extraction_to_rows(
            extraction=result,

            content_input=(
                content_input
            ),

            annotation_role=(
                "ai_content_understanding"
            ),

            prompt_version=(
                CONTENT_UNDERSTANDING_PROMPT_VERSION
            ),

            evidence_source_text=(
                content_input[
                    "model_full_content_text"
                ]
            ),

            pipeline_run_id=(
                pipeline_run_id
            ),
        )

        annotation_rows.extend(
            rows
        )

        log_rows.append(
            {
                "pipeline_run_id":
                    pipeline_run_id,

                "content_id":
                    content_id,

                "annotation_role":
                    "ai_content_understanding",

                "status":
                    "success",

                "label_count":
                    len(rows),

                "error_type":
                    None,

                "error_message":
                    None,

                "processed_at":
                    utc_now_iso(),
            }
        )

        print(
            f"[INFO]   → "
            f"{len(rows)} labels"
        )

    except Exception as exc:

        print(
            "[ERROR]  AI content "
            f"understanding failed: "
            f"{exc}"
        )

        log_rows.append(
            {
                "pipeline_run_id":
                    pipeline_run_id,

                "content_id":
                    content_id,

                "annotation_role":
                    "ai_content_understanding",

                "status":
                    "failed",

                "label_count":
                    0,

                "error_type":
                    type(exc).__name__,

                "error_message":
                    str(exc),

                "processed_at":
                    utc_now_iso(),
            }
        )

    return (
        annotation_rows,
        log_rows,
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():

    pipeline_run_id = (
        "3a_"
        + datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    print(
        "=========================================="
    )
    print(
        "Sprint 2 Deliverable 3A"
    )
    print(
        "Inferred Content Intent v0.3 "
        "+ AI Content Understanding"
    )
    print(
        "=========================================="
    )

    print(
        f"Pipeline run : "
        f"{pipeline_run_id}"
    )

    print(
        f"Model        : "
        f"{MODEL_NAME}"
    )

    print(
        f"Semantic     : "
        f"{SEMANTIC_MODEL_NAME}"
    )

    print(
        f"Taxonomy     : "
        f"{TAXONOMY_VERSION}"
    )

    content_df, transcript_df = (
        load_source_data()
    )

    print(
        f"Content rows : "
        f"{len(content_df)}"
    )

    all_annotations = []
    all_logs = []

    # ========================================================
    # EXTRACTION
    # ========================================================

    for _, row in (
        content_df.iterrows()
    ):

        (
            annotations,
            logs,
        ) = process_content(
            row,
            transcript_df,
            pipeline_run_id,
        )

        all_annotations.extend(
            annotations
        )

        all_logs.extend(
            logs
        )

    annotation_df = pd.DataFrame(
        all_annotations
    )

    run_log_df = pd.DataFrame(
        all_logs
    )

    # ========================================================
    # QC
    # ========================================================

    if not annotation_df.empty:

        if (
            annotation_df[
                "annotation_id"
            ]
            .duplicated()
            .any()
        ):

            raise ValueError(
                "Duplicate annotation_id detected."
            )

        invalid_categories = (
            set(
                annotation_df[
                    "taxonomy_category"
                ].dropna()
            )
            - set(
                CATEGORY_NAMES
            )
        )

        if invalid_categories:

            raise ValueError(
                "Unexpected taxonomy categories: "
                f"{invalid_categories}"
            )

        invalid_confidence = (
            ~annotation_df[
                "label_confidence"
            ].between(
                0,
                1,
            )
        )

        if invalid_confidence.any():

            raise ValueError(
                "Confidence outside 0–1 detected."
            )

    # ========================================================
    # SEMANTIC DIAGNOSTIC
    # ========================================================

    all_similarity_rows = []

    for content_id in (
        annotation_df[
            "content_id"
        ]
        .dropna()
        .unique()
        if not annotation_df.empty
        else []
    ):

        intent_df = annotation_df[
            (
                annotation_df[
                    "content_id"
                ]
                == content_id
            )
            &
            (
                annotation_df[
                    "annotation_role"
                ]
                == "inferred_content_intent"
            )
        ]

        content_understanding_df = (
            annotation_df[
                (
                    annotation_df[
                        "content_id"
                    ]
                    == content_id
                )
                &
                (
                    annotation_df[
                        "annotation_role"
                    ]
                    == "ai_content_understanding"
                )
            ]
        )

        similarity_df = (
            compute_similarity_matrix(
                intent_df,
                content_understanding_df,
            )
        )

        if not similarity_df.empty:

            all_similarity_rows.append(
                similarity_df
            )

    if all_similarity_rows:

        similarity_full_df = (
            pd.concat(
                all_similarity_rows,
                ignore_index=True,
            )
        )

        semantic_match_df = (
            get_best_semantic_matches(
                similarity_full_df
            )
        )

    else:

        similarity_full_df = (
            pd.DataFrame()
        )

        semantic_match_df = (
            pd.DataFrame()
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_df = (
        build_summary(
            annotation_df,
            semantic_match_df,
            content_df,
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    annotation_df.to_csv(
        ANNOTATION_OUTPUT,
        index=False,
    )

    run_log_df.to_csv(
        RUN_LOG_OUTPUT,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    semantic_match_df.to_csv(
        SEMANTIC_MATCH_OUTPUT,
        index=False,
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print(
        "=========================================="
    )
    print(
        "3A Extraction Complete"
    )
    print(
        "=========================================="
    )

    print(
        f"Annotations : "
        f"{len(annotation_df)}"
    )

    if not annotation_df.empty:

        intent_count = (
            annotation_df[
                "annotation_role"
            ]
            .eq(
                "inferred_content_intent"
            )
            .sum()
        )

        content_count = (
            annotation_df[
                "annotation_role"
            ]
            .eq(
                "ai_content_understanding"
            )
            .sum()
        )

        evidence_pass_rate = (
            annotation_df[
                "evidence_valid"
            ]
            .mean()
        )

        print(
            f"Intent labels : "
            f"{intent_count}"
        )

        print(
            f"Content labels: "
            f"{content_count}"
        )

        print(
            f"Evidence pass : "
            f"{evidence_pass_rate:.1%}"
        )

    success_calls = (
        run_log_df[
            "status"
        ]
        .eq(
            "success"
        )
        .sum()
        if not run_log_df.empty
        else 0
    )

    failed_calls = (
        run_log_df[
            "status"
        ]
        .eq(
            "failed"
        )
        .sum()
        if not run_log_df.empty
        else 0
    )

    print(
        f"Successful model calls: "
        f"{success_calls}"
    )

    print(
        f"Failed model calls    : "
        f"{failed_calls}"
    )

    if not semantic_match_df.empty:

        mean_semantic = (
            semantic_match_df[
                "semantic_similarity"
            ].mean()
        )

        print(
            f"Mean best semantic match: "
            f"{mean_semantic:.3f}"
        )

    print()
    print(
        f"Saved: "
        f"{ANNOTATION_OUTPUT}"
    )

    print(
        f"Saved: "
        f"{RUN_LOG_OUTPUT}"
    )

    print(
        f"Saved: "
        f"{SUMMARY_OUTPUT}"
    )

    print(
        f"Saved: "
        f"{SEMANTIC_MATCH_OUTPUT}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_pipeline()