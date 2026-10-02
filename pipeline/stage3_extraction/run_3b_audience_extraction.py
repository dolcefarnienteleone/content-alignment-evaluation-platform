from __future__ import annotations

import os
import re
import time
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

AUDIENCE_COMMENT_CSV = Path(
    "data/raw/audience_comment.csv"
)

OUTPUT_DIR = Path(
    "data/processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ANNOTATION_OUTPUT = (
    OUTPUT_DIR
    / "taxonomy_annotation_raw_3b.csv"
)

RUN_LOG_OUTPUT = (
    OUTPUT_DIR
    / "taxonomy_extraction_3b_run_log.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "taxonomy_extraction_3b_summary.csv"
)

COMMENT_STATUS_OUTPUT = (
    OUTPUT_DIR
    / "audience_comment_annotation_status_3b.csv"
)


# ------------------------------------------------------------
# MODEL / VERSION CONFIG
# ------------------------------------------------------------

MODEL_NAME = os.getenv(
    "AUDIENCE_EXTRACTION_MODEL",
    "gpt-5.6-terra",
)

TAXONOMY_VERSION = (
    "audience_perception_ontology_v0.1"
)

PROMPT_VERSION = (
    "audience_perception_extraction_v0.1"
)

PIPELINE_VERSION = "3b_v0.1"


# ------------------------------------------------------------
# BATCH CONFIG
# ------------------------------------------------------------

BATCH_SIZE = 15

MAX_COMMENTS_PER_CONTENT = 100

# Keep replies because they are part of audience perception.
INCLUDE_REPLIES = True

# Avoid sending extremely large batches.
MAX_BATCH_CHARS = 20_000

SLEEP_BETWEEN_CALLS_SECONDS = 0.5


# ============================================================
# OPENAI SETUP
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. "
        "Add it to your .env file."
    )

client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ============================================================
# AUDIENCE ONTOLOGY
# ============================================================

AUDIENCE_CATEGORIES = {

    "Emotional Response": (
        "The emotional reaction expressed by the audience, "
        "such as comfort, healing, happiness, pride, sadness, "
        "being touched, heartbreak, excitement, or anger."
    ),

    "Relationship Perception": (
        "How audiences perceive relationships between creators "
        "or members, including friendship, team love, chemistry, "
        "family-like bonds, affection, support, or closeness."
    ),

    "Creator Perception": (
        "How audiences perceive or describe the creator, "
        "including authenticity, humor, naturalness, warmth, "
        "inspiration, personality, professionalism, or image."
    ),

    "Behavioral Response": (
        "An expressed audience action or behavioral intention, "
        "such as wanting more content, replaying, recommending, "
        "buying, searching for related content, or following."
    ),

    "Discussion Topics": (
        "Specific elements of the content that the audience "
        "focuses on, such as performance, vocals, outfit, "
        "lyrics, story, production, visuals, choreography, "
        "or specific scenes."
    ),

    "Community Signals": (
        "Signals of fan or community culture, including memes, "
        "inside jokes, fan theories, shipping, running jokes, "
        "shared language, fandom references, or recurring lore."
    ),
}


CATEGORY_NAMES = list(
    AUDIENCE_CATEGORIES.keys()
)


CategoryLiteral = Literal[
    "Emotional Response",
    "Relationship Perception",
    "Creator Perception",
    "Behavioral Response",
    "Discussion Topics",
    "Community Signals",
]


# ============================================================
# STRUCTURED OUTPUT
# ============================================================

class AudienceLabel(BaseModel):

    comment_id: str = Field(
        description=(
            "The exact supplied comment_id "
            "for the comment being annotated."
        )
    )

    raw_label: str = Field(
        description=(
            "Concise natural-language description "
            "of the audience signal. "
            "Do not force wording to match "
            "an existing ontology label."
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

    evidence_span: str = Field(
        description=(
            "Exact substring copied from the target "
            "comment text. Never paraphrase."
        )
    )

    explanation: str = Field(
        description=(
            "Short explanation of why the target "
            "comment supports the label."
        )
    )


class AudienceBatchExtraction(BaseModel):

    labels: list[AudienceLabel]


# ============================================================
# UTILS
# ============================================================

def utc_now_iso() -> str:

    return datetime.now(
        timezone.utc
    ).isoformat()


def make_annotation_id() -> str:

    return (
        f"ann_{uuid.uuid4().hex[:16]}"
    )


def make_batch_id(
    content_id: str,
    batch_number: int,
) -> str:

    safe_content_id = re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        content_id,
    )

    return (
        f"3b_{safe_content_id}_"
        f"{batch_number:04d}"
    )


def clean_string(value) -> str:

    if pd.isna(value):
        return ""

    return str(value).strip()


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
# LOAD COMMENTS
# ============================================================

def load_comments() -> pd.DataFrame:

    if not AUDIENCE_COMMENT_CSV.exists():

        raise FileNotFoundError(
            f"Missing: {AUDIENCE_COMMENT_CSV}"
        )

    df = pd.read_csv(
        AUDIENCE_COMMENT_CSV
    )

    required_columns = {
        "comment_id",
        "content_id",
        "comment_text",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "audience_comment.csv is missing "
            f"required columns: {sorted(missing)}"
        )

    df["comment_id"] = (
        df["comment_id"]
        .astype(str)
        .str.strip()
    )

    df["content_id"] = (
        df["content_id"]
        .astype(str)
        .str.strip()
    )

    df["comment_text"] = (
        df["comment_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Remove empty rows.
    df = df[
        df["comment_text"] != ""
    ].copy()

    if (
        not INCLUDE_REPLIES
        and "is_reply" in df.columns
    ):

        df = df[
            ~df["is_reply"].fillna(False)
        ].copy()

    # Preserve existing ingestion order.
    df["_source_order"] = range(
        len(df)
    )

    return df


# ============================================================
# PARENT COMMENT CONTEXT
# ============================================================

def build_comment_lookup(
    comments_df: pd.DataFrame,
) -> dict[str, str]:

    return dict(
        zip(
            comments_df["comment_id"],
            comments_df["comment_text"],
        )
    )


def get_parent_text(
    row: pd.Series,
    comment_lookup: dict[str, str],
) -> str:

    parent_id = row.get(
        "parent_comment_id"
    )

    if pd.isna(parent_id):
        return ""

    parent_id = str(
        parent_id
    ).strip()

    if not parent_id:
        return ""

    return comment_lookup.get(
        parent_id,
        "",
    )


# ============================================================
# COMMENT SELECTION
# ============================================================

def select_comments_per_content(
    comments_df: pd.DataFrame,
) -> pd.DataFrame:

    selected_parts = []

    for content_id, group in (
        comments_df.groupby(
            "content_id",
            sort=False,
        )
    ):

        group = group.sort_values(
            "_source_order"
        )

        selected = group.head(
            MAX_COMMENTS_PER_CONTENT
        )

        selected_parts.append(
            selected
        )

    if not selected_parts:

        return pd.DataFrame()

    return pd.concat(
        selected_parts,
        ignore_index=True,
    )


# ============================================================
# BATCH BUILDING
# ============================================================

def build_batches(
    content_comments: pd.DataFrame,
) -> list[pd.DataFrame]:

    batches = []

    current_rows = []
    current_chars = 0

    for _, row in (
        content_comments.iterrows()
    ):

        comment_text = clean_string(
            row["comment_text"]
        )

        estimated_chars = (
            len(comment_text)
            + 200
        )

        if (
            current_rows
            and (
                len(current_rows)
                >= BATCH_SIZE
                or
                current_chars
                + estimated_chars
                > MAX_BATCH_CHARS
            )
        ):

            batches.append(
                pd.DataFrame(
                    current_rows
                )
            )

            current_rows = []
            current_chars = 0

        current_rows.append(
            row.to_dict()
        )

        current_chars += (
            estimated_chars
        )

    if current_rows:

        batches.append(
            pd.DataFrame(
                current_rows
            )
        )

    return batches


# ============================================================
# PROMPT HELPERS
# ============================================================

def taxonomy_definition_text() -> str:

    return "\n".join(
        f"- {category}: {definition}"
        for (
            category,
            definition,
        )
        in AUDIENCE_CATEGORIES.items()
    )


def system_prompt() -> str:

    return f"""
You are performing audience-perception annotation for creator content.

Your job is to identify meaningful audience signals from individual
comments and replies.

Audience taxonomy categories:

{taxonomy_definition_text()}

IMPORTANT RULES:

1. Each label MUST refer to exactly one supplied comment_id.

2. A comment may receive:
   - zero labels,
   - one label,
   - or multiple labels.

3. DO NOT force every comment into the taxonomy.

4. Comments such as:
   "first",
   random spam,
   unrelated promotion,
   or text without meaningful interpretable response
   should normally receive zero labels.

5. Emoji-only comments should receive a label only when the emotional
   meaning is sufficiently clear. Do not over-interpret ambiguous emojis.

6. Distinguish TOPIC from RESPONSE.

   Example:
   "That outfit at 3:22"
   may support:
       Discussion Topics → outfit

   It does NOT automatically support:
       Emotional Response
       Creator Perception

7. A statement may support multiple categories when independently
   justified.

   Example:
   "I love how they always take care of each other. This made me cry."

   may support:
       Relationship Perception → mutual care
       Emotional Response → emotionally touched

8. raw_label should be concise natural language.
   Do NOT force raw_label to use an existing ontology label.

9. evidence_span MUST be an exact substring from the TARGET COMMENT.
   Never paraphrase evidence.

10. Parent comment text may be provided as CONTEXT for a reply.
    It must NOT be used as evidence for the reply annotation.

11. Do not infer demographic, psychological, or private attributes
    about the commenter.

12. Do not use external knowledge about the artist, fandom, or content.

13. Do not infer sentiment merely from popularity, likes, or ranking.

14. Avoid redundant labels describing essentially the same signal.

15. Confidence is evidence strength:

    0.90–1.00:
        directly and explicitly expressed

    0.75–0.89:
        strongly supported with minor interpretation

    0.60–0.74:
        plausible but requires noticeable interpretation

    below 0.60:
        weak or ambiguous

16. Prefer no label over a speculative label.

17. comment_id MUST exactly match one of the supplied IDs.
""".strip()


# ============================================================
# BUILD MODEL INPUT
# ============================================================

def build_batch_input(
    batch_df: pd.DataFrame,
    comment_lookup: dict[str, str],
) -> str:

    blocks = []

    for _, row in (
        batch_df.iterrows()
    ):

        comment_id = row[
            "comment_id"
        ]

        comment_text = clean_string(
            row[
                "comment_text"
            ]
        )

        parent_text = get_parent_text(
            row,
            comment_lookup,
        )

        block = (
            f"COMMENT_ID: {comment_id}\n"
            f"TARGET_COMMENT:\n"
            f"{comment_text}"
        )

        if parent_text:

            block += (
                "\n\nPARENT_COMMENT_CONTEXT:\n"
                f"{parent_text}"
            )

        blocks.append(
            block
        )

    return (
        "\n\n"
        "==============================\n\n"
        .join(blocks)
    )


# ============================================================
# LLM EXTRACTION
# ============================================================

def extract_audience_batch(
    batch_df: pd.DataFrame,
    comment_lookup: dict[str, str],
) -> AudienceBatchExtraction:

    batch_text = build_batch_input(
        batch_df,
        comment_lookup,
    )

    response = client.responses.parse(

        model=MODEL_NAME,

        input=[
            {
                "role": "system",
                "content": system_prompt(),
            },
            {
                "role": "user",
                "content": (
                    "Annotate the following audience "
                    "comments.\n\n"
                    f"{batch_text}"
                ),
            },
        ],

        text_format=(
            AudienceBatchExtraction
        ),
    )

    if response.output_parsed is None:

        raise RuntimeError(
            "No parsed structured output "
            "returned for audience batch."
        )

    return response.output_parsed


# ============================================================
# VALIDATE + FLATTEN
# ============================================================

def extraction_to_rows(
    *,
    extraction: AudienceBatchExtraction,
    batch_df: pd.DataFrame,
    batch_id: str,
    pipeline_run_id: str,
) -> tuple[
    list[dict],
    dict[str, int],
]:

    rows = []

    comment_text_map = dict(
        zip(
            batch_df[
                "comment_id"
            ],
            batch_df[
                "comment_text"
            ],
        )
    )

    valid_comment_ids = set(
        comment_text_map.keys()
    )

    label_counts = {
        comment_id: 0
        for comment_id
        in valid_comment_ids
    }

    seen_keys = set()

    for label in extraction.labels:

        comment_id = (
            label.comment_id.strip()
        )

        # ----------------------------------------------------
        # Prevent hallucinated IDs.
        # ----------------------------------------------------

        if (
            comment_id
            not in valid_comment_ids
        ):

            print(
                "[WARNING] Model returned unknown "
                f"comment_id: {comment_id}. Skipped."
            )

            continue

        source_text = (
            comment_text_map[
                comment_id
            ]
        )

        (
            evidence_valid,
            evidence_validation_type,
        ) = validate_evidence(
            label.evidence_span,
            source_text,
        )

        # ----------------------------------------------------
        # Deduplicate identical model annotations.
        # ----------------------------------------------------

        dedupe_key = (
            comment_id,
            label.raw_label
            .strip()
            .casefold(),
            label.taxonomy_category,
        )

        if dedupe_key in seen_keys:
            continue

        seen_keys.add(
            dedupe_key
        )

        rows.append(
            {
                "annotation_id":
                    make_annotation_id(),

                "pipeline_run_id":
                    pipeline_run_id,

                "pipeline_version":
                    PIPELINE_VERSION,

                "batch_id":
                    batch_id,

                "content_id":
                    batch_df[
                        "content_id"
                    ].iloc[0],

                "target_type":
                    "comment",

                "target_id":
                    comment_id,

                "comment_id":
                    comment_id,

                "annotation_role":
                    "audience_perception",

                "raw_label":
                    label.raw_label.strip(),

                # Deliverable 4
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
                    "comment",

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
                    PROMPT_VERSION,

                "annotator_id":
                    f"model:{MODEL_NAME}",

                "review_status":
                    "pending",

                "reviewed_by":
                    None,

                "reviewed_at":
                    None,

                "created_at":
                    utc_now_iso(),
            }
        )

        label_counts[
            comment_id
        ] += 1

    return (
        rows,
        label_counts,
    )


# ============================================================
# COMMENT STATUS TABLE
# ============================================================

def build_comment_status_rows(
    batch_df: pd.DataFrame,
    label_counts: dict[str, int],
    *,
    batch_id: str,
    pipeline_run_id: str,
    batch_status: str,
) -> list[dict]:

    rows = []

    for _, row in (
        batch_df.iterrows()
    ):

        comment_id = row[
            "comment_id"
        ]

        label_count = (
            label_counts.get(
                comment_id,
                0,
            )
        )

        rows.append(
            {
                "pipeline_run_id":
                    pipeline_run_id,

                "batch_id":
                    batch_id,

                "content_id":
                    row[
                        "content_id"
                    ],

                "comment_id":
                    comment_id,

                "is_reply":
                    row.get(
                        "is_reply",
                        False,
                    ),

                "comment_time":
                    row.get(
                        "comment_time"
                    ),

                "hours_after_publish":
                    row.get(
                        "hours_after_publish"
                    ),

                "engagement_stage":
                    row.get(
                        "engagement_stage"
                    ),

                "like_count":
                    row.get(
                        "like_count"
                    ),

                "annotation_status":
                    batch_status,

                "label_count":
                    label_count,

                "has_audience_signal":
                    label_count > 0,

                "processed_at":
                    utc_now_iso(),
            }
        )

    return rows


# ============================================================
# PROCESS ONE CONTENT
# ============================================================

def process_content(
    content_id: str,
    content_comments: pd.DataFrame,
    comment_lookup: dict[str, str],
    pipeline_run_id: str,
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
]:

    batches = build_batches(
        content_comments
    )

    annotation_rows = []
    run_log_rows = []
    status_rows = []

    print()
    print(
        f"[INFO] Processing {content_id}"
    )

    print(
        f"       Comments selected: "
        f"{len(content_comments)}"
    )

    print(
        f"       Batches          : "
        f"{len(batches)}"
    )

    for batch_number, batch_df in enumerate(
        batches,
        start=1,
    ):

        batch_id = make_batch_id(
            content_id,
            batch_number,
        )

        try:

            print(
                f"[INFO]   Batch "
                f"{batch_number}/{len(batches)} "
                f"({len(batch_df)} comments)..."
            )

            extraction = (
                extract_audience_batch(
                    batch_df,
                    comment_lookup,
                )
            )

            (
                rows,
                label_counts,
            ) = extraction_to_rows(

                extraction=extraction,

                batch_df=batch_df,

                batch_id=batch_id,

                pipeline_run_id=(
                    pipeline_run_id
                ),
            )

            annotation_rows.extend(
                rows
            )

            status_rows.extend(
                build_comment_status_rows(
                    batch_df,
                    label_counts,
                    batch_id=batch_id,
                    pipeline_run_id=(
                        pipeline_run_id
                    ),
                    batch_status="success",
                )
            )

            run_log_rows.append(
                {
                    "pipeline_run_id":
                        pipeline_run_id,

                    "batch_id":
                        batch_id,

                    "content_id":
                        content_id,

                    "status":
                        "success",

                    "comment_count":
                        len(batch_df),

                    "annotation_count":
                        len(rows),

                    "annotated_comment_count":
                        sum(
                            1
                            for count
                            in label_counts.values()
                            if count > 0
                        ),

                    "zero_label_comment_count":
                        sum(
                            1
                            for count
                            in label_counts.values()
                            if count == 0
                        ),

                    "error_type":
                        None,

                    "error_message":
                        None,

                    "processed_at":
                        utc_now_iso(),
                }
            )

            print(
                f"[INFO]     → "
                f"{len(rows)} annotations"
            )

        except Exception as exc:

            print(
                f"[ERROR]   Batch "
                f"{batch_number} failed: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            empty_counts = {
                comment_id: 0
                for comment_id
                in batch_df[
                    "comment_id"
                ].tolist()
            }

            status_rows.extend(
                build_comment_status_rows(
                    batch_df,
                    empty_counts,
                    batch_id=batch_id,
                    pipeline_run_id=(
                        pipeline_run_id
                    ),
                    batch_status="failed",
                )
            )

            run_log_rows.append(
                {
                    "pipeline_run_id":
                        pipeline_run_id,

                    "batch_id":
                        batch_id,

                    "content_id":
                        content_id,

                    "status":
                        "failed",

                    "comment_count":
                        len(batch_df),

                    "annotation_count":
                        0,

                    "annotated_comment_count":
                        0,

                    "zero_label_comment_count":
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

    return (
        annotation_rows,
        run_log_rows,
        status_rows,
    )


# ============================================================
# BUILD SUMMARY
# ============================================================

def build_summary(
    annotation_df: pd.DataFrame,
    status_df: pd.DataFrame,
) -> pd.DataFrame:

    if status_df.empty:
        return pd.DataFrame()

    summary_rows = []

    for content_id, status_group in (
        status_df.groupby(
            "content_id"
        )
    ):

        if annotation_df.empty:

            annotation_group = (
                pd.DataFrame()
            )

        else:

            annotation_group = (
                annotation_df[
                    annotation_df[
                        "content_id"
                    ]
                    == content_id
                ]
            )

        successful_comments = (
            status_group[
                status_group[
                    "annotation_status"
                ]
                == "success"
            ]
        )

        processed_comment_count = (
            len(
                successful_comments
            )
        )

        annotated_comment_count = int(
            successful_comments[
                "has_audience_signal"
            ].sum()
        )

        zero_label_count = (
            processed_comment_count
            - annotated_comment_count
        )

        if annotation_group.empty:

            annotation_count = 0
            avg_confidence = None
            evidence_pass_rate = None
            category_distribution = ""

        else:

            annotation_count = (
                len(annotation_group)
            )

            avg_confidence = round(
                annotation_group[
                    "label_confidence"
                ].mean(),
                3,
            )

            evidence_pass_rate = round(
                annotation_group[
                    "evidence_valid"
                ].mean(),
                3,
            )

            category_counts = (
                annotation_group[
                    "taxonomy_category"
                ]
                .value_counts()
            )

            category_distribution = (
                " | ".join(
                    f"{category}:{count}"
                    for (
                        category,
                        count,
                    )
                    in category_counts.items()
                )
            )

        summary_rows.append(
            {
                "content_id":
                    content_id,

                "processed_comment_count":
                    processed_comment_count,

                "annotated_comment_count":
                    annotated_comment_count,

                "zero_label_comment_count":
                    zero_label_count,

                "audience_signal_rate":
                    (
                        round(
                            annotated_comment_count
                            / processed_comment_count,
                            3,
                        )
                        if processed_comment_count > 0
                        else None
                    ),

                "annotation_count":
                    annotation_count,

                "avg_labels_per_annotated_comment":
                    (
                        round(
                            annotation_count
                            / annotated_comment_count,
                            3,
                        )
                        if annotated_comment_count > 0
                        else None
                    ),

                "avg_label_confidence":
                    avg_confidence,

                "evidence_pass_rate":
                    evidence_pass_rate,

                "category_distribution":
                    category_distribution,

                "created_at":
                    utc_now_iso(),
            }
        )

    return pd.DataFrame(
        summary_rows
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():

    pipeline_run_id = (
        "3b_"
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
        "Sprint 2 Deliverable 3B"
    )

    print(
        "Audience Perception Extraction"
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
        f"Taxonomy     : "
        f"{TAXONOMY_VERSION}"
    )

    print(
        f"Batch size   : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Max comments : "
        f"{MAX_COMMENTS_PER_CONTENT}"
    )

    comments_df = load_comments()

    selected_df = (
        select_comments_per_content(
            comments_df
        )
    )

    if selected_df.empty:

        raise ValueError(
            "No comments available "
            "for extraction."
        )

    comment_lookup = (
        build_comment_lookup(
            comments_df
        )
    )

    print(
        f"Content count: "
        f"{selected_df['content_id'].nunique()}"
    )

    print(
        f"Comment rows : "
        f"{len(selected_df)}"
    )

    all_annotations = []
    all_logs = []
    all_status = []

    # ========================================================
    # PROCESS CONTENT
    # ========================================================

    for content_id, group in (
        selected_df.groupby(
            "content_id",
            sort=False,
        )
    ):

        (
            annotations,
            logs,
            statuses,
        ) = process_content(

            content_id=content_id,

            content_comments=group,

            comment_lookup=(
                comment_lookup
            ),

            pipeline_run_id=(
                pipeline_run_id
            ),
        )

        all_annotations.extend(
            annotations
        )

        all_logs.extend(
            logs
        )

        all_status.extend(
            statuses
        )

    annotation_df = pd.DataFrame(
        all_annotations
    )

    run_log_df = pd.DataFrame(
        all_logs
    )

    status_df = pd.DataFrame(
        all_status
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
                ]
                .dropna()
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
            ]
            .between(
                0,
                1,
            )
        )

        if invalid_confidence.any():

            raise ValueError(
                "Confidence outside 0–1 detected."
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_df = build_summary(
        annotation_df,
        status_df,
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

    status_df.to_csv(
        COMMENT_STATUS_OUTPUT,
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
        "3B Extraction Complete"
    )

    print(
        "=========================================="
    )

    print(
        f"Processed comments : "
        f"{len(status_df)}"
    )

    print(
        f"Annotations        : "
        f"{len(annotation_df)}"
    )

    if not status_df.empty:

        successful_status = (
            status_df[
                status_df[
                    "annotation_status"
                ]
                == "success"
            ]
        )

        annotated_comments = int(
            successful_status[
                "has_audience_signal"
            ].sum()
        )

        zero_label_comments = (
            len(successful_status)
            - annotated_comments
        )

        print(
            f"Annotated comments : "
            f"{annotated_comments}"
        )

        print(
            f"Zero-label comments: "
            f"{zero_label_comments}"
        )

    if not annotation_df.empty:

        evidence_pass = (
            annotation_df[
                "evidence_valid"
            ].mean()
        )

        avg_confidence = (
            annotation_df[
                "label_confidence"
            ].mean()
        )

        print(
            f"Evidence pass      : "
            f"{evidence_pass:.1%}"
        )

        print(
            f"Avg confidence     : "
            f"{avg_confidence:.3f}"
        )

    if not run_log_df.empty:

        successful_batches = int(
            run_log_df[
                "status"
            ]
            .eq(
                "success"
            )
            .sum()
        )

        failed_batches = int(
            run_log_df[
                "status"
            ]
            .eq(
                "failed"
            )
            .sum()
        )

        print(
            f"Successful batches : "
            f"{successful_batches}"
        )

        print(
            f"Failed batches     : "
            f"{failed_batches}"
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
        f"{COMMENT_STATUS_OUTPUT}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_pipeline()