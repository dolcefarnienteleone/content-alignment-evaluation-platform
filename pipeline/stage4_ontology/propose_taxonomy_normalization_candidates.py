from __future__ import annotations

import os
import time
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

CANDIDATE_CSV = Path(
    "data/processed/taxonomy_normalization_candidates.csv"
)

MEMBER_CSV = Path(
    "data/processed/taxonomy_normalization_candidate_members.csv"
)

OUTPUT_DIR = Path(
    "data/processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


PROPOSAL_OUTPUT = (
    OUTPUT_DIR
    / "taxonomy_normalization_candidates_proposed.csv"
)

RUN_LOG_OUTPUT = (
    OUTPUT_DIR
    / "taxonomy_normalization_proposal_run_log.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "taxonomy_normalization_proposal_summary.csv"
)


# ============================================================
# MODEL CONFIG
# ============================================================

MODEL_NAME = os.getenv(
    "NORMALIZATION_PROPOSAL_MODEL",
    "gpt-5.6-terra",
)

PROMPT_VERSION = (
    "taxonomy_normalization_proposal_v0.1"
)

PIPELINE_VERSION = "4b2_v0.1"

CLUSTERS_PER_BATCH = 12

MAX_MEMBER_LABELS_PER_CLUSTER = 20

SLEEP_BETWEEN_CALLS_SECONDS = 0.5


# ============================================================
# SETUP
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

if not OPENAI_API_KEY:

    raise RuntimeError(
        "OPENAI_API_KEY is missing."
    )


client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ============================================================
# ONTOLOGY V0.1
# ============================================================

CREATOR_ONTOLOGY = {

    "Core Values": [
        "Hope",
        "Growth",
        "Comfort",
        "Authenticity",
        "Friendship",
        "Courage",
        "Support",
        "Consistency",
    ],

    "Personality": [
        "Passionate",
        "Humble",
        "Calm",
        "Funny",
        "Playful",
        "Endurance",
        "Care",
        "Supportive",
        "Melancholy",
        "Sensibility",
    ],

    "Relationship Style": [
        "Supportive",
        "Caring",
        "Team-oriented",
        "Mentor",
        "Lead",
        "Loving",
        "Brotherhood",
        "Family",
        "Inspirational",
    ],

    "Creative Style": [
        "Artistic",
        "Experimental",
        "Performance-focused",
        "Professional",
        "Inspiring",
        "Encouraging",
        "Unexpected",
        "Exotic",
        "Unique",
    ],

    "Communication Style": [
        "Casual",
        "Emotional",
        "Reflective",
        "Educational",
        "Logical",
        "Rap",
        "Informative",
    ],

    "Lifestyle": [
        "Daily Life",
        "Behind-the-scenes",
        "Travel",
        "Hobby",
        "Game",
    ],

    "Visual Identity": [
        "Cute",
        "Cool",
        "Fashion",
        "Luxury",
        "Handsome/Beautiful",
        "Unique",
        "Rabbit-like",
        "Cat-like",
    ],
}


AUDIENCE_ONTOLOGY = {

    "Emotional Response": [
        "Comfort",
        "Healing",
        "Happy",
        "Emotional",
        "Proud",
        "Heartbroken",
        "Touching",
        "Angry",
        "Sad",
    ],

    "Relationship Perception": [
        "Team Love",
        "Friendship",
        "Family",
        "Chemistry",
        "Love",
    ],

    "Creator Perception": [
        "Authentic",
        "Funny",
        "Natural",
        "Inspiring",
    ],

    "Behavioral Response": [
        "Want More",
        "Replay",
        "Recommend",
        "Buy",
        "Find Related",
    ],

    "Discussion Topics": [
        "Performance",
        "Outfit",
        "Lyrics",
        "Story",
    ],

    "Community Signals": [
        "Inside Jokes",
        "Fan Culture",
        "Shipping",
        "Meme",
        "Shared Language",
    ],
}


# ============================================================
# HUMAN-QC POLICY FINDINGS
# ============================================================

POLICY_NOTES = """
Human QC findings that MUST guide ontology proposals:

1. Generic commenter-to-commenter interactions alone are NOT valid
   Community Signals.

   Examples:
   - thank you
   - 감사합니다
   - I agree
   - this is very true
   - greetings
   - generic acknowledgement

   These should normally be REJECTED unless they carry additional
   content-, creator-, or fandom-specific meaning.

2. Emoji-only responses are normally insufficient for a specific
   audience taxonomy signal unless the meaning is unusually explicit.

3. Community Signals should represent shared/content-derived meaning,
   such as:
   - fandom culture
   - memes
   - inside jokes
   - shared language
   - shipping
   - recurring references
   - content-specific callbacks

4. Content-specific humorous callbacks/reference behavior may represent
   a taxonomy gap. A concept similar to "Content Callback / Reference"
   is a valid NEW NODE candidate if supported by recurring clusters.

5. Discussion Topics answer:
   "What content element, scene, subject, performance, or creative
   component is the audience talking about?"

6. Creator Perception answers:
   "What broader characteristic, quality, personality, image, or
   capability is the audience attributing to the creator?"

7. A reaction to one specific scene should not automatically be promoted
   into a stable Creator Perception.

8. Behavioral Response should represent audience actions or expressed
   intentions, such as requesting, replaying, buying, recommending,
   wanting more, or seeking related content.

9. Category assignment should prioritize analytical usefulness, not just
   semantic similarity.
""".strip()


# ============================================================
# STRUCTURED OUTPUT
# ============================================================

OntologyDecision = Literal[
    "existing_node",
    "new_node",
    "ambiguous",
    "reject",
]


class ClusterProposal(BaseModel):

    candidate_id: str = Field(
        description=(
            "Exact candidate_id supplied in the input."
        )
    )

    proposed_normalized_label: str | None = Field(
        description=(
            "Recommended canonical analytical concept. "
            "Null when rejected or too ambiguous."
        )
    )

    proposed_taxonomy_category: str | None = Field(
        description=(
            "Recommended taxonomy category. "
            "May differ from the model-assigned current category "
            "if category correction is justified."
        )
    )

    ontology_decision: OntologyDecision

    proposal_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the ontology proposal, "
            "not a calibrated probability."
        ),
    )

    proposal_reason: str = Field(
        description=(
            "Short explanation of why this canonical concept "
            "and ontology decision are appropriate."
        )
    )

    category_change_reason: str | None = Field(
        description=(
            "Explanation when proposed category differs from "
            "the original category."
        )
    )

    human_review_needed: bool

    possible_existing_node: str | None = Field(
        description=(
            "If ontology_decision is existing_node, provide the "
            "exact existing ontology leaf node. Otherwise null."
        )
    )


class ProposalBatch(BaseModel):

    proposals: list[ClusterProposal]


# ============================================================
# UTILS
# ============================================================

def utc_now_iso() -> str:

    return datetime.now(
        timezone.utc
    ).isoformat()


def clean_text(value) -> str:

    if pd.isna(value):
        return ""

    return str(value).strip()


def ontology_to_text(
    ontology: dict[str, list[str]],
) -> str:

    blocks = []

    for category, labels in (
        ontology.items()
    ):

        blocks.append(
            f"{category}:\n"
            + "\n".join(
                f"  - {label}"
                for label in labels
            )
        )

    return "\n\n".join(
        blocks
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data(
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    if not CANDIDATE_CSV.exists():

        raise FileNotFoundError(
            f"Missing: {CANDIDATE_CSV}"
        )

    if not MEMBER_CSV.exists():

        raise FileNotFoundError(
            f"Missing: {MEMBER_CSV}"
        )

    candidate_df = pd.read_csv(
        CANDIDATE_CSV
    )

    member_df = pd.read_csv(
        MEMBER_CSV
    )

    required_candidate_columns = {
        "candidate_id",
        "inventory_source",
        "annotation_role",
        "taxonomy_category",
        "cluster_size_unique_labels",
        "total_annotation_count",
        "representative_raw_label",
        "mean_pairwise_similarity",
        "cluster_cohesion",
        "is_singleton_cluster",
    }

    missing = (
        required_candidate_columns
        - set(candidate_df.columns)
    )

    if missing:

        raise ValueError(
            "Candidate file missing columns: "
            f"{sorted(missing)}"
        )

    required_member_columns = {
        "candidate_id",
        "raw_label",
        "annotation_count",
        "avg_confidence",
    }

    missing = (
        required_member_columns
        - set(member_df.columns)
    )

    if missing:

        raise ValueError(
            "Member file missing columns: "
            f"{sorted(missing)}"
        )

    candidate_df[
        "candidate_id"
    ] = (
        candidate_df[
            "candidate_id"
        ]
        .astype(str)
        .str.strip()
    )

    member_df[
        "candidate_id"
    ] = (
        member_df[
            "candidate_id"
        ]
        .astype(str)
        .str.strip()
    )

    return (
        candidate_df,
        member_df,
    )


# ============================================================
# BUILD CLUSTER DETAIL
# ============================================================

def build_cluster_detail(
    candidate_row: pd.Series,
    member_df: pd.DataFrame,
) -> str:

    candidate_id = (
        candidate_row[
            "candidate_id"
        ]
    )

    members = member_df[
        member_df[
            "candidate_id"
        ]
        == candidate_id
    ].copy()

    members = (
        members
        .sort_values(
            [
                "annotation_count",
                "avg_confidence",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(
            MAX_MEMBER_LABELS_PER_CLUSTER
        )
    )

    member_lines = []

    for _, member in (
        members.iterrows()
    ):

        member_lines.append(
            "- "
            f"{member['raw_label']} "
            f"(annotations="
            f"{member['annotation_count']}, "
            f"confidence="
            f"{member['avg_confidence']})"
        )

    mean_similarity = (
        candidate_row[
            "mean_pairwise_similarity"
        ]
    )

    if pd.isna(
        mean_similarity
    ):
        mean_similarity_text = (
            "singleton"
        )
    else:
        mean_similarity_text = (
            f"{float(mean_similarity):.4f}"
        )

    return f"""
CANDIDATE_ID:
{candidate_id}

SOURCE:
{candidate_row["inventory_source"]}

ANNOTATION_ROLE:
{candidate_row["annotation_role"]}

CURRENT_CATEGORY:
{candidate_row["taxonomy_category"]}

UNIQUE_RAW_LABELS:
{candidate_row["cluster_size_unique_labels"]}

TOTAL_ANNOTATIONS:
{candidate_row["total_annotation_count"]}

REPRESENTATIVE_RAW_LABEL:
{candidate_row["representative_raw_label"]}

CLUSTER_COHESION:
{candidate_row["cluster_cohesion"]}

MEAN_PAIRWISE_SIMILARITY:
{mean_similarity_text}

RAW LABEL MEMBERS:
{chr(10).join(member_lines)}
""".strip()


# ============================================================
# PROMPT
# ============================================================

def build_system_prompt(
    source: str,
) -> str:

    if source == "3A_content":

        ontology_text = (
            ontology_to_text(
                CREATOR_ONTOLOGY
            )
        )

        ontology_name = (
            "Creator / Content Ontology v0.1"
        )

    elif source == "3B_audience":

        ontology_text = (
            ontology_to_text(
                AUDIENCE_ONTOLOGY
            )
        )

        ontology_name = (
            "Audience Perception Ontology v0.1"
        )

    else:

        raise ValueError(
            f"Unknown source: {source}"
        )

    return f"""
You are helping design a production-oriented ontology for an
AI content-understanding and audience-understanding system.

You are NOT performing raw text extraction.

You are reviewing semantic clusters that were generated from free-form
LLM raw labels.

Your task is to propose a stable canonical analytical concept for each
cluster and decide how that concept relates to the current ontology.

CURRENT ONTOLOGY:

{ontology_name}

{ontology_text}


ONTOLOGY DECISIONS:

1. existing_node

Use when the cluster can be represented well by an EXISTING ontology
leaf node.

- proposed_normalized_label should use that existing node.
- possible_existing_node should contain the exact existing node.

2. new_node

Use when the cluster represents a meaningful, reusable analytical
concept that:
- is relevant to the product,
- is not adequately represented by an existing node,
- and is useful enough to justify adding to ontology v0.2.

Do NOT create highly specific nodes that only describe one person,
one video, one country, or one isolated event when a more general
analytical abstraction is possible.

Example:

"requests Indonesian subtitles"
"request for Spanish subtitles"
"needs English subtitles"

should prefer a reusable concept such as:

"Subtitle Request"

rather than one ontology node for every language.

3. ambiguous

Use when:
- the cluster mixes multiple concepts,
- current evidence is insufficient,
- the proper category is unclear,
- or human review is needed before normalization.

4. reject

Use when the cluster does not represent a useful in-scope analytical
signal, even if the raw language itself is understandable.

Examples can include:
- generic commenter-to-commenter acknowledgements,
- spam,
- greetings,
- low-information social interaction,
- labels generated from over-interpretation.


CANONICAL CONCEPT PRINCIPLES:

- Preserve meaningful analytical distinctions.
- Do not over-generalize.
- Do not create unnecessary wording variants.
- Prefer reusable concepts over video-specific descriptions.
- Separate semantic meaning from entity names when possible.
- Category assignment should reflect analytical purpose.
- The canonical concept should be understandable in a dashboard,
  feature table, or alignment engine.
- Existing ontology nodes should be preferred when they sufficiently
  preserve the meaning.
- Do not force a poor match to the existing ontology.


IMPORTANT HUMAN-QC POLICY:

{POLICY_NOTES}


FINAL IMPORTANT RULE:

Semantic clustering only tells you that labels are linguistically or
conceptually similar.

It does NOT prove that they belong to the same ontology node.

Use product meaning and ontology boundaries, not embedding similarity
alone.
""".strip()


# ============================================================
# CALL MODEL
# ============================================================

def propose_batch(
    batch_df: pd.DataFrame,
    member_df: pd.DataFrame,
) -> ProposalBatch:

    sources = (
        batch_df[
            "inventory_source"
        ]
        .unique()
        .tolist()
    )

    if len(sources) != 1:

        raise ValueError(
            "Each proposal batch must contain "
            "only one inventory_source."
        )

    source = sources[0]

    system_prompt = (
        build_system_prompt(
            source
        )
    )

    cluster_blocks = []

    for _, row in (
        batch_df.iterrows()
    ):

        cluster_blocks.append(
            build_cluster_detail(
                row,
                member_df,
            )
        )

    user_prompt = (
        "Review the following semantic candidate clusters.\n\n"
        + "\n\n"
        + (
            "\n\n"
            "========================================\n\n"
        ).join(
            cluster_blocks
        )
    )

    response = (
        client.responses.parse(

            model=MODEL_NAME,

            input=[
                {
                    "role":
                        "system",

                    "content":
                        system_prompt,
                },
                {
                    "role":
                        "user",

                    "content":
                        user_prompt,
                },
            ],

            text_format=(
                ProposalBatch
            ),
        )
    )

    if response.output_parsed is None:

        raise RuntimeError(
            "No parsed normalization "
            "proposal returned."
        )

    return (
        response.output_parsed
    )


# ============================================================
# CREATE BATCHES
# ============================================================

def create_batches(
    candidate_df: pd.DataFrame,
) -> list[pd.DataFrame]:

    batches = []

    # Keep source + category together.
    grouped = (
        candidate_df.groupby(
            [
                "inventory_source",
                "taxonomy_category",
            ],
            sort=False,
        )
    )

    for _, group in grouped:

        group = (
            group
            .sort_values(
                [
                    "is_singleton_cluster",
                    "total_annotation_count",
                    "cluster_size_unique_labels",
                ],
                ascending=[
                    True,
                    False,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )

        for start in range(
            0,
            len(group),
            CLUSTERS_PER_BATCH,
        ):

            batch = group.iloc[
                start:
                start
                + CLUSTERS_PER_BATCH
            ].copy()

            batches.append(
                batch
            )

    return batches


# ============================================================
# RUN PIPELINE
# ============================================================

def run_pipeline():

    pipeline_run_id = (
        "4b2_"
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
        "Deliverable 4B.2"
    )

    print(
        "LLM-assisted Canonical Concept Proposal"
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

    (
        candidate_df,
        member_df,
    ) = load_data()

    print(
        f"Candidate clusters: "
        f"{len(candidate_df)}"
    )

    batches = create_batches(
        candidate_df
    )

    print(
        f"Proposal batches   : "
        f"{len(batches)}"
    )

    proposals = []
    run_logs = []

    # ========================================================
    # RUN BATCHES
    # ========================================================

    for batch_number, batch_df in enumerate(
        batches,
        start=1,
    ):

        source = (
            batch_df[
                "inventory_source"
            ].iloc[0]
        )

        category = (
            batch_df[
                "taxonomy_category"
            ].iloc[0]
        )

        print(
            f"[INFO] Batch "
            f"{batch_number}/{len(batches)} "
            f"| {source} "
            f"| {category} "
            f"| {len(batch_df)} clusters"
        )

        try:

            result = propose_batch(
                batch_df,
                member_df,
            )

            valid_candidate_ids = set(
                batch_df[
                    "candidate_id"
                ]
            )

            returned_ids = set()

            for proposal in (
                result.proposals
            ):

                candidate_id = (
                    proposal
                    .candidate_id
                    .strip()
                )

                if (
                    candidate_id
                    not in valid_candidate_ids
                ):

                    print(
                        "[WARNING] Unknown "
                        f"candidate_id returned: "
                        f"{candidate_id}"
                    )

                    continue

                returned_ids.add(
                    candidate_id
                )

                proposals.append(
                    {
                        "candidate_id":
                            candidate_id,

                        "pipeline_run_id":
                            pipeline_run_id,

                        "pipeline_version":
                            PIPELINE_VERSION,

                        "proposal_model":
                            MODEL_NAME,

                        "proposal_prompt_version":
                            PROMPT_VERSION,

                        "proposed_normalized_label":
                            (
                                proposal
                                .proposed_normalized_label
                            ),

                        "proposed_taxonomy_category":
                            (
                                proposal
                                .proposed_taxonomy_category
                            ),

                        "ontology_decision":
                            (
                                proposal
                                .ontology_decision
                            ),

                        "possible_existing_node":
                            (
                                proposal
                                .possible_existing_node
                            ),

                        "proposal_confidence":
                            (
                                proposal
                                .proposal_confidence
                            ),

                        "proposal_reason":
                            (
                                proposal
                                .proposal_reason
                            ),

                        "category_change_reason":
                            (
                                proposal
                                .category_change_reason
                            ),

                        "human_review_needed":
                            (
                                proposal
                                .human_review_needed
                            ),

                        "proposal_created_at":
                            utc_now_iso(),
                    }
                )

            missing_ids = (
                valid_candidate_ids
                - returned_ids
            )

            if missing_ids:

                print(
                    "[WARNING] Missing proposals "
                    f"for {len(missing_ids)} "
                    "candidate(s)."
                )

            run_logs.append(
                {
                    "pipeline_run_id":
                        pipeline_run_id,

                    "batch_number":
                        batch_number,

                    "inventory_source":
                        source,

                    "taxonomy_category":
                        category,

                    "candidate_count":
                        len(batch_df),

                    "proposal_count":
                        len(returned_ids),

                    "missing_proposal_count":
                        len(missing_ids),

                    "status":
                        (
                            "success"
                            if not missing_ids
                            else "partial"
                        ),

                    "error_type":
                        None,

                    "error_message":
                        None,

                    "processed_at":
                        utc_now_iso(),
                }
            )

        except Exception as exc:

            print(
                "[ERROR] "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            run_logs.append(
                {
                    "pipeline_run_id":
                        pipeline_run_id,

                    "batch_number":
                        batch_number,

                    "inventory_source":
                        source,

                    "taxonomy_category":
                        category,

                    "candidate_count":
                        len(batch_df),

                    "proposal_count":
                        0,

                    "missing_proposal_count":
                        len(batch_df),

                    "status":
                        "failed",

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
    # BUILD OUTPUT
    # ========================================================

    proposal_df = pd.DataFrame(
        proposals
    )

    if not proposal_df.empty:

        if (
            proposal_df[
                "candidate_id"
            ]
            .duplicated()
            .any()
        ):

            print(
                "[WARNING] Duplicate proposal "
                "candidate IDs detected. "
                "Keeping first."
            )

            proposal_df = (
                proposal_df
                .drop_duplicates(
                    subset=[
                        "candidate_id"
                    ],
                    keep="first",
                )
            )

    proposed_df = (
        candidate_df.merge(
            proposal_df,
            on="candidate_id",
            how="left",
        )
    )

    # --------------------------------------------------------
    # Human-review columns
    # --------------------------------------------------------

    proposed_df[
        "human_final_normalized_label"
    ] = ""

    proposed_df[
        "human_final_taxonomy_category"
    ] = ""

    proposed_df[
        "human_final_ontology_decision"
    ] = ""

    proposed_df[
        "human_review_notes"
    ] = ""

    proposed_df[
        "human_review_status_final"
    ] = "pending"

    # ========================================================
    # SUMMARY
    # ========================================================

    if proposal_df.empty:

        summary_df = pd.DataFrame()

    else:

        summary_rows = []

        # -----------------------------------------------
        # Overall decision distribution
        # -----------------------------------------------

        for decision, group in (
            proposal_df.groupby(
                "ontology_decision"
            )
        ):

            summary_rows.append(
                {
                    "summary_level":
                        "decision",

                    "inventory_source":
                        "ALL",

                    "taxonomy_category":
                        "ALL",

                    "ontology_decision":
                        decision,

                    "candidate_count":
                        len(group),

                    "avg_proposal_confidence":
                        round(
                            group[
                                "proposal_confidence"
                            ].mean(),
                            4,
                        ),

                    "human_review_needed_rate":
                        round(
                            group[
                                "human_review_needed"
                            ].mean(),
                            4,
                        ),
                }
            )

        # -----------------------------------------------
        # Source/category distribution
        # -----------------------------------------------

        merged_summary = (
            candidate_df[
                [
                    "candidate_id",
                    "inventory_source",
                    "taxonomy_category",
                ]
            ]
            .merge(
                proposal_df[
                    [
                        "candidate_id",
                        "ontology_decision",
                        "proposal_confidence",
                        "human_review_needed",
                    ]
                ],
                on="candidate_id",
                how="inner",
            )
        )

        for (
            source,
            category,
        ), group in (
            merged_summary.groupby(
                [
                    "inventory_source",
                    "taxonomy_category",
                ]
            )
        ):

            summary_rows.append(
                {
                    "summary_level":
                        "category",

                    "inventory_source":
                        source,

                    "taxonomy_category":
                        category,

                    "ontology_decision":
                        "ALL",

                    "candidate_count":
                        len(group),

                    "avg_proposal_confidence":
                        round(
                            group[
                                "proposal_confidence"
                            ].mean(),
                            4,
                        ),

                    "human_review_needed_rate":
                        round(
                            group[
                                "human_review_needed"
                            ].mean(),
                            4,
                        ),
                }
            )

        summary_df = pd.DataFrame(
            summary_rows
        )

    run_log_df = pd.DataFrame(
        run_logs
    )

    # ========================================================
    # SAVE
    # ========================================================

    proposed_df.to_csv(
        PROPOSAL_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    run_log_df.to_csv(
        RUN_LOG_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print(
        "=========================================="
    )

    print(
        "4B.2 Proposal Complete"
    )

    print(
        "=========================================="
    )

    print(
        f"Candidates          : "
        f"{len(candidate_df)}"
    )

    print(
        f"Proposals generated : "
        f"{len(proposal_df)}"
    )

    if not proposal_df.empty:

        print()
        print(
            "Ontology decisions:"
        )

        print(
            proposal_df[
                "ontology_decision"
            ]
            .value_counts()
            .to_string()
        )

        print()
        print(
            "Human review needed:"
        )

        print(
            proposal_df[
                "human_review_needed"
            ]
            .value_counts()
            .to_string()
        )

        print()
        print(
            f"Average proposal confidence: "
            f"{proposal_df['proposal_confidence'].mean():.3f}"
        )

    print()
    print(
        "Saved:"
    )

    print(
        f"- {PROPOSAL_OUTPUT}"
    )

    print(
        f"- {RUN_LOG_OUTPUT}"
    )

    print(
        f"- {SUMMARY_OUTPUT}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_pipeline()