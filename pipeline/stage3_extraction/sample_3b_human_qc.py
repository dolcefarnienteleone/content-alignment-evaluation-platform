from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

RANDOM_SEED = 42

COMMENT_CSV = Path(
    "data/raw/audience_comment.csv"
)

ANNOTATION_CSV = Path(
    "data/processed/taxonomy_annotation_raw_3b.csv"
)

STATUS_CSV = Path(
    "data/processed/audience_comment_annotation_status_3b.csv"
)

OUTPUT_DIR = Path(
    "data/evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "audience_human_qc_sample.csv"
)


# ------------------------------------------------------------
# SAMPLE DESIGN
# ------------------------------------------------------------

N_STANDARD = 20
N_MULTI_LABEL = 10
N_LOW_CONFIDENCE = 10
N_ZERO_LABEL = 10

TOTAL_SAMPLE = (
    N_STANDARD
    + N_MULTI_LABEL
    + N_LOW_CONFIDENCE
    + N_ZERO_LABEL
)

LOW_CONFIDENCE_THRESHOLD = 0.80


# ============================================================
# SET RANDOM SEED
# ============================================================

random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    comments_df = pd.read_csv(
        COMMENT_CSV
    )

    annotation_df = pd.read_csv(
        ANNOTATION_CSV
    )

    status_df = pd.read_csv(
        STATUS_CSV
    )

    for df in [
        comments_df,
        annotation_df,
        status_df,
    ]:

        if "comment_id" in df.columns:

            df["comment_id"] = (
                df["comment_id"]
                .astype(str)
                .str.strip()
            )

        if "content_id" in df.columns:

            df["content_id"] = (
                df["content_id"]
                .astype(str)
                .str.strip()
            )

    return (
        comments_df,
        annotation_df,
        status_df,
    )


# ============================================================
# AGGREGATE AI ANNOTATIONS TO COMMENT LEVEL
# ============================================================

def aggregate_annotations(
    annotation_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        content_id,
        comment_id,
    ), group in annotation_df.groupby(
        [
            "content_id",
            "comment_id",
        ]
    ):

        labels = (
            group[
                "raw_label"
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )

        categories = (
            group[
                "taxonomy_category"
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )

        confidences = (
            group[
                "label_confidence"
            ]
            .astype(float)
            .tolist()
        )

        evidence_spans = (
            group[
                "evidence_span"
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )

        explanations = (
            group[
                "explanation"
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )

        annotation_ids = (
            group[
                "annotation_id"
            ]
            .astype(str)
            .tolist()
        )

        label_objects = []

        for (
            annotation_id,
            category,
            label,
            confidence,
            evidence,
            explanation,
        ) in zip(
            annotation_ids,
            categories,
            labels,
            confidences,
            evidence_spans,
            explanations,
        ):

            label_objects.append(
                {
                    "annotation_id":
                        annotation_id,

                    "category":
                        category,

                    "label":
                        label,

                    "confidence":
                        round(
                            confidence,
                            3,
                        ),

                    "evidence":
                        evidence,

                    "explanation":
                        explanation,
                }
            )

        rows.append(
            {
                "content_id":
                    content_id,

                "comment_id":
                    comment_id,

                "ai_label_count":
                    len(group),

                "ai_labels":
                    " | ".join(
                        labels
                    ),

                "ai_categories":
                    " | ".join(
                        categories
                    ),

                "ai_avg_confidence":
                    round(
                        np.mean(
                            confidences
                        ),
                        3,
                    ),

                "ai_min_confidence":
                    round(
                        np.min(
                            confidences
                        ),
                        3,
                    ),

                "ai_evidence":
                    " | ".join(
                        evidence_spans
                    ),

                "ai_annotation_detail":
                    json.dumps(
                        label_objects,
                        ensure_ascii=False,
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# BUILD COMMENT-LEVEL QC POPULATION
# ============================================================

def build_qc_population(
    comments_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    status_df: pd.DataFrame,
) -> pd.DataFrame:

    annotation_agg = (
        aggregate_annotations(
            annotation_df
        )
    )

    # Keep only comments actually processed successfully in 3B.
    status_success = status_df[
        status_df[
            "annotation_status"
        ]
        == "success"
    ].copy()

    population = (
        status_success.merge(
            comments_df,
            on=[
                "content_id",
                "comment_id",
            ],
            how="left",
            suffixes=(
                "",
                "_source",
            ),
        )
    )

    population = (
        population.merge(
            annotation_agg,
            on=[
                "content_id",
                "comment_id",
            ],
            how="left",
        )
    )

    # --------------------------------------------------------
    # Zero-label rows will have NaN annotation fields.
    # --------------------------------------------------------

    population[
        "ai_label_count"
    ] = (
        population[
            "ai_label_count"
        ]
        .fillna(0)
        .astype(int)
    )

    population[
        "ai_labels"
    ] = (
        population[
            "ai_labels"
        ]
        .fillna("")
    )

    population[
        "ai_categories"
    ] = (
        population[
            "ai_categories"
        ]
        .fillna("")
    )

    population[
        "ai_evidence"
    ] = (
        population[
            "ai_evidence"
        ]
        .fillna("")
    )

    population[
        "ai_annotation_detail"
    ] = (
        population[
            "ai_annotation_detail"
        ]
        .fillna("[]")
    )

    population[
        "has_ai_signal"
    ] = (
        population[
            "ai_label_count"
        ]
        > 0
    )

    return population


# ============================================================
# BALANCED SAMPLING HELPER
# ============================================================

def balanced_sample(
    df: pd.DataFrame,
    n: int,
    *,
    used_comment_ids: set[str],
) -> pd.DataFrame:

    available = df[
        ~df[
            "comment_id"
        ].isin(
            used_comment_ids
        )
    ].copy()

    if available.empty:

        return pd.DataFrame()

    # Randomize first.
    available = available.sample(
        frac=1,
        random_state=RANDOM_SEED,
    )

    selected_rows = []

    # --------------------------------------------------------
    # First pass:
    # Try to spread across content IDs.
    # --------------------------------------------------------

    content_ids = (
        available[
            "content_id"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    random.shuffle(
        content_ids
    )

    for content_id in content_ids:

        if (
            len(selected_rows)
            >= n
        ):
            break

        candidates = available[
            available[
                "content_id"
            ]
            == content_id
        ]

        if candidates.empty:
            continue

        row = candidates.sample(
            n=1,
            random_state=(
                RANDOM_SEED
                + len(selected_rows)
            ),
        ).iloc[0]

        selected_rows.append(
            row
        )

        used_comment_ids.add(
            row[
                "comment_id"
            ]
        )

    # --------------------------------------------------------
    # Second pass:
    # Fill remaining slots randomly.
    # --------------------------------------------------------

    remaining_needed = (
        n
        - len(selected_rows)
    )

    if remaining_needed > 0:

        remaining = available[
            ~available[
                "comment_id"
            ].isin(
                used_comment_ids
            )
        ]

        if not remaining.empty:

            take_n = min(
                remaining_needed,
                len(remaining),
            )

            extra = remaining.sample(
                n=take_n,
                random_state=(
                    RANDOM_SEED
                    + 100
                ),
            )

            for _, row in (
                extra.iterrows()
            ):

                selected_rows.append(
                    row
                )

                used_comment_ids.add(
                    row[
                        "comment_id"
                    ]
                )

    if not selected_rows:

        return pd.DataFrame()

    return pd.DataFrame(
        selected_rows
    )


# ============================================================
# SAMPLE GROUPS
# ============================================================

def create_sample(
    population: pd.DataFrame,
) -> pd.DataFrame:

    used_comment_ids = set()

    sample_parts = []

    # --------------------------------------------------------
    # 1. Multi-label
    # --------------------------------------------------------

    multi_pool = population[
        population[
            "ai_label_count"
        ]
        >= 2
    ]

    multi_sample = balanced_sample(
        multi_pool,
        N_MULTI_LABEL,
        used_comment_ids=(
            used_comment_ids
        ),
    )

    multi_sample[
        "qc_sample_group"
    ] = "multi_label"

    sample_parts.append(
        multi_sample
    )

    # --------------------------------------------------------
    # 2. Lower confidence
    # --------------------------------------------------------

    low_conf_pool = population[
        (
            population[
                "ai_label_count"
            ]
            > 0
        )
        &
        (
            population[
                "ai_min_confidence"
            ]
            < LOW_CONFIDENCE_THRESHOLD
        )
    ]

    # If too few <0.80 cases exist,
    # use the lowest-confidence remaining cases.
    available_low_conf = low_conf_pool[
        ~low_conf_pool[
            "comment_id"
        ].isin(
            used_comment_ids
        )
    ]

    if (
        len(available_low_conf)
        < N_LOW_CONFIDENCE
    ):

        low_conf_pool = (
            population[
                population[
                    "ai_label_count"
                ]
                > 0
            ]
            .sort_values(
                "ai_min_confidence",
                ascending=True,
            )
        )

    low_conf_sample = balanced_sample(
        low_conf_pool,
        N_LOW_CONFIDENCE,
        used_comment_ids=(
            used_comment_ids
        ),
    )

    low_conf_sample[
        "qc_sample_group"
    ] = "lower_confidence"

    sample_parts.append(
        low_conf_sample
    )

    # --------------------------------------------------------
    # 3. Zero-label
    # --------------------------------------------------------

    zero_pool = population[
        population[
            "ai_label_count"
        ]
        == 0
    ]

    zero_sample = balanced_sample(
        zero_pool,
        N_ZERO_LABEL,
        used_comment_ids=(
            used_comment_ids
        ),
    )

    zero_sample[
        "qc_sample_group"
    ] = "zero_label"

    sample_parts.append(
        zero_sample
    )

    # --------------------------------------------------------
    # 4. Standard annotated
    #
    # Exclude multi-label and lower-confidence cases where
    # possible so this group represents ordinary cases.
    # --------------------------------------------------------

    standard_pool = population[
        (
            population[
                "ai_label_count"
            ]
            == 1
        )
        &
        (
            population[
                "ai_min_confidence"
            ]
            >= LOW_CONFIDENCE_THRESHOLD
        )
    ]

    standard_sample = balanced_sample(
        standard_pool,
        N_STANDARD,
        used_comment_ids=(
            used_comment_ids
        ),
    )

    standard_sample[
        "qc_sample_group"
    ] = "standard"

    sample_parts.append(
        standard_sample
    )

    sample = pd.concat(
        sample_parts,
        ignore_index=True,
    )

    # Randomize final review order so the human reviewer
    # isn't reviewing all difficult cases together.
    sample = sample.sample(
        frac=1,
        random_state=(
            RANDOM_SEED + 999
        ),
    ).reset_index(
        drop=True
    )

    sample.insert(
        0,
        "qc_id",
        [
            f"QC_{i:03d}"
            for i
            in range(
                1,
                len(sample) + 1
            )
        ],
    )

    return sample


# ============================================================
# BUILD HUMAN REVIEW TEMPLATE
# ============================================================

def build_review_template(
    sample: pd.DataFrame,
) -> pd.DataFrame:

    preferred_columns = [
        # Identification
        "qc_id",
        "qc_sample_group",
        "content_id",
        "comment_id",

        # Source information
        "comment_text",
        "is_reply",
        "like_count",
        "comment_time",
        "engagement_stage",

        # AI prediction
        "ai_label_count",
        "ai_categories",
        "ai_labels",
        "ai_avg_confidence",
        "ai_min_confidence",
        "ai_evidence",
        "ai_annotation_detail",
    ]

    existing_columns = [
        column
        for column
        in preferred_columns
        if column in sample.columns
    ]

    review_df = sample[
        existing_columns
    ].copy()

    # ========================================================
    # HUMAN REVIEW FIELDS
    # ========================================================

    # --------------------------------------------------------
    # Main judgment:
    #
    # correct
    # partially_correct
    # incorrect
    # --------------------------------------------------------

    review_df[
        "human_overall_judgment"
    ] = ""

    # --------------------------------------------------------
    # Does the comment actually contain a meaningful
    # audience signal?
    #
    # yes / no
    # --------------------------------------------------------

    review_df[
        "human_has_signal"
    ] = ""

    # --------------------------------------------------------
    # Are AI taxonomy categories correct?
    #
    # yes / partial / no / not_applicable
    # --------------------------------------------------------

    review_df[
        "human_category_judgment"
    ] = ""

    # --------------------------------------------------------
    # Are raw labels semantically appropriate?
    #
    # yes / partial / no / not_applicable
    # --------------------------------------------------------

    review_df[
        "human_label_judgment"
    ] = ""

    # --------------------------------------------------------
    # If AI missed something or classification is wrong,
    # manually provide preferred category / labels.
    #
    # Use "|" to separate multiple values.
    # --------------------------------------------------------

    review_df[
        "human_corrected_categories"
    ] = ""

    review_df[
        "human_corrected_labels"
    ] = ""

    # --------------------------------------------------------
    # Error classification.
    #
    # none
    # false_positive
    # false_negative
    # wrong_category
    # over_interpretation
    # under_interpretation
    # redundant_label
    # ambiguous
    # other
    # --------------------------------------------------------

    review_df[
        "human_error_type"
    ] = ""

    review_df[
        "human_notes"
    ] = ""

    return review_df


# ============================================================
# PRINT SAMPLE DIAGNOSTICS
# ============================================================

def print_diagnostics(
    sample: pd.DataFrame,
):

    print()
    print(
        "=========================================="
    )
    print(
        "3B.1 Human QC Sample"
    )
    print(
        "=========================================="
    )

    print(
        f"Sample rows : {len(sample)}"
    )

    print()
    print(
        "Sample groups:"
    )

    print(
        sample[
            "qc_sample_group"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Content coverage:"
    )

    print(
        sample[
            "content_id"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    annotated = sample[
        sample[
            "ai_label_count"
        ]
        > 0
    ]

    if not annotated.empty:

        print()
        print(
            "AI category coverage:"
        )

        categories = (
            annotated[
                "ai_categories"
            ]
            .str.split(
                r"\s*\|\s*"
            )
            .explode()
        )

        print(
            categories
            .value_counts()
            .to_string()
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading 3B outputs..."
    )

    (
        comments_df,
        annotation_df,
        status_df,
    ) = load_data()

    population = (
        build_qc_population(
            comments_df,
            annotation_df,
            status_df,
        )
    )

    print(
        f"QC population: "
        f"{len(population)} comments"
    )

    sample = create_sample(
        population
    )

    review_df = (
        build_review_template(
            sample
        )
    )

    review_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print_diagnostics(
        sample
    )

    print()
    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print()
    print(
        "Fill the human_* columns manually, "
        "save the reviewed file as:"
    )

    print(
        "data/evaluation/"
        "audience_human_qc_reviewed.csv"
    )


if __name__ == "__main__":
    main()