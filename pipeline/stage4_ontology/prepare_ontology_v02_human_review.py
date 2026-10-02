from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = Path(
    "data/processed/ontology_v02_consolidation_candidates.csv"
)

OUTPUT_DIR = Path(
    "data/evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REVIEW_OUTPUT = (
    OUTPUT_DIR
    / "ontology_v02_human_review.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "ontology_v02_human_review_summary.csv"
)


# ============================================================
# REVIEW SAMPLE SIZE
# ============================================================

TARGET_REVIEW_SIZE = 45

# We don't want every singleton just because it was
# labeled "high" by upstream review logic.
MAX_SINGLETON_SHARE = 0.30


# ============================================================
# SUPPORT / RISK THRESHOLDS
# ============================================================

HIGH_SUPPORT_ANNOTATION_COUNT = 5
MEDIUM_SUPPORT_ANNOTATION_COUNT = 3

HIGH_RAW_LABEL_SUPPORT = 5

LOW_CONFIDENCE_THRESHOLD = 0.80

HIGH_HUMAN_REVIEW_RATE = 0.50


# ============================================================
# CATEGORY PRIORITY
#
# These are especially important for downstream Alignment.
# ============================================================

CATEGORY_PRIORITY = {

    # Audience side
    "Creator Perception": 3,
    "Relationship Perception": 3,
    "Emotional Response": 3,

    "Behavioral Response": 2,
    "Community Signals": 2,
    "Discussion Topics": 2,

    # Content / creator side
    "Relationship Style": 3,
    "Core Values": 3,
    "Personality": 3,

    "Creative Style": 2,
    "Communication Style": 2,
    "Lifestyle": 1,
    "Visual Identity": 2,
}


# ============================================================
# HELPERS
# ============================================================

def clean_text(value) -> str:

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_bool(value) -> bool:

    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
        }
    )


def safe_numeric(
    series: pd.Series,
    default=0,
) -> pd.Series:

    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .fillna(default)
    )


# ============================================================
# LOAD
# ============================================================

def load_candidates() -> pd.DataFrame:

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Missing input file: {INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV
    )

    required_columns = {
        "consolidation_id",
        "inventory_source",
        "proposed_taxonomy_category",
        "proposal_cluster_count",
        "unique_proposed_label_count",
        "representative_proposed_label",
        "member_proposed_labels",
        "supporting_candidate_ids",
        "supporting_raw_label_count",
        "raw_annotation_count",
        "avg_proposal_confidence",
        "mean_pairwise_similarity",
        "consolidation_cohesion",
        "human_review_needed_rate",
        "review_priority",
        "is_singleton_consolidation",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "ontology_v02_consolidation_candidates.csv "
            "is missing required columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()

    # --------------------------------------------------------
    # Clean strings
    # --------------------------------------------------------

    text_columns = [
        "consolidation_id",
        "inventory_source",
        "proposed_taxonomy_category",
        "representative_proposed_label",
        "member_proposed_labels",
        "review_priority",
        "consolidation_cohesion",
    ]

    for column in text_columns:

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # Numerics
    # --------------------------------------------------------

    numeric_columns = [
        "proposal_cluster_count",
        "unique_proposed_label_count",
        "supporting_raw_label_count",
        "raw_annotation_count",
        "avg_proposal_confidence",
        "mean_pairwise_similarity",
        "human_review_needed_rate",
    ]

    for column in numeric_columns:

        df[column] = safe_numeric(
            df[column]
        )

    df[
        "is_singleton_consolidation"
    ] = (
        df[
            "is_singleton_consolidation"
        ]
        .apply(
            normalize_bool
        )
    )

    return df


# ============================================================
# BUILD REVIEW SIGNALS
# ============================================================

def add_review_signals(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    # --------------------------------------------------------
    # Support indicators
    # --------------------------------------------------------

    result[
        "has_high_annotation_support"
    ] = (
        result[
            "raw_annotation_count"
        ]
        >= HIGH_SUPPORT_ANNOTATION_COUNT
    )

    result[
        "has_medium_annotation_support"
    ] = (
        result[
            "raw_annotation_count"
        ]
        >= MEDIUM_SUPPORT_ANNOTATION_COUNT
    )

    result[
        "has_high_raw_label_support"
    ] = (
        result[
            "supporting_raw_label_count"
        ]
        >= HIGH_RAW_LABEL_SUPPORT
    )

    result[
        "is_multi_proposal"
    ] = (
        result[
            "proposal_cluster_count"
        ]
        >= 2
    )

    # --------------------------------------------------------
    # Risk indicators
    # --------------------------------------------------------

    result[
        "has_low_proposal_confidence"
    ] = (
        result[
            "avg_proposal_confidence"
        ]
        < LOW_CONFIDENCE_THRESHOLD
    )

    result[
        "has_high_human_review_rate"
    ] = (
        result[
            "human_review_needed_rate"
        ]
        >= HIGH_HUMAN_REVIEW_RATE
    )

    result[
        "has_low_or_medium_cohesion"
    ] = (
        result[
            "consolidation_cohesion"
        ]
        .isin(
            [
                "low",
                "medium",
            ]
        )
    )

    # --------------------------------------------------------
    # Downstream importance
    # --------------------------------------------------------

    result[
        "category_priority_score"
    ] = (
        result[
            "proposed_taxonomy_category"
        ]
        .map(
            CATEGORY_PRIORITY
        )
        .fillna(1)
        .astype(int)
    )

    return result


# ============================================================
# REVIEW SCORE
# ============================================================

def calculate_review_score(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    score = np.zeros(
        len(result),
        dtype=float,
    )

    # ========================================================
    # PRIORITY
    # ========================================================

    score += np.where(
        result[
            "review_priority"
        ]
        == "high",
        4.0,
        0.0,
    )

    score += np.where(
        result[
            "review_priority"
        ]
        == "medium",
        2.0,
        0.0,
    )

    # ========================================================
    # DATA SUPPORT
    # ========================================================

    score += np.where(
        result[
            "has_high_annotation_support"
        ],
        4.0,
        0.0,
    )

    score += np.where(
        (
            ~result[
                "has_high_annotation_support"
            ]
        )
        &
        result[
            "has_medium_annotation_support"
        ],
        2.0,
        0.0,
    )

    score += np.where(
        result[
            "has_high_raw_label_support"
        ],
        2.0,
        0.0,
    )

    # ========================================================
    # CONSOLIDATION SUPPORT
    # ========================================================

    score += np.where(
        result[
            "proposal_cluster_count"
        ]
        >= 3,
        3.0,
        0.0,
    )

    score += np.where(
        (
            result[
                "proposal_cluster_count"
            ]
            == 2
        ),
        1.5,
        0.0,
    )

    # ========================================================
    # UNCERTAINTY / RISK
    # ========================================================

    score += np.where(
        result[
            "has_low_proposal_confidence"
        ],
        2.0,
        0.0,
    )

    score += np.where(
        result[
            "has_high_human_review_rate"
        ],
        2.0,
        0.0,
    )

    score += np.where(
        result[
            "has_low_or_medium_cohesion"
        ],
        1.0,
        0.0,
    )

    # ========================================================
    # DOWNSTREAM IMPORTANCE
    # ========================================================

    score += (
        result[
            "category_priority_score"
        ]
        * 0.75
    )

    # ========================================================
    # SINGLETON PENALTY
    #
    # Do not eliminate them, just reduce automatic priority.
    # ========================================================

    score -= np.where(
        result[
            "is_singleton_consolidation"
        ],
        2.0,
        0.0,
    )

    # Very weak singleton.
    score -= np.where(
        (
            result[
                "is_singleton_consolidation"
            ]
        )
        &
        (
            result[
                "raw_annotation_count"
            ]
            <= 1
        ),
        1.0,
        0.0,
    )

    result[
        "human_review_score"
    ] = np.round(
        score,
        3,
    )

    return result


# ============================================================
# WHY SELECTED
# ============================================================

def build_selection_reason(
    row: pd.Series,
) -> str:

    reasons = []

    if (
        row[
            "review_priority"
        ]
        == "high"
    ):
        reasons.append(
            "high upstream review priority"
        )

    if row[
        "has_high_annotation_support"
    ]:

        reasons.append(
            "high annotation support"
        )

    elif row[
        "has_medium_annotation_support"
    ]:

        reasons.append(
            "moderate annotation support"
        )

    if row[
        "proposal_cluster_count"
    ] >= 3:

        reasons.append(
            "supported by 3+ proposal clusters"
        )

    elif row[
        "proposal_cluster_count"
    ] == 2:

        reasons.append(
            "multi-proposal concept"
        )

    if row[
        "has_high_raw_label_support"
    ]:

        reasons.append(
            "high raw-label support"
        )

    if row[
        "has_low_proposal_confidence"
    ]:

        reasons.append(
            "lower proposal confidence"
        )

    if row[
        "has_high_human_review_rate"
    ]:

        reasons.append(
            "high upstream human-review rate"
        )

    if row[
        "has_low_or_medium_cohesion"
    ]:

        reasons.append(
            "non-high semantic cohesion"
        )

    if row[
        "is_singleton_consolidation"
    ]:

        reasons.append(
            "singleton concept"
        )

    if not reasons:

        reasons.append(
            "coverage / sanity-check candidate"
        )

    return "; ".join(
        reasons
    )


# ============================================================
# SELECT REVIEW SET
# ============================================================

def select_review_candidates(
    df: pd.DataFrame,
) -> pd.DataFrame:

    ranked = (
        df
        .sort_values(
            [
                "human_review_score",
                "raw_annotation_count",
                "proposal_cluster_count",
                "avg_proposal_confidence",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    selected_rows = []

    selected_ids = set()

    max_singletons = math.floor(
        TARGET_REVIEW_SIZE
        * MAX_SINGLETON_SHARE
    )

    singleton_count = 0

    # ========================================================
    # PASS 1
    # Ensure high-value multi-proposal / supported candidates
    # ========================================================

    priority_pool = ranked[
        (
            ranked[
                "is_multi_proposal"
            ]
        )
        |
        (
            ranked[
                "raw_annotation_count"
            ]
            >= HIGH_SUPPORT_ANNOTATION_COUNT
        )
    ]

    for _, row in (
        priority_pool.iterrows()
    ):

        if (
            len(selected_rows)
            >= TARGET_REVIEW_SIZE
        ):
            break

        candidate_id = (
            row[
                "consolidation_id"
            ]
        )

        if candidate_id in selected_ids:
            continue

        selected_rows.append(
            row
        )

        selected_ids.add(
            candidate_id
        )

        if row[
            "is_singleton_consolidation"
        ]:

            singleton_count += 1

    # ========================================================
    # PASS 2
    # Fill remaining with risk-based candidates
    # while limiting singleton dominance
    # ========================================================

    for _, row in (
        ranked.iterrows()
    ):

        if (
            len(selected_rows)
            >= TARGET_REVIEW_SIZE
        ):
            break

        candidate_id = (
            row[
                "consolidation_id"
            ]
        )

        if candidate_id in selected_ids:
            continue

        is_singleton = bool(
            row[
                "is_singleton_consolidation"
            ]
        )

        if (
            is_singleton
            and singleton_count
            >= max_singletons
        ):

            continue

        selected_rows.append(
            row
        )

        selected_ids.add(
            candidate_id
        )

        if is_singleton:

            singleton_count += 1

    # ========================================================
    # PASS 3
    # If still not enough, remove singleton cap.
    # ========================================================

    if (
        len(selected_rows)
        < TARGET_REVIEW_SIZE
    ):

        for _, row in (
            ranked.iterrows()
        ):

            if (
                len(selected_rows)
                >= TARGET_REVIEW_SIZE
            ):
                break

            candidate_id = (
                row[
                    "consolidation_id"
                ]
            )

            if candidate_id in selected_ids:
                continue

            selected_rows.append(
                row
            )

            selected_ids.add(
                candidate_id
            )

    review_df = pd.DataFrame(
        selected_rows
    )

    return review_df


# ============================================================
# ADD CATEGORY COVERAGE
# ============================================================

def ensure_category_coverage(
    review_df: pd.DataFrame,
    full_df: pd.DataFrame,
) -> pd.DataFrame:

    selected_ids = set(
        review_df[
            "consolidation_id"
        ]
    )

    categories_present = set(
        review_df[
            [
                "inventory_source",
                "proposed_taxonomy_category",
            ]
        ]
        .apply(
            tuple,
            axis=1,
        )
    )

    all_categories = (
        full_df[
            [
                "inventory_source",
                "proposed_taxonomy_category",
            ]
        ]
        .drop_duplicates()
    )

    additions = []

    for _, category_row in (
        all_categories.iterrows()
    ):

        key = (
            category_row[
                "inventory_source"
            ],
            category_row[
                "proposed_taxonomy_category"
            ],
        )

        if key in categories_present:
            continue

        pool = full_df[
            (
                full_df[
                    "inventory_source"
                ]
                == key[0]
            )
            &
            (
                full_df[
                    "proposed_taxonomy_category"
                ]
                == key[1]
            )
            &
            (
                ~full_df[
                    "consolidation_id"
                ]
                .isin(
                    selected_ids
                )
            )
        ]

        if pool.empty:
            continue

        best = (
            pool
            .sort_values(
                [
                    "human_review_score",
                    "raw_annotation_count",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .iloc[0]
        )

        additions.append(
            best
        )

        selected_ids.add(
            best[
                "consolidation_id"
            ]
        )

    if additions:

        review_df = pd.concat(
            [
                review_df,
                pd.DataFrame(
                    additions
                ),
            ],
            ignore_index=True,
        )

    return review_df


# ============================================================
# BUILD HUMAN REVIEW TEMPLATE
# ============================================================

def build_review_template(
    review_df: pd.DataFrame,
) -> pd.DataFrame:

    result = review_df.copy()

    # --------------------------------------------------------
    # Selection reason
    # --------------------------------------------------------

    result[
        "selection_reason"
    ] = (
        result.apply(
            build_selection_reason,
            axis=1,
        )
    )

    # --------------------------------------------------------
    # Suggested human review decision values:
    #
    # approve_new_node
    # merge_existing_node
    # reject
    # defer
    # --------------------------------------------------------

    result[
        "human_final_concept"
    ] = ""

    result[
        "human_ontology_decision"
    ] = ""

    result[
        "human_final_category"
    ] = ""

    result[
        "human_merge_target_existing_node"
    ] = ""

    result[
        "human_review_notes"
    ] = ""

    result[
        "human_review_status"
    ] = ""

    # --------------------------------------------------------
    # Reorder for usability
    # --------------------------------------------------------

    preferred_columns = [
        # Review identity
        "consolidation_id",
        "inventory_source",
        "proposed_taxonomy_category",

        # Concept
        "representative_proposed_label",
        "member_proposed_labels",

        # Support
        "proposal_cluster_count",
        "unique_proposed_label_count",
        "supporting_raw_label_count",
        "raw_annotation_count",

        # Reliability
        "avg_proposal_confidence",
        "mean_pairwise_similarity",
        "consolidation_cohesion",
        "human_review_needed_rate",
        "review_priority",

        # Selection
        "human_review_score",
        "selection_reason",
        "is_singleton_consolidation",

        # Traceability
        "supporting_candidate_ids",

        # Human fields
        "human_final_concept",
        "human_ontology_decision",
        "human_final_category",
        "human_merge_target_existing_node",
        "human_review_notes",
        "human_review_status",
    ]

    existing = [
        column
        for column
        in preferred_columns
        if column in result.columns
    ]

    return result[
        existing
    ].copy()


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    review_df: pd.DataFrame,
    full_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    rows.append(
        {
            "summary_level":
                "overall",

            "inventory_source":
                "ALL",

            "taxonomy_category":
                "ALL",

            "full_candidate_count":
                len(full_df),

            "review_candidate_count":
                len(review_df),

            "review_coverage_rate":
                round(
                    len(review_df)
                    / len(full_df),
                    4,
                )
                if len(full_df)
                else None,

            "review_raw_annotation_support":
                int(
                    review_df[
                        "raw_annotation_count"
                    ]
                    .sum()
                ),

            "full_raw_annotation_support":
                int(
                    full_df[
                        "raw_annotation_count"
                    ]
                    .sum()
                ),

            "annotation_support_coverage":
                round(
                    review_df[
                        "raw_annotation_count"
                    ]
                    .sum()
                    /
                    full_df[
                        "raw_annotation_count"
                    ]
                    .sum(),
                    4,
                )
                if (
                    full_df[
                        "raw_annotation_count"
                    ]
                    .sum()
                    > 0
                )
                else None,

            "singleton_count":
                int(
                    review_df[
                        "is_singleton_consolidation"
                    ]
                    .sum()
                ),

            "multi_proposal_count":
                int(
                    (
                        ~review_df[
                            "is_singleton_consolidation"
                        ]
                    )
                    .sum()
                ),
        }
    )

    # --------------------------------------------------------
    # Category
    # --------------------------------------------------------

    grouped = (
        review_df.groupby(
            [
                "inventory_source",
                "proposed_taxonomy_category",
            ],
            dropna=False,
        )
    )

    for (
        source,
        category,
    ), group in grouped:

        full_group = full_df[
            (
                full_df[
                    "inventory_source"
                ]
                == source
            )
            &
            (
                full_df[
                    "proposed_taxonomy_category"
                ]
                == category
            )
        ]

        rows.append(
            {
                "summary_level":
                    "category",

                "inventory_source":
                    source,

                "taxonomy_category":
                    category,

                "full_candidate_count":
                    len(full_group),

                "review_candidate_count":
                    len(group),

                "review_coverage_rate":
                    round(
                        len(group)
                        / len(full_group),
                        4,
                    )
                    if len(full_group)
                    else None,

                "review_raw_annotation_support":
                    int(
                        group[
                            "raw_annotation_count"
                        ]
                        .sum()
                    ),

                "full_raw_annotation_support":
                    int(
                        full_group[
                            "raw_annotation_count"
                        ]
                        .sum()
                    ),

                "annotation_support_coverage":
                    round(
                        group[
                            "raw_annotation_count"
                        ]
                        .sum()
                        /
                        full_group[
                            "raw_annotation_count"
                        ]
                        .sum(),
                        4,
                    )
                    if (
                        full_group[
                            "raw_annotation_count"
                        ]
                        .sum()
                        > 0
                    )
                    else None,

                "singleton_count":
                    int(
                        group[
                            "is_singleton_consolidation"
                        ]
                        .sum()
                    ),

                "multi_proposal_count":
                    int(
                        (
                            ~group[
                                "is_singleton_consolidation"
                            ]
                        )
                        .sum()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PRINT DIAGNOSTICS
# ============================================================

def print_diagnostics(
    review_df: pd.DataFrame,
    full_df: pd.DataFrame,
    summary_df: pd.DataFrame,
):

    print()
    print(
        "=========================================="
    )

    print(
        "Ontology v0.2 — Human Review Preparation"
    )

    print(
        "=========================================="
    )

    print(
        f"Total consolidation candidates : "
        f"{len(full_df)}"
    )

    print(
        f"Selected for human review      : "
        f"{len(review_df)}"
    )

    print(
        f"Singletons in review           : "
        f"{int(review_df['is_singleton_consolidation'].sum())}"
    )

    print(
        f"Multi-proposal concepts        : "
        f"{int((~review_df['is_singleton_consolidation']).sum())}"
    )

    print()
    print(
        "Review priority distribution:"
    )

    print(
        review_df[
            "review_priority"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Category distribution:"
    )

    category_counts = (
        review_df.groupby(
            [
                "inventory_source",
                "proposed_taxonomy_category",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    print(
        category_counts.to_string(
            index=False
        )
    )

    overall = summary_df[
        summary_df[
            "summary_level"
        ]
        == "overall"
    ]

    if not overall.empty:

        row = overall.iloc[0]

        print()
        print(
            "Coverage:"
        )

        print(
            f"Candidate coverage       : "
            f"{row['review_coverage_rate']:.1%}"
        )

        print(
            f"Annotation support cover : "
            f"{row['annotation_support_coverage']:.1%}"
        )

    print()
    print(
        "Top review candidates:"
    )

    top = (
        review_df
        .sort_values(
            [
                "human_review_score",
                "raw_annotation_count",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(
            25
        )
    )

    print(
        top[
            [
                "inventory_source",
                "proposed_taxonomy_category",
                "representative_proposed_label",
                "proposal_cluster_count",
                "supporting_raw_label_count",
                "raw_annotation_count",
                "avg_proposal_confidence",
                "human_review_score",
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading ontology v0.2 "
        "consolidation candidates..."
    )

    candidates_df = (
        load_candidates()
    )

    print(
        f"[INFO] Loaded "
        f"{len(candidates_df)} candidates."
    )

    candidates_df = (
        add_review_signals(
            candidates_df
        )
    )

    candidates_df = (
        calculate_review_score(
            candidates_df
        )
    )

    # ========================================================
    # SELECT
    # ========================================================

    review_df = (
        select_review_candidates(
            candidates_df
        )
    )

    # Ensure every source/category has at least
    # one candidate represented.
    review_df = (
        ensure_category_coverage(
            review_df,
            candidates_df,
        )
    )

    # --------------------------------------------------------
    # Deduplicate just in case.
    # --------------------------------------------------------

    review_df = (
        review_df
        .drop_duplicates(
            subset=[
                "consolidation_id"
            ]
        )
        .sort_values(
            [
                "human_review_score",
                "raw_annotation_count",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    review_template_df = (
        build_review_template(
            review_df
        )
    )

    summary_df = (
        build_summary(
            review_df,
            candidates_df,
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    review_template_df.to_csv(
        REVIEW_OUTPUT,
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

    print_diagnostics(
        review_df,
        candidates_df,
        summary_df,
    )

    print()
    print(
        "Saved:"
    )

    print(
        f"- {REVIEW_OUTPUT}"
    )

    print(
        f"- {SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":

    main()