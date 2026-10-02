from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ANNOTATION_3A_CSV = Path(
    "data/processed/taxonomy_annotation_raw_3a.csv"
)

ANNOTATION_3B_CSV = Path(
    "data/processed/taxonomy_annotation_raw_3b.csv"
)

OUTPUT_DIR = Path(
    "data/processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

INVENTORY_OUTPUT = (
    OUTPUT_DIR
    / "taxonomy_label_inventory.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "taxonomy_label_inventory_summary.csv"
)

EXAMPLES_OUTPUT = (
    OUTPUT_DIR
    / "taxonomy_label_examples.csv"
)


# ------------------------------------------------------------
# INVENTORY SETTINGS
# ------------------------------------------------------------

# Number of evidence examples preserved for each unique raw label.
MAX_EXAMPLES_PER_LABEL = 3

# Labels appearing once are treated as long-tail singleton labels.
LONG_TAIL_FREQUENCY_THRESHOLD = 1


# ============================================================
# HELPERS
# ============================================================

def clean_text(value) -> str:
    """
    Convert a value to normalized display text.
    """

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_label_key(value) -> str:
    """
    Create a lightweight comparison key.

    IMPORTANT:
    This is NOT ontology normalization.

    It only helps group labels whose differences are superficial,
    such as:
        "Playful Group Banter"
        "playful group banter "
    """

    text = clean_text(value)

    text = text.casefold()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def safe_mean(series: pd.Series) -> float | None:

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return None

    return round(
        float(
            numeric.mean()
        ),
        4,
    )


def safe_min(series: pd.Series) -> float | None:

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return None

    return round(
        float(
            numeric.min()
        ),
        4,
    )


def safe_max(series: pd.Series) -> float | None:

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return None

    return round(
        float(
            numeric.max()
        ),
        4,
    )


def safe_rate(
    numerator: int,
    denominator: int,
) -> float | None:

    if denominator == 0:
        return None

    return round(
        numerator / denominator,
        4,
    )


# ============================================================
# LOAD
# ============================================================

def load_annotation_file(
    path: Path,
    source_name: str,
) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing input file: {path}"
        )

    df = pd.read_csv(
        path
    )

    required_columns = {
        "annotation_id",
        "content_id",
        "annotation_role",
        "raw_label",
        "taxonomy_category",
        "label_confidence",
        "evidence_span",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"{path.name} is missing required columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()

    df[
        "inventory_source"
    ] = source_name

    df[
        "raw_label"
    ] = (
        df[
            "raw_label"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df[
        "taxonomy_category"
    ] = (
        df[
            "taxonomy_category"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df[
        "annotation_role"
    ] = (
        df[
            "annotation_role"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df[
        "label_key"
    ] = (
        df[
            "raw_label"
        ]
        .apply(
            normalize_label_key
        )
    )

    # Remove empty labels.
    df = df[
        df[
            "label_key"
        ]
        != ""
    ].copy()

    return df


def load_all_annotations() -> pd.DataFrame:

    df_3a = load_annotation_file(
        ANNOTATION_3A_CSV,
        "3A_content",
    )

    df_3b = load_annotation_file(
        ANNOTATION_3B_CSV,
        "3B_audience",
    )

    combined = pd.concat(
        [
            df_3a,
            df_3b,
        ],
        ignore_index=True,
    )

    if combined[
        "annotation_id"
    ].duplicated().any():

        duplicate_count = int(
            combined[
                "annotation_id"
            ]
            .duplicated()
            .sum()
        )

        print(
            f"[WARNING] Found "
            f"{duplicate_count} duplicate annotation IDs."
        )

    return combined


# ============================================================
# LABEL INVENTORY
# ============================================================

def build_inventory(
    annotation_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    group_columns = [
        "inventory_source",
        "annotation_role",
        "taxonomy_category",
        "label_key",
    ]

    for keys, group in (
        annotation_df.groupby(
            group_columns,
            dropna=False,
        )
    ):

        (
            inventory_source,
            annotation_role,
            taxonomy_category,
            label_key,
        ) = keys

        raw_label_variants = (
            group[
                "raw_label"
            ]
            .value_counts()
        )

        canonical_display_label = (
            raw_label_variants.index[0]
        )

        annotation_count = len(
            group
        )

        unique_content_count = (
            group[
                "content_id"
            ]
            .nunique()
        )

        if "target_id" in group.columns:

            unique_target_count = (
                group[
                    "target_id"
                ]
                .nunique()
            )

        else:

            unique_target_count = None

        evidence_valid_rate = None

        if (
            "evidence_valid"
            in group.columns
        ):

            valid_values = (
                group[
                    "evidence_valid"
                ]
                .dropna()
            )

            if not valid_values.empty:

                if (
                    valid_values.dtype
                    == bool
                ):

                    evidence_valid_rate = round(
                        float(
                            valid_values.mean()
                        ),
                        4,
                    )

                else:

                    normalized_valid = (
                        valid_values
                        .astype(str)
                        .str.lower()
                        .map(
                            {
                                "true": 1,
                                "false": 0,
                                "1": 1,
                                "0": 0,
                            }
                        )
                        .dropna()
                    )

                    if not normalized_valid.empty:

                        evidence_valid_rate = round(
                            float(
                                normalized_valid.mean()
                            ),
                            4,
                        )

        rows.append(
            {
                "inventory_source":
                    inventory_source,

                "annotation_role":
                    annotation_role,

                "taxonomy_category":
                    taxonomy_category,

                "raw_label":
                    canonical_display_label,

                "label_key":
                    label_key,

                "raw_label_variant_count":
                    len(
                        raw_label_variants
                    ),

                "raw_label_variants":
                    " | ".join(
                        raw_label_variants
                        .index
                        .astype(str)
                        .tolist()
                    ),

                "annotation_count":
                    annotation_count,

                "unique_content_count":
                    unique_content_count,

                "unique_target_count":
                    unique_target_count,

                "avg_confidence":
                    safe_mean(
                        group[
                            "label_confidence"
                        ]
                    ),

                "min_confidence":
                    safe_min(
                        group[
                            "label_confidence"
                        ]
                    ),

                "max_confidence":
                    safe_max(
                        group[
                            "label_confidence"
                        ]
                    ),

                "evidence_valid_rate":
                    evidence_valid_rate,

                "is_singleton":
                    annotation_count
                    <= LONG_TAIL_FREQUENCY_THRESHOLD,

                "is_recurring":
                    annotation_count
                    > LONG_TAIL_FREQUENCY_THRESHOLD,

                # Deliverable 4B/4C fields.
                # Intentionally blank for human/normalization review.
                "proposed_normalized_label":
                    "",

                "ontology_status":
                    "",

                "normalization_notes":
                    "",
            }
        )

    inventory_df = pd.DataFrame(
        rows
    )

    inventory_df = (
        inventory_df
        .sort_values(
            [
                "inventory_source",
                "taxonomy_category",
                "annotation_count",
                "raw_label",
            ],
            ascending=[
                True,
                True,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    inventory_df.insert(
        0,
        "inventory_id",
        [
            f"INV_{i:04d}"
            for i
            in range(
                1,
                len(inventory_df)
                + 1
            )
        ],
    )

    return inventory_df


# ============================================================
# EXAMPLE TABLE
# ============================================================

def build_examples(
    annotation_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
) -> pd.DataFrame:

    inventory_lookup = (
        inventory_df[
            [
                "inventory_id",
                "inventory_source",
                "annotation_role",
                "taxonomy_category",
                "label_key",
                "raw_label",
            ]
        ]
        .copy()
    )

    merged = annotation_df.merge(
        inventory_lookup,
        on=[
            "inventory_source",
            "annotation_role",
            "taxonomy_category",
            "label_key",
        ],
        how="left",
        suffixes=(
            "_annotation",
            "_inventory",
        ),
    )

    example_rows = []

    for (
        inventory_id,
        group,
    ) in merged.groupby(
        "inventory_id",
        dropna=False,
    ):

        # Prefer higher-confidence examples.
        group = group.copy()

        group[
            "_confidence_numeric"
        ] = pd.to_numeric(
            group[
                "label_confidence"
            ],
            errors="coerce",
        )

        group = (
            group
            .sort_values(
                "_confidence_numeric",
                ascending=False,
                na_position="last",
            )
            .head(
                MAX_EXAMPLES_PER_LABEL
            )
        )

        for rank, (
            _,
            row,
        ) in enumerate(
            group.iterrows(),
            start=1,
        ):

            example_rows.append(
                {
                    "inventory_id":
                        inventory_id,

                    "example_rank":
                        rank,

                    "inventory_source":
                        row[
                            "inventory_source"
                        ],

                    "annotation_role":
                        row[
                            "annotation_role"
                        ],

                    "taxonomy_category":
                        row[
                            "taxonomy_category"
                        ],

                    "raw_label":
                        row[
                            "raw_label_inventory"
                        ],

                    "content_id":
                        row[
                            "content_id"
                        ],

                    "target_type":
                        row.get(
                            "target_type",
                            "",
                        ),

                    "target_id":
                        row.get(
                            "target_id",
                            "",
                        ),

                    "comment_id":
                        row.get(
                            "comment_id",
                            "",
                        ),

                    "label_confidence":
                        row[
                            "label_confidence"
                        ],

                    "evidence_span":
                        row[
                            "evidence_span"
                        ],

                    "explanation":
                        row.get(
                            "explanation",
                            "",
                        ),

                    "annotation_id":
                        row[
                            "annotation_id"
                        ],
                }
            )

    return pd.DataFrame(
        example_rows
    )


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    annotation_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    total_annotations = len(
        annotation_df
    )

    total_unique_labels = len(
        inventory_df
    )

    singleton_count = int(
        inventory_df[
            "is_singleton"
        ].sum()
    )

    recurring_count = int(
        inventory_df[
            "is_recurring"
        ].sum()
    )

    rows.append(
        {
            "summary_level":
                "overall",

            "inventory_source":
                "ALL",

            "annotation_role":
                "ALL",

            "taxonomy_category":
                "ALL",

            "annotation_count":
                total_annotations,

            "unique_raw_label_count":
                total_unique_labels,

            "singleton_label_count":
                singleton_count,

            "recurring_label_count":
                recurring_count,

            "singleton_rate":
                safe_rate(
                    singleton_count,
                    total_unique_labels,
                ),

            "recurring_rate":
                safe_rate(
                    recurring_count,
                    total_unique_labels,
                ),

            "avg_annotations_per_unique_label":
                round(
                    total_annotations
                    / total_unique_labels,
                    4,
                )
                if total_unique_labels
                else None,

            "avg_confidence":
                safe_mean(
                    annotation_df[
                        "label_confidence"
                    ]
                ),
        }
    )

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    for source, group in (
        annotation_df.groupby(
            "inventory_source"
        )
    ):

        inventory_group = (
            inventory_df[
                inventory_df[
                    "inventory_source"
                ]
                == source
            ]
        )

        unique_count = len(
            inventory_group
        )

        singleton_count = int(
            inventory_group[
                "is_singleton"
            ].sum()
        )

        recurring_count = int(
            inventory_group[
                "is_recurring"
            ].sum()
        )

        rows.append(
            {
                "summary_level":
                    "source",

                "inventory_source":
                    source,

                "annotation_role":
                    "ALL",

                "taxonomy_category":
                    "ALL",

                "annotation_count":
                    len(group),

                "unique_raw_label_count":
                    unique_count,

                "singleton_label_count":
                    singleton_count,

                "recurring_label_count":
                    recurring_count,

                "singleton_rate":
                    safe_rate(
                        singleton_count,
                        unique_count,
                    ),

                "recurring_rate":
                    safe_rate(
                        recurring_count,
                        unique_count,
                    ),

                "avg_annotations_per_unique_label":
                    round(
                        len(group)
                        / unique_count,
                        4,
                    )
                    if unique_count
                    else None,

                "avg_confidence":
                    safe_mean(
                        group[
                            "label_confidence"
                        ]
                    ),
            }
        )

    # --------------------------------------------------------
    # Source + annotation role + category
    # --------------------------------------------------------

    grouped = (
        annotation_df.groupby(
            [
                "inventory_source",
                "annotation_role",
                "taxonomy_category",
            ],
            dropna=False,
        )
    )

    for (
        source,
        role,
        category,
    ), group in grouped:

        inventory_group = (
            inventory_df[
                (
                    inventory_df[
                        "inventory_source"
                    ]
                    == source
                )
                &
                (
                    inventory_df[
                        "annotation_role"
                    ]
                    == role
                )
                &
                (
                    inventory_df[
                        "taxonomy_category"
                    ]
                    == category
                )
            ]
        )

        unique_count = len(
            inventory_group
        )

        singleton_count = int(
            inventory_group[
                "is_singleton"
            ].sum()
        )

        recurring_count = int(
            inventory_group[
                "is_recurring"
            ].sum()
        )

        rows.append(
            {
                "summary_level":
                    "category",

                "inventory_source":
                    source,

                "annotation_role":
                    role,

                "taxonomy_category":
                    category,

                "annotation_count":
                    len(group),

                "unique_raw_label_count":
                    unique_count,

                "singleton_label_count":
                    singleton_count,

                "recurring_label_count":
                    recurring_count,

                "singleton_rate":
                    safe_rate(
                        singleton_count,
                        unique_count,
                    ),

                "recurring_rate":
                    safe_rate(
                        recurring_count,
                        unique_count,
                    ),

                "avg_annotations_per_unique_label":
                    round(
                        len(group)
                        / unique_count,
                        4,
                    )
                    if unique_count
                    else None,

                "avg_confidence":
                    safe_mean(
                        group[
                            "label_confidence"
                        ]
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
    annotation_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    summary_df: pd.DataFrame,
):

    print()
    print(
        "=========================================="
    )
    print(
        "Deliverable 4A — Taxonomy Label Inventory"
    )
    print(
        "=========================================="
    )

    print(
        f"Annotations loaded : "
        f"{len(annotation_df)}"
    )

    print(
        f"Unique raw labels  : "
        f"{len(inventory_df)}"
    )

    singleton_count = int(
        inventory_df[
            "is_singleton"
        ].sum()
    )

    recurring_count = int(
        inventory_df[
            "is_recurring"
        ].sum()
    )

    print(
        f"Singleton labels    : "
        f"{singleton_count}"
    )

    print(
        f"Recurring labels    : "
        f"{recurring_count}"
    )

    print()
    print(
        "By source:"
    )

    source_summary = (
        summary_df[
            summary_df[
                "summary_level"
            ]
            == "source"
        ]
    )

    if not source_summary.empty:

        print(
            source_summary[
                [
                    "inventory_source",
                    "annotation_count",
                    "unique_raw_label_count",
                    "singleton_rate",
                    "avg_annotations_per_unique_label",
                    "avg_confidence",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print()
    print(
        "Top recurring raw labels:"
    )

    top_labels = (
        inventory_df[
            inventory_df[
                "annotation_count"
            ]
            > 1
        ]
        .sort_values(
            "annotation_count",
            ascending=False,
        )
        .head(20)
    )

    if top_labels.empty:

        print(
            "No recurring labels found."
        )

    else:

        print(
            top_labels[
                [
                    "inventory_source",
                    "taxonomy_category",
                    "raw_label",
                    "annotation_count",
                    "unique_content_count",
                    "avg_confidence",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print()
    print(
        "Category-level long-tail summary:"
    )

    category_summary = (
        summary_df[
            summary_df[
                "summary_level"
            ]
            == "category"
        ]
        .sort_values(
            [
                "inventory_source",
                "annotation_count",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    if not category_summary.empty:

        print(
            category_summary[
                [
                    "inventory_source",
                    "annotation_role",
                    "taxonomy_category",
                    "annotation_count",
                    "unique_raw_label_count",
                    "singleton_rate",
                    "avg_annotations_per_unique_label",
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
        "Loading 3A + 3B raw taxonomy annotations..."
    )

    annotation_df = (
        load_all_annotations()
    )

    print(
        f"Loaded "
        f"{len(annotation_df)} annotations."
    )

    # ========================================================
    # BUILD INVENTORY
    # ========================================================

    inventory_df = (
        build_inventory(
            annotation_df
        )
    )

    # ========================================================
    # BUILD EXAMPLES
    # ========================================================

    examples_df = (
        build_examples(
            annotation_df,
            inventory_df,
        )
    )

    # ========================================================
    # BUILD SUMMARY
    # ========================================================

    summary_df = (
        build_summary(
            annotation_df,
            inventory_df,
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    inventory_df.to_csv(
        INVENTORY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    examples_df.to_csv(
        EXAMPLES_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # PRINT
    # ========================================================

    print_diagnostics(
        annotation_df,
        inventory_df,
        summary_df,
    )

    print()
    print(
        "Saved:"
    )

    print(
        f"- {INVENTORY_OUTPUT}"
    )

    print(
        f"- {SUMMARY_OUTPUT}"
    )

    print(
        f"- {EXAMPLES_OUTPUT}"
    )


if __name__ == "__main__":

    main()