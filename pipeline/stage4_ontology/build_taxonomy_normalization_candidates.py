from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = Path(
    "data/processed/taxonomy_label_inventory.csv"
)

OUTPUT_DIR = Path(
    "data/processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CANDIDATE_OUTPUT = (
    OUTPUT_DIR
    / "taxonomy_normalization_candidates.csv"
)

MEMBER_OUTPUT = (
    OUTPUT_DIR
    / "taxonomy_normalization_candidate_members.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "taxonomy_normalization_candidate_summary.csv"
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

SEMANTIC_MODEL_NAME = (
    "sentence-transformers/"
    "distiluse-base-multilingual-cased-v1"
)


# ============================================================
# CLUSTERING CONFIG
# ============================================================

# Agglomerative clustering uses distance:
#
# cosine_distance = 1 - cosine_similarity
#
# Example:
# similarity 0.75
# → distance 0.25
#
# Labels farther apart than this threshold
# will generally not merge.
#
# This is intentionally conservative.
DISTANCE_THRESHOLD = 0.32

# Equivalent rough similarity:
# 1 - 0.32 = 0.68
#
# IMPORTANT:
# This is only candidate discovery.
# It is NOT our final ontology-match threshold.

MIN_LABELS_FOR_CLUSTERING = 2

# Helpful diagnostic thresholds.
HIGH_COHESION_THRESHOLD = 0.75
MEDIUM_COHESION_THRESHOLD = 0.60

# Avoid printing overly large member strings.
MAX_MEMBER_LABELS_IN_SUMMARY = 30


# ============================================================
# LOAD MODEL
# ============================================================

print(
    f"[INFO] Loading semantic model: "
    f"{SEMANTIC_MODEL_NAME}"
)

semantic_model = SentenceTransformer(
    SEMANTIC_MODEL_NAME
)


# ============================================================
# UTILS
# ============================================================

def clean_text(value) -> str:

    if pd.isna(value):
        return ""

    return str(value).strip()


def safe_float(value) -> float | None:

    try:

        if pd.isna(value):
            return None

        return float(value)

    except Exception:
        return None


def weighted_mean(
    values: list[float],
    weights: list[float],
) -> float | None:

    if not values:
        return None

    values_array = np.asarray(
        values,
        dtype=float,
    )

    weights_array = np.asarray(
        weights,
        dtype=float,
    )

    valid = (
        ~np.isnan(values_array)
        & ~np.isnan(weights_array)
    )

    values_array = (
        values_array[valid]
    )

    weights_array = (
        weights_array[valid]
    )

    if len(values_array) == 0:
        return None

    if weights_array.sum() == 0:

        return round(
            float(
                values_array.mean()
            ),
            4,
        )

    return round(
        float(
            np.average(
                values_array,
                weights=weights_array,
            )
        ),
        4,
    )


def classify_cluster_cohesion(
    similarity: float | None,
) -> str:

    if similarity is None:
        return "singleton"

    if (
        similarity
        >= HIGH_COHESION_THRESHOLD
    ):
        return "high"

    if (
        similarity
        >= MEDIUM_COHESION_THRESHOLD
    ):
        return "medium"

    return "low"


def make_candidate_id(
    source: str,
    role: str,
    category: str,
    index: int,
) -> str:

    def sanitize(text: str) -> str:

        return (
            text
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
        )

    return (
        "NC_"
        f"{sanitize(source)}_"
        f"{sanitize(role)}_"
        f"{sanitize(category)}_"
        f"{index:04d}"
    )


# ============================================================
# LOAD INVENTORY
# ============================================================

def load_inventory() -> pd.DataFrame:

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Missing input file: "
            f"{INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV
    )

    required_columns = {
        "inventory_id",
        "inventory_source",
        "annotation_role",
        "taxonomy_category",
        "raw_label",
        "annotation_count",
        "avg_confidence",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "taxonomy_label_inventory.csv "
            "is missing required columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()

    text_columns = [
        "inventory_id",
        "inventory_source",
        "annotation_role",
        "taxonomy_category",
        "raw_label",
    ]

    for column in text_columns:

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    df[
        "annotation_count"
    ] = (
        pd.to_numeric(
            df[
                "annotation_count"
            ],
            errors="coerce",
        )
        .fillna(1)
        .astype(int)
    )

    df[
        "avg_confidence"
    ] = pd.to_numeric(
        df[
            "avg_confidence"
        ],
        errors="coerce",
    )

    df = df[
        df[
            "raw_label"
        ]
        != ""
    ].copy()

    return df


# ============================================================
# EMBEDDING
# ============================================================

def embed_labels(
    labels: list[str],
) -> np.ndarray:

    embeddings = semantic_model.encode(
        labels,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(
        embeddings,
        dtype=float,
    )


# ============================================================
# CLUSTERING
# ============================================================

def cluster_labels(
    embeddings: np.ndarray,
) -> np.ndarray:

    n = len(
        embeddings
    )

    if (
        n
        < MIN_LABELS_FOR_CLUSTERING
    ):

        return np.zeros(
            n,
            dtype=int,
        )

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=(
            DISTANCE_THRESHOLD
        ),
    )

    labels = model.fit_predict(
        embeddings
    )

    return labels


# ============================================================
# CLUSTER DIAGNOSTICS
# ============================================================

def compute_cluster_similarity_stats(
    embeddings: np.ndarray,
) -> dict:

    n = len(
        embeddings
    )

    if n <= 1:

        return {
            "mean_pairwise_similarity":
                None,

            "min_pairwise_similarity":
                None,

            "max_pairwise_similarity":
                None,
        }

    matrix = cosine_similarity(
        embeddings
    )

    # Upper triangle without diagonal.
    upper_indices = np.triu_indices(
        n,
        k=1,
    )

    values = matrix[
        upper_indices
    ]

    if len(values) == 0:

        return {
            "mean_pairwise_similarity":
                None,

            "min_pairwise_similarity":
                None,

            "max_pairwise_similarity":
                None,
        }

    return {
        "mean_pairwise_similarity":
            round(
                float(
                    np.mean(values)
                ),
                4,
            ),

        "min_pairwise_similarity":
            round(
                float(
                    np.min(values)
                ),
                4,
            ),

        "max_pairwise_similarity":
            round(
                float(
                    np.max(values)
                ),
                4,
            ),
    }


def select_representative_label(
    cluster_df: pd.DataFrame,
    cluster_embeddings: np.ndarray,
) -> tuple[str, str]:

    """
    Pick a representative raw label.

    We combine:
    - semantic centrality
    - frequency

    This is only a review aid.
    It is NOT automatically the final normalized label.
    """

    if len(
        cluster_df
    ) == 1:

        row = (
            cluster_df.iloc[0]
        )

        return (
            row[
                "inventory_id"
            ],
            row[
                "raw_label"
            ],
        )

    similarity_matrix = (
        cosine_similarity(
            cluster_embeddings
        )
    )

    centrality = (
        similarity_matrix.mean(
            axis=1
        )
    )

    frequency = (
        cluster_df[
            "annotation_count"
        ]
        .to_numpy(
            dtype=float
        )
    )

    # Normalize frequency so it doesn't dominate.
    if (
        frequency.max()
        > frequency.min()
    ):

        frequency_norm = (
            frequency
            - frequency.min()
        ) / (
            frequency.max()
            - frequency.min()
        )

    else:

        frequency_norm = (
            np.ones_like(
                frequency
            )
        )

    score = (
        0.75
        * centrality
        +
        0.25
        * frequency_norm
    )

    best_index = int(
        np.argmax(
            score
        )
    )

    row = (
        cluster_df.iloc[
            best_index
        ]
    )

    return (
        row[
            "inventory_id"
        ],
        row[
            "raw_label"
        ],
    )


# ============================================================
# PROCESS CATEGORY
# ============================================================

def process_category_group(
    group_df: pd.DataFrame,
    *,
    source: str,
    annotation_role: str,
    category: str,
) -> tuple[
    list[dict],
    list[dict],
]:

    group_df = (
        group_df
        .copy()
        .reset_index(
            drop=True
        )
    )

    raw_labels = (
        group_df[
            "raw_label"
        ]
        .tolist()
    )

    embeddings = (
        embed_labels(
            raw_labels
        )
    )

    cluster_ids = (
        cluster_labels(
            embeddings
        )
    )

    group_df[
        "_cluster_id"
    ] = cluster_ids

    candidate_rows = []
    member_rows = []

    cluster_order = (
        sorted(
            group_df[
                "_cluster_id"
            ]
            .unique()
            .tolist()
        )
    )

    for cluster_number, cluster_id in enumerate(
        cluster_order,
        start=1,
    ):

        mask = (
            group_df[
                "_cluster_id"
            ]
            == cluster_id
        )

        cluster_df = (
            group_df[
                mask
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        cluster_embeddings = (
            embeddings[
                mask.to_numpy()
            ]
        )

        candidate_id = (
            make_candidate_id(
                source,
                annotation_role,
                category,
                cluster_number,
            )
        )

        stats = (
            compute_cluster_similarity_stats(
                cluster_embeddings
            )
        )

        (
            representative_inventory_id,
            representative_label,
        ) = select_representative_label(
            cluster_df,
            cluster_embeddings,
        )

        total_annotations = int(
            cluster_df[
                "annotation_count"
            ]
            .sum()
        )

        unique_label_count = (
            len(
                cluster_df
            )
        )

        avg_confidence = (
            weighted_mean(
                cluster_df[
                    "avg_confidence"
                ]
                .fillna(
                    np.nan
                )
                .tolist(),

                cluster_df[
                    "annotation_count"
                ]
                .tolist(),
            )
        )

        member_labels = (
            cluster_df[
                "raw_label"
            ]
            .tolist()
        )

        member_labels_display = (
            member_labels[
                :MAX_MEMBER_LABELS_IN_SUMMARY
            ]
        )

        if (
            len(member_labels)
            > MAX_MEMBER_LABELS_IN_SUMMARY
        ):

            member_labels_display.append(
                f"... +"
                f"{len(member_labels) - MAX_MEMBER_LABELS_IN_SUMMARY}"
                f" more"
            )

        cohesion = (
            classify_cluster_cohesion(
                stats[
                    "mean_pairwise_similarity"
                ]
            )
        )

        candidate_rows.append(
            {
                "candidate_id":
                    candidate_id,

                "inventory_source":
                    source,

                "annotation_role":
                    annotation_role,

                "taxonomy_category":
                    category,

                "cluster_size_unique_labels":
                    unique_label_count,

                "total_annotation_count":
                    total_annotations,

                "representative_inventory_id":
                    representative_inventory_id,

                "representative_raw_label":
                    representative_label,

                "member_raw_labels":
                    " | ".join(
                        member_labels_display
                    ),

                "avg_confidence":
                    avg_confidence,

                "mean_pairwise_similarity":
                    stats[
                        "mean_pairwise_similarity"
                    ],

                "min_pairwise_similarity":
                    stats[
                        "min_pairwise_similarity"
                    ],

                "max_pairwise_similarity":
                    stats[
                        "max_pairwise_similarity"
                    ],

                "cluster_cohesion":
                    cohesion,

                "is_singleton_cluster":
                    unique_label_count
                    == 1,

                "semantic_model":
                    SEMANTIC_MODEL_NAME,

                "distance_threshold":
                    DISTANCE_THRESHOLD,

                # --------------------------------------------
                # HUMAN / LLM-ASSISTED REVIEW FIELDS
                # --------------------------------------------

                "proposed_normalized_label":
                    "",

                # existing_node
                # new_node
                # ambiguous
                # reject
                "ontology_decision":
                    "",

                "proposed_taxonomy_category":
                    "",

                "reviewer_notes":
                    "",

                "human_review_status":
                    "pending",
            }
        )

        # ----------------------------------------------------
        # MEMBER TABLE
        # ----------------------------------------------------

        representative_embedding = (
            cluster_embeddings[
                cluster_df.index[
                    cluster_df[
                        "inventory_id"
                    ]
                    == representative_inventory_id
                ][0]
            ]
            if (
                representative_inventory_id
                in cluster_df[
                    "inventory_id"
                ].values
            )
            else None
        )

        if (
            representative_embedding
            is not None
        ):

            representative_similarities = (
                cosine_similarity(
                    cluster_embeddings,
                    representative_embedding.reshape(
                        1,
                        -1,
                    ),
                )
                .reshape(
                    -1
                )
            )

        else:

            representative_similarities = (
                np.full(
                    len(cluster_df),
                    np.nan,
                )
            )

        for member_index, (
            _,
            member_row,
        ) in enumerate(
            cluster_df.iterrows()
        ):

            member_rows.append(
                {
                    "candidate_id":
                        candidate_id,

                    "inventory_id":
                        member_row[
                            "inventory_id"
                        ],

                    "inventory_source":
                        source,

                    "annotation_role":
                        annotation_role,

                    "taxonomy_category":
                        category,

                    "raw_label":
                        member_row[
                            "raw_label"
                        ],

                    "annotation_count":
                        member_row[
                            "annotation_count"
                        ],

                    "unique_content_count":
                        member_row.get(
                            "unique_content_count"
                        ),

                    "avg_confidence":
                        member_row[
                            "avg_confidence"
                        ],

                    "similarity_to_representative":
                        round(
                            float(
                                representative_similarities[
                                    member_index
                                ]
                            ),
                            4,
                        )
                        if not np.isnan(
                            representative_similarities[
                                member_index
                            ]
                        )
                        else None,

                    "is_representative":
                        (
                            member_row[
                                "inventory_id"
                            ]
                            == representative_inventory_id
                        ),

                    "proposed_normalized_label":
                        "",

                    "normalization_status":
                        "",

                    "member_review_notes":
                        "",
                }
            )

    return (
        candidate_rows,
        member_rows,
    )


# ============================================================
# BUILD ALL CANDIDATES
# ============================================================

def build_candidates(
    inventory_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    all_candidates = []
    all_members = []

    grouped = (
        inventory_df.groupby(
            [
                "inventory_source",
                "annotation_role",
                "taxonomy_category",
            ],
            dropna=False,
        )
    )

    total_groups = (
        grouped.ngroups
    )

    print(
        f"[INFO] Category groups: "
        f"{total_groups}"
    )

    for group_index, (
        keys,
        group_df,
    ) in enumerate(
        grouped,
        start=1,
    ):

        (
            source,
            annotation_role,
            category,
        ) = keys

        print(
            f"[INFO] "
            f"{group_index}/{total_groups} "
            f"{source} | "
            f"{annotation_role} | "
            f"{category} | "
            f"{len(group_df)} labels"
        )

        (
            candidate_rows,
            member_rows,
        ) = process_category_group(
            group_df,
            source=source,
            annotation_role=(
                annotation_role
            ),
            category=category,
        )

        all_candidates.extend(
            candidate_rows
        )

        all_members.extend(
            member_rows
        )

    candidate_df = pd.DataFrame(
        all_candidates
    )

    member_df = pd.DataFrame(
        all_members
    )

    return (
        candidate_df,
        member_df,
    )


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    candidate_df: pd.DataFrame,
    member_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    total_candidates = (
        len(
            candidate_df
        )
    )

    total_labels = (
        len(
            member_df
        )
    )

    singleton_clusters = int(
        candidate_df[
            "is_singleton_cluster"
        ]
        .sum()
    )

    multi_label_clusters = (
        total_candidates
        - singleton_clusters
    )

    labels_in_multi_clusters = int(
        candidate_df.loc[
            ~candidate_df[
                "is_singleton_cluster"
            ],
            "cluster_size_unique_labels",
        ]
        .sum()
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

            "unique_raw_label_count":
                total_labels,

            "candidate_cluster_count":
                total_candidates,

            "singleton_cluster_count":
                singleton_clusters,

            "multi_label_cluster_count":
                multi_label_clusters,

            "labels_in_multi_label_clusters":
                labels_in_multi_clusters,

            "cluster_compression_ratio":
                round(
                    (
                        total_candidates
                        / total_labels
                    ),
                    4,
                )
                if total_labels
                else None,

            "multi_cluster_label_coverage":
                round(
                    (
                        labels_in_multi_clusters
                        / total_labels
                    ),
                    4,
                )
                if total_labels
                else None,
        }
    )

    # --------------------------------------------------------
    # Source/category
    # --------------------------------------------------------

    grouped = (
        candidate_df.groupby(
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

        corresponding_members = (
            member_df[
                member_df[
                    "candidate_id"
                ].isin(
                    group[
                        "candidate_id"
                    ]
                )
            ]
        )

        raw_label_count = (
            len(
                corresponding_members
            )
        )

        candidate_count = (
            len(
                group
            )
        )

        singleton_count = int(
            group[
                "is_singleton_cluster"
            ]
            .sum()
        )

        multi_cluster_count = (
            candidate_count
            - singleton_count
        )

        labels_in_multi = int(
            group.loc[
                ~group[
                    "is_singleton_cluster"
                ],
                "cluster_size_unique_labels",
            ]
            .sum()
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

                "unique_raw_label_count":
                    raw_label_count,

                "candidate_cluster_count":
                    candidate_count,

                "singleton_cluster_count":
                    singleton_count,

                "multi_label_cluster_count":
                    multi_cluster_count,

                "labels_in_multi_label_clusters":
                    labels_in_multi,

                "cluster_compression_ratio":
                    round(
                        candidate_count
                        / raw_label_count,
                        4,
                    )
                    if raw_label_count
                    else None,

                "multi_cluster_label_coverage":
                    round(
                        labels_in_multi
                        / raw_label_count,
                        4,
                    )
                    if raw_label_count
                    else None,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

def print_diagnostics(
    candidate_df: pd.DataFrame,
    member_df: pd.DataFrame,
    summary_df: pd.DataFrame,
):

    print()
    print(
        "=========================================="
    )
    print(
        "Deliverable 4B — Normalization Candidates"
    )
    print(
        "=========================================="
    )

    print(
        f"Raw labels          : "
        f"{len(member_df)}"
    )

    print(
        f"Candidate clusters  : "
        f"{len(candidate_df)}"
    )

    singleton_count = int(
        candidate_df[
            "is_singleton_cluster"
        ]
        .sum()
    )

    multi_count = (
        len(candidate_df)
        - singleton_count
    )

    print(
        f"Singleton clusters  : "
        f"{singleton_count}"
    )

    print(
        f"Multi-label clusters: "
        f"{multi_count}"
    )

    if len(member_df) > 0:

        compression = (
            len(candidate_df)
            / len(member_df)
        )

        print(
            f"Compression ratio   : "
            f"{compression:.3f}"
        )

    print()
    print(
        "Largest candidate clusters:"
    )

    largest = (
        candidate_df[
            ~candidate_df[
                "is_singleton_cluster"
            ]
        ]
        .sort_values(
            [
                "cluster_size_unique_labels",
                "total_annotation_count",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(
            20
        )
    )

    if largest.empty:

        print(
            "No multi-label clusters found."
        )

    else:

        print(
            largest[
                [
                    "inventory_source",
                    "taxonomy_category",
                    "cluster_size_unique_labels",
                    "total_annotation_count",
                    "representative_raw_label",
                    "mean_pairwise_similarity",
                    "cluster_cohesion",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print()
    print(
        "Category summary:"
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
                "unique_raw_label_count",
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
                    "unique_raw_label_count",
                    "candidate_cluster_count",
                    "singleton_cluster_count",
                    "cluster_compression_ratio",
                    "multi_cluster_label_coverage",
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
        "Loading taxonomy label inventory..."
    )

    inventory_df = (
        load_inventory()
    )

    print(
        f"[INFO] Loaded "
        f"{len(inventory_df)} raw labels."
    )

    # ========================================================
    # BUILD
    # ========================================================

    (
        candidate_df,
        member_df,
    ) = build_candidates(
        inventory_df
    )

    summary_df = (
        build_summary(
            candidate_df,
            member_df,
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    candidate_df.to_csv(
        CANDIDATE_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    member_df.to_csv(
        MEMBER_OUTPUT,
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
        candidate_df,
        member_df,
        summary_df,
    )

    print()
    print(
        "Saved:"
    )

    print(
        f"- {CANDIDATE_OUTPUT}"
    )

    print(
        f"- {MEMBER_OUTPUT}"
    )

    print(
        f"- {SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":

    main()