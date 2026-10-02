from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIG
# ============================================================

PROPOSAL_CSV = Path(
    "data/processed/taxonomy_normalization_candidates_proposed.csv"
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

CONSOLIDATION_OUTPUT = (
    OUTPUT_DIR
    / "ontology_v02_consolidation_candidates.csv"
)

CONSOLIDATION_MEMBER_OUTPUT = (
    OUTPUT_DIR
    / "ontology_v02_consolidation_members.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "ontology_v02_consolidation_summary.csv"
)


# ============================================================
# SEMANTIC MODEL
# ============================================================

SEMANTIC_MODEL_NAME = (
    "sentence-transformers/"
    "distiluse-base-multilingual-cased-v1"
)


# ============================================================
# CLUSTERING SETTINGS
# ============================================================

# Slightly more conservative than a broad synonym merge.
#
# distance = 1 - cosine_similarity
#
# 0.28 roughly corresponds to ~0.72 similarity,
# although average-linkage clustering does not use
# a simple pairwise threshold rule.
DISTANCE_THRESHOLD = 0.28

HIGH_COHESION_THRESHOLD = 0.78
MEDIUM_COHESION_THRESHOLD = 0.65

MIN_CLUSTER_SIZE_FOR_AUTO_REVIEW_PRIORITY = 2

# High-support concept:
# at least this many underlying annotations.
HIGH_SUPPORT_ANNOTATION_COUNT = 5

# High confidence proposal.
HIGH_PROPOSAL_CONFIDENCE = 0.85


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


def safe_mean(
    series: pd.Series,
) -> float | None:

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return round(
        float(
            values.mean()
        ),
        4,
    )


def weighted_mean(
    values: pd.Series,
    weights: pd.Series,
) -> float | None:

    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    weights = pd.to_numeric(
        weights,
        errors="coerce",
    )

    valid = (
        values.notna()
        & weights.notna()
    )

    values = values[
        valid
    ]

    weights = weights[
        valid
    ]

    if values.empty:
        return None

    if weights.sum() == 0:

        return round(
            float(
                values.mean()
            ),
            4,
        )

    return round(
        float(
            np.average(
                values,
                weights=weights,
            )
        ),
        4,
    )


def classify_cohesion(
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


def make_consolidation_id(
    source: str,
    category: str,
    index: int,
) -> str:

    def sanitize(text: str) -> str:

        text = (
            str(text)
            .lower()
            .strip()
        )

        text = re.sub(
            r"[^a-z0-9]+",
            "_",
            text,
        )

        return text.strip("_")

    return (
        f"ONT_{sanitize(source)}_"
        f"{sanitize(category)}_"
        f"{index:04d}"
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data(
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    if not PROPOSAL_CSV.exists():

        raise FileNotFoundError(
            f"Missing: {PROPOSAL_CSV}"
        )

    if not MEMBER_CSV.exists():

        raise FileNotFoundError(
            f"Missing: {MEMBER_CSV}"
        )

    proposal_df = pd.read_csv(
        PROPOSAL_CSV
    )

    member_df = pd.read_csv(
        MEMBER_CSV
    )

    required_proposal_cols = {
        "candidate_id",
        "inventory_source",
        "taxonomy_category",
        "cluster_size_unique_labels",
        "total_annotation_count",
        "proposal_confidence",
        "ontology_decision_y",
        "proposed_normalized_label_y",
        "proposed_taxonomy_category_y",
        "human_review_needed",
    }

    missing = (
        required_proposal_cols
        - set(
            proposal_df.columns
        )
    )

    if missing:

        raise ValueError(
            "Proposal file missing columns: "
            f"{sorted(missing)}"
        )

    required_member_cols = {
        "candidate_id",
        "inventory_id",
        "raw_label",
        "annotation_count",
    }

    missing = (
        required_member_cols
        - set(
            member_df.columns
        )
    )

    if missing:

        raise ValueError(
            "Member file missing columns: "
            f"{sorted(missing)}"
        )

    proposal_df[
        "candidate_id"
    ] = (
        proposal_df[
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

    proposal_df[
        "human_review_needed"
    ] = (
        proposal_df[
            "human_review_needed"
        ]
        .apply(
            normalize_bool
        )
    )

    return (
        proposal_df,
        member_df,
    )


# ============================================================
# FILTER NEW NODE PROPOSALS
# ============================================================

def get_new_node_proposals(
    proposal_df: pd.DataFrame,
) -> pd.DataFrame:

    new_df = proposal_df[
        proposal_df[
            "ontology_decision_y"
        ]
        == "new_node"
    ].copy()

    new_df[
        "proposal_label"
    ] = (
        new_df[
            "proposed_normalized_label_y"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    new_df[
        "proposal_category"
    ] = (
        new_df[
            "proposed_taxonomy_category_y"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Fallback to original category if proposal omitted it.
    missing_category = (
        new_df[
            "proposal_category"
        ]
        == ""
    )

    new_df.loc[
        missing_category,
        "proposal_category",
    ] = new_df.loc[
        missing_category,
        "taxonomy_category",
    ]

    new_df = new_df[
        new_df[
            "proposal_label"
        ]
        != ""
    ].copy()

    return new_df


# ============================================================
# EMBEDDINGS
# ============================================================

def embed_labels(
    labels: list[str],
) -> np.ndarray:

    embeddings = (
        semantic_model.encode(
            labels,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )

    return np.asarray(
        embeddings,
        dtype=float,
    )


# ============================================================
# CLUSTER PROPOSALS
# ============================================================

def cluster_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:

    if len(
        embeddings
    ) == 1:

        return np.array(
            [0],
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

    return model.fit_predict(
        embeddings
    )


# ============================================================
# SIMILARITY STATS
# ============================================================

def compute_similarity_stats(
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

    upper = np.triu_indices(
        n,
        k=1,
    )

    values = matrix[
        upper
    ]

    return {
        "mean_pairwise_similarity":
            round(
                float(
                    values.mean()
                ),
                4,
            ),

        "min_pairwise_similarity":
            round(
                float(
                    values.min()
                ),
                4,
            ),

        "max_pairwise_similarity":
            round(
                float(
                    values.max()
                ),
                4,
            ),
    }


# ============================================================
# REPRESENTATIVE CONCEPT
# ============================================================

def choose_representative_proposal(
    cluster_df: pd.DataFrame,
    embeddings: np.ndarray,
) -> pd.Series:

    if len(
        cluster_df
    ) == 1:

        return (
            cluster_df.iloc[0]
        )

    similarity_matrix = (
        cosine_similarity(
            embeddings
        )
    )

    semantic_centrality = (
        similarity_matrix.mean(
            axis=1
        )
    )

    proposal_confidence = (
        pd.to_numeric(
            cluster_df[
                "proposal_confidence"
            ],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    annotation_support = (
        pd.to_numeric(
            cluster_df[
                "total_annotation_count"
            ],
            errors="coerce",
        )
        .fillna(1)
        .to_numpy(
            dtype=float
        )
    )

    # Log transform prevents one very large cluster
    # from dominating the representative.
    annotation_support = (
        np.log1p(
            annotation_support
        )
    )

    if (
        annotation_support.max()
        > annotation_support.min()
    ):

        annotation_support = (
            annotation_support
            - annotation_support.min()
        ) / (
            annotation_support.max()
            - annotation_support.min()
        )

    else:

        annotation_support = (
            np.ones_like(
                annotation_support
            )
        )

    score = (
        0.60
        * semantic_centrality
        +
        0.25
        * proposal_confidence
        +
        0.15
        * annotation_support
    )

    best_index = int(
        np.argmax(
            score
        )
    )

    return cluster_df.iloc[
        best_index
    ]


# ============================================================
# SUPPORT FROM RAW MEMBERS
# ============================================================

def get_raw_member_support(
    cluster_candidate_ids: list[str],
    member_df: pd.DataFrame,
) -> dict:

    members = member_df[
        member_df[
            "candidate_id"
        ]
        .isin(
            cluster_candidate_ids
        )
    ].copy()

    if members.empty:

        return {
            "supporting_raw_label_count":
                0,

            "raw_annotation_count":
                0,

            "member_raw_labels":
                "",
        }

    supporting_raw_label_count = (
        members[
            "raw_label"
        ]
        .nunique()
    )

    raw_annotation_count = int(
        pd.to_numeric(
            members[
                "annotation_count"
            ],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    # Most-supported labels first.
    member_display = (
        members
        .sort_values(
            "annotation_count",
            ascending=False,
        )
        .drop_duplicates(
            subset=[
                "raw_label"
            ]
        )
    )

    labels = (
        member_display[
            "raw_label"
        ]
        .astype(str)
        .tolist()
    )

    return {
        "supporting_raw_label_count":
            int(
                supporting_raw_label_count
            ),

        "raw_annotation_count":
            raw_annotation_count,

        "member_raw_labels":
            " | ".join(
                labels[:30]
            ),
    }


# ============================================================
# REVIEW PRIORITY
# ============================================================

def determine_review_priority(
    *,
    proposal_group_size: int,
    raw_annotation_count: int,
    avg_proposal_confidence: float | None,
    mean_similarity: float | None,
    human_review_needed_rate: float,
) -> str:

    # Highest priority:
    # multiple proposal clusters + meaningful support.
    if (
        proposal_group_size >= 2
        and raw_annotation_count
        >= HIGH_SUPPORT_ANNOTATION_COUNT
    ):

        return "high"

    # Existing proposal uncertainty.
    if (
        human_review_needed_rate
        >= 0.5
    ):

        return "high"

    # Low-confidence or weak cohesion.
    if (
        avg_proposal_confidence is not None
        and avg_proposal_confidence
        < 0.75
    ):

        return "medium"

    if (
        mean_similarity is not None
        and mean_similarity
        < MEDIUM_COHESION_THRESHOLD
    ):

        return "medium"

    # Singleton proposals with little support
    # are lower-priority ontology additions.
    if (
        proposal_group_size == 1
        and raw_annotation_count <= 1
    ):

        return "low"

    return "medium"


# ============================================================
# PROCESS SOURCE + CATEGORY
# ============================================================

def process_group(
    group_df: pd.DataFrame,
    member_df: pd.DataFrame,
    *,
    source: str,
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

    proposal_labels = (
        group_df[
            "proposal_label"
        ]
        .tolist()
    )

    embeddings = embed_labels(
        proposal_labels
    )

    cluster_ids = (
        cluster_embeddings(
            embeddings
        )
    )

    group_df[
        "_consolidation_cluster"
    ] = cluster_ids

    candidate_rows = []
    member_rows = []

    unique_clusters = sorted(
        group_df[
            "_consolidation_cluster"
        ]
        .unique()
        .tolist()
    )

    for sequence, cluster_id in enumerate(
        unique_clusters,
        start=1,
    ):

        mask = (
            group_df[
                "_consolidation_cluster"
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

        cluster_embs = (
            embeddings[
                mask.to_numpy()
            ]
        )

        consolidation_id = (
            make_consolidation_id(
                source,
                category,
                sequence,
            )
        )

        stats = (
            compute_similarity_stats(
                cluster_embs
            )
        )

        representative_row = (
            choose_representative_proposal(
                cluster_df,
                cluster_embs,
            )
        )

        candidate_ids = (
            cluster_df[
                "candidate_id"
            ]
            .tolist()
        )

        support = (
            get_raw_member_support(
                candidate_ids,
                member_df,
            )
        )

        proposal_group_size = (
            len(
                cluster_df
            )
        )

        total_candidate_annotations = int(
            pd.to_numeric(
                cluster_df[
                    "total_annotation_count"
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        unique_proposed_labels = (
            cluster_df[
                "proposal_label"
            ]
            .nunique()
        )

        avg_proposal_confidence = (
            weighted_mean(
                cluster_df[
                    "proposal_confidence"
                ],
                cluster_df[
                    "total_annotation_count"
                ],
            )
        )

        human_review_needed_rate = (
            round(
                float(
                    cluster_df[
                        "human_review_needed"
                    ]
                    .mean()
                ),
                4,
            )
        )

        cohesion = classify_cohesion(
            stats[
                "mean_pairwise_similarity"
            ]
        )

        review_priority = (
            determine_review_priority(
                proposal_group_size=(
                    proposal_group_size
                ),
                raw_annotation_count=(
                    support[
                        "raw_annotation_count"
                    ]
                ),
                avg_proposal_confidence=(
                    avg_proposal_confidence
                ),
                mean_similarity=(
                    stats[
                        "mean_pairwise_similarity"
                    ]
                ),
                human_review_needed_rate=(
                    human_review_needed_rate
                ),
            )
        )

        proposed_labels_display = (
            cluster_df[
                "proposal_label"
            ]
            .drop_duplicates()
            .tolist()
        )

        candidate_rows.append(
            {
                "consolidation_id":
                    consolidation_id,

                "inventory_source":
                    source,

                "proposed_taxonomy_category":
                    category,

                "proposal_cluster_count":
                    proposal_group_size,

                "unique_proposed_label_count":
                    unique_proposed_labels,

                "representative_proposed_label":
                    representative_row[
                        "proposal_label"
                    ],

                "member_proposed_labels":
                    " | ".join(
                        proposed_labels_display
                    ),

                "supporting_candidate_ids":
                    " | ".join(
                        candidate_ids
                    ),

                "supporting_raw_label_count":
                    support[
                        "supporting_raw_label_count"
                    ],

                "raw_annotation_count":
                    support[
                        "raw_annotation_count"
                    ],

                "candidate_annotation_count":
                    total_candidate_annotations,

                "avg_proposal_confidence":
                    avg_proposal_confidence,

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

                "consolidation_cohesion":
                    cohesion,

                "human_review_needed_rate":
                    human_review_needed_rate,

                "review_priority":
                    review_priority,

                "is_singleton_consolidation":
                    proposal_group_size
                    == 1,

                "semantic_model":
                    SEMANTIC_MODEL_NAME,

                "distance_threshold":
                    DISTANCE_THRESHOLD,

                # --------------------------------------------
                # HUMAN ONTOLOGY v0.2 REVIEW FIELDS
                # --------------------------------------------

                "human_final_concept":
                    "",

                # approve_new_node
                # merge_existing_node
                # reject
                # defer
                "human_ontology_decision":
                    "",

                "human_final_category":
                    "",

                "human_review_notes":
                    "",

                "human_review_status":
                    "pending",
            }
        )

        # ====================================================
        # MEMBER DETAIL
        # ====================================================

        representative_embedding = (
            semantic_model.encode(
                [
                    representative_row[
                        "proposal_label"
                    ]
                ],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
        )

        similarities = (
            cluster_embs
            @ representative_embedding
        )

        for member_index, (
            _,
            row,
        ) in enumerate(
            cluster_df.iterrows()
        ):

            member_rows.append(
                {
                    "consolidation_id":
                        consolidation_id,

                    "candidate_id":
                        row[
                            "candidate_id"
                        ],

                    "inventory_source":
                        source,

                    "proposed_taxonomy_category":
                        category,

                    "proposed_normalized_label":
                        row[
                            "proposal_label"
                        ],

                    "original_taxonomy_category":
                        row[
                            "taxonomy_category"
                        ],

                    "proposal_confidence":
                        row[
                            "proposal_confidence"
                        ],

                    "proposal_reason":
                        row.get(
                            "proposal_reason",
                            "",
                        ),

                    "human_review_needed":
                        row[
                            "human_review_needed"
                        ],

                    "total_annotation_count":
                        row[
                            "total_annotation_count"
                        ],

                    "cluster_size_unique_labels":
                        row[
                            "cluster_size_unique_labels"
                        ],

                    "similarity_to_representative":
                        round(
                            float(
                                similarities[
                                    member_index
                                ]
                            ),
                            4,
                        ),

                    "is_representative":
                        (
                            row[
                                "candidate_id"
                            ]
                            ==
                            representative_row[
                                "candidate_id"
                            ]
                        ),
                }
            )

    return (
        candidate_rows,
        member_rows,
    )


# ============================================================
# BUILD CONSOLIDATION
# ============================================================

def build_consolidation(
    new_df: pd.DataFrame,
    member_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    all_candidates = []
    all_members = []

    grouped = (
        new_df.groupby(
            [
                "inventory_source",
                "proposal_category",
            ],
            dropna=False,
        )
    )

    total_groups = (
        grouped.ngroups
    )

    print(
        f"[INFO] Source/category groups: "
        f"{total_groups}"
    )

    for index, (
        keys,
        group_df,
    ) in enumerate(
        grouped,
        start=1,
    ):

        (
            source,
            category,
        ) = keys

        print(
            f"[INFO] "
            f"{index}/{total_groups} "
            f"| {source} "
            f"| {category} "
            f"| proposals="
            f"{len(group_df)}"
        )

        (
            candidate_rows,
            member_rows,
        ) = process_group(
            group_df,
            member_df,
            source=source,
            category=category,
        )

        all_candidates.extend(
            candidate_rows
        )

        all_members.extend(
            member_rows
        )

    return (
        pd.DataFrame(
            all_candidates
        ),
        pd.DataFrame(
            all_members
        ),
    )


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    new_df: pd.DataFrame,
    consolidation_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    total_new_proposals = (
        len(
            new_df
        )
    )

    total_consolidated = (
        len(
            consolidation_df
        )
    )

    compression_ratio = (
        total_consolidated
        / total_new_proposals
        if total_new_proposals
        else None
    )

    rows.append(
        {
            "summary_level":
                "overall",

            "inventory_source":
                "ALL",

            "taxonomy_category":
                "ALL",

            "new_node_proposal_count":
                total_new_proposals,

            "consolidated_concept_count":
                total_consolidated,

            "compression_ratio":
                round(
                    compression_ratio,
                    4,
                )
                if compression_ratio
                is not None
                else None,

            "compression_percent":
                round(
                    1
                    - compression_ratio,
                    4,
                )
                if compression_ratio
                is not None
                else None,

            "singleton_concept_count":
                int(
                    consolidation_df[
                        "is_singleton_consolidation"
                    ]
                    .sum()
                ),

            "multi_proposal_concept_count":
                int(
                    (
                        ~consolidation_df[
                            "is_singleton_consolidation"
                        ]
                    ).sum()
                ),
        }
    )

    # --------------------------------------------------------
    # SOURCE / CATEGORY
    # --------------------------------------------------------

    for (
        source,
        category,
    ), group in (
        consolidation_df.groupby(
            [
                "inventory_source",
                "proposed_taxonomy_category",
            ],
            dropna=False,
        )
    ):

        original_count = (
            len(
                new_df[
                    (
                        new_df[
                            "inventory_source"
                        ]
                        == source
                    )
                    &
                    (
                        new_df[
                            "proposal_category"
                        ]
                        == category
                    )
                ]
            )
        )

        consolidated_count = (
            len(
                group
            )
        )

        ratio = (
            consolidated_count
            / original_count
            if original_count
            else None
        )

        rows.append(
            {
                "summary_level":
                    "category",

                "inventory_source":
                    source,

                "taxonomy_category":
                    category,

                "new_node_proposal_count":
                    original_count,

                "consolidated_concept_count":
                    consolidated_count,

                "compression_ratio":
                    round(
                        ratio,
                        4,
                    )
                    if ratio
                    is not None
                    else None,

                "compression_percent":
                    round(
                        1 - ratio,
                        4,
                    )
                    if ratio
                    is not None
                    else None,

                "singleton_concept_count":
                    int(
                        group[
                            "is_singleton_consolidation"
                        ]
                        .sum()
                    ),

                "multi_proposal_concept_count":
                    int(
                        (
                            ~group[
                                "is_singleton_consolidation"
                            ]
                        ).sum()
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
    new_df: pd.DataFrame,
    consolidation_df: pd.DataFrame,
    summary_df: pd.DataFrame,
):

    print()
    print(
        "=========================================="
    )

    print(
        "Deliverable 4C — Ontology Consolidation"
    )

    print(
        "=========================================="
    )

    print(
        f"New-node proposals        : "
        f"{len(new_df)}"
    )

    print(
        f"Consolidated concepts     : "
        f"{len(consolidation_df)}"
    )

    if len(
        new_df
    ) > 0:

        ratio = (
            len(consolidation_df)
            / len(new_df)
        )

        print(
            f"Compression ratio         : "
            f"{ratio:.3f}"
        )

        print(
            f"Concept reduction         : "
            f"{1 - ratio:.1%}"
        )

    print(
        f"Singleton concepts        : "
        f"{int(consolidation_df['is_singleton_consolidation'].sum())}"
    )

    print(
        f"Multi-proposal concepts   : "
        f"{int((~consolidation_df['is_singleton_consolidation']).sum())}"
    )

    print()
    print(
        "Review priority:"
    )

    print(
        consolidation_df[
            "review_priority"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Largest consolidated concept groups:"
    )

    largest = (
        consolidation_df
        .sort_values(
            [
                "proposal_cluster_count",
                "raw_annotation_count",
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

    print(
        largest[
            [
                "inventory_source",
                "proposed_taxonomy_category",
                "proposal_cluster_count",
                "raw_annotation_count",
                "representative_proposed_label",
                "mean_pairwise_similarity",
                "consolidation_cohesion",
                "review_priority",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "High-priority ontology review candidates:"
    )

    high_priority = (
        consolidation_df[
            consolidation_df[
                "review_priority"
            ]
            == "high"
        ]
        .sort_values(
            [
                "raw_annotation_count",
                "proposal_cluster_count",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(
            30
        )
    )

    if high_priority.empty:

        print(
            "No high-priority candidates."
        )

    else:

        print(
            high_priority[
                [
                    "inventory_source",
                    "proposed_taxonomy_category",
                    "representative_proposed_label",
                    "proposal_cluster_count",
                    "supporting_raw_label_count",
                    "raw_annotation_count",
                    "avg_proposal_confidence",
                    "consolidation_cohesion",
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
    )

    print(
        category_summary.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading ontology proposals..."
    )

    (
        proposal_df,
        member_df,
    ) = load_data()

    new_df = (
        get_new_node_proposals(
            proposal_df
        )
    )

    print(
        f"[INFO] Total proposals    : "
        f"{len(proposal_df)}"
    )

    print(
        f"[INFO] New-node proposals : "
        f"{len(new_df)}"
    )

    # ========================================================
    # CONSOLIDATE
    # ========================================================

    (
        consolidation_df,
        consolidation_member_df,
    ) = build_consolidation(
        new_df,
        member_df,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_df = (
        build_summary(
            new_df,
            consolidation_df,
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    consolidation_df.to_csv(
        CONSOLIDATION_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    consolidation_member_df.to_csv(
        CONSOLIDATION_MEMBER_OUTPUT,
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
        new_df,
        consolidation_df,
        summary_df,
    )

    print()
    print(
        "Saved:"
    )

    print(
        f"- {CONSOLIDATION_OUTPUT}"
    )

    print(
        f"- {CONSOLIDATION_MEMBER_OUTPUT}"
    )

    print(
        f"- {SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":

    main()