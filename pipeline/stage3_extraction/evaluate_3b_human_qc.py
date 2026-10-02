from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = Path(
    "data/evaluation/audience_human_qc_reviewed.csv"
)

OUTPUT_DIR = Path(
    "data/evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METRICS_OUTPUT = (
    OUTPUT_DIR
    / "audience_human_qc_metrics.csv"
)

CONFUSION_OUTPUT = (
    OUTPUT_DIR
    / "audience_signal_confusion_matrix.csv"
)

ERROR_OUTPUT = (
    OUTPUT_DIR
    / "audience_human_qc_error_analysis.csv"
)

GROUP_OUTPUT = (
    OUTPUT_DIR
    / "audience_human_qc_group_analysis.csv"
)

EVALUATED_OUTPUT = (
    OUTPUT_DIR
    / "audience_human_qc_evaluated.csv"
)


# ============================================================
# CANONICALIZATION MAPS
# ============================================================

JUDGMENT_MAP = {
    "correct": "correct",
    "yes": "yes",

    "partially_correct": "partial",
    "partially correct": "partial",
    "partial": "partial",

    "incorrect": "incorrect",
    "no": "no",

    "not_applicable": "not_applicable",
    "not applicable": "not_applicable",
    "n/a": "not_applicable",
    "na": "not_applicable",
}


ERROR_TYPE_MAP = {
    "none": "none",

    "over_interpretation":
        "over_interpretation",

    "over interpretation":
        "over_interpretation",

    # typo currently present in reviewed QC
    "under_intepretation":
        "under_interpretation",

    "under_interpretation":
        "under_interpretation",

    "under interpretation":
        "under_interpretation",

    "false_positive":
        "false_positive",

    "false_negative":
        "false_negative",

    "wrong_category":
        "wrong_category",

    "redundant_label":
        "redundant_label",

    "ambiguous":
        "ambiguous",

    "other":
        "other",
}


# ============================================================
# HELPERS
# ============================================================

def clean_value(value) -> str:
    """
    Normalize basic human-entered string values.
    """

    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .lower()
    )


def canonicalize_judgment(value) -> str:
    """
    Canonicalize judgment labels.
    """

    cleaned = clean_value(value)

    if not cleaned:
        return ""

    return JUDGMENT_MAP.get(
        cleaned,
        cleaned,
    )


def canonicalize_error_type(value) -> str:
    """
    Canonicalize human_error_type values.
    """

    cleaned = clean_value(value)

    if not cleaned:
        return "none"

    return ERROR_TYPE_MAP.get(
        cleaned,
        cleaned,
    )


def safe_divide(
    numerator: float,
    denominator: float,
) -> float | None:

    if denominator == 0:
        return None

    return numerator / denominator


def round_metric(
    value: float | None,
    decimals: int = 4,
):

    if value is None:
        return None

    return round(
        float(value),
        decimals,
    )


# ============================================================
# LOAD + VALIDATE
# ============================================================

def load_data() -> pd.DataFrame:

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Missing input file: {INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV
    )

    required_columns = {
        "qc_id",
        "qc_sample_group",
        "content_id",
        "comment_id",
        "comment_text",
        "ai_label_count",
        "human_overall_judgment",
        "human_has_signal",
        "human_category_judgment",
        "human_label_judgment",
        "human_error_type",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Reviewed QC file is missing "
            f"required columns: {sorted(missing)}"
        )

    if df["qc_id"].duplicated().any():

        raise ValueError(
            "Duplicate qc_id detected."
        )

    return df


# ============================================================
# PREPARE EVALUATION FIELDS
# ============================================================

def prepare_evaluation_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    # --------------------------------------------------------
    # Canonicalize human review fields
    # --------------------------------------------------------

    result[
        "human_overall_judgment_canonical"
    ] = (
        result[
            "human_overall_judgment"
        ]
        .apply(
            canonicalize_judgment
        )
    )

    result[
        "human_category_judgment_canonical"
    ] = (
        result[
            "human_category_judgment"
        ]
        .apply(
            canonicalize_judgment
        )
    )

    result[
        "human_label_judgment_canonical"
    ] = (
        result[
            "human_label_judgment"
        ]
        .apply(
            canonicalize_judgment
        )
    )

    result[
        "human_error_type_canonical"
    ] = (
        result[
            "human_error_type"
        ]
        .apply(
            canonicalize_error_type
        )
    )

    result[
        "human_has_signal_canonical"
    ] = (
        result[
            "human_has_signal"
        ]
        .apply(
            clean_value
        )
    )

    # --------------------------------------------------------
    # AI signal prediction
    # --------------------------------------------------------

    result[
        "ai_has_signal"
    ] = (
        pd.to_numeric(
            result[
                "ai_label_count"
            ],
            errors="coerce",
        )
        .fillna(0)
        > 0
    )

    result[
        "human_has_signal_bool"
    ] = (
        result[
            "human_has_signal_canonical"
        ]
        == "yes"
    )

    # --------------------------------------------------------
    # Signal detection classification
    # --------------------------------------------------------

    conditions = [
        (
            result["ai_has_signal"]
            & result[
                "human_has_signal_bool"
            ]
        ),

        (
            result["ai_has_signal"]
            & ~result[
                "human_has_signal_bool"
            ]
        ),

        (
            ~result["ai_has_signal"]
            & result[
                "human_has_signal_bool"
            ]
        ),

        (
            ~result["ai_has_signal"]
            & ~result[
                "human_has_signal_bool"
            ]
        ),
    ]

    choices = [
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
    ]

    result[
        "signal_detection_result"
    ] = np.select(
        conditions,
        choices,
        default="unknown",
    )

    # --------------------------------------------------------
    # Overall interpretation scoring
    #
    # correct = 1
    # partial = 0.5
    # incorrect = 0
    # --------------------------------------------------------

    overall_score_map = {
        "correct": 1.0,
        "partial": 0.5,
        "incorrect": 0.0,
    }

    result[
        "overall_quality_score"
    ] = (
        result[
            "human_overall_judgment_canonical"
        ]
        .map(
            overall_score_map
        )
    )

    # --------------------------------------------------------
    # Category / Label numeric scoring
    #
    # yes     = 1
    # partial = 0.5
    # no      = 0
    #
    # N/A stays missing and does not enter accuracy denominator.
    # --------------------------------------------------------

    interpretation_score_map = {
        "yes": 1.0,
        "partial": 0.5,
        "no": 0.0,
    }

    result[
        "category_quality_score"
    ] = (
        result[
            "human_category_judgment_canonical"
        ]
        .map(
            interpretation_score_map
        )
    )

    result[
        "label_quality_score"
    ] = (
        result[
            "human_label_judgment_canonical"
        ]
        .map(
            interpretation_score_map
        )
    )

    return result


# ============================================================
# SIGNAL DETECTION METRICS
# ============================================================

def calculate_signal_metrics(
    df: pd.DataFrame,
) -> dict:

    counts = (
        df[
            "signal_detection_result"
        ]
        .value_counts()
        .to_dict()
    )

    tp = int(
        counts.get(
            "true_positive",
            0,
        )
    )

    fp = int(
        counts.get(
            "false_positive",
            0,
        )
    )

    fn = int(
        counts.get(
            "false_negative",
            0,
        )
    )

    tn = int(
        counts.get(
            "true_negative",
            0,
        )
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    if (
        precision is not None
        and recall is not None
        and precision + recall > 0
    ):

        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    else:
        f1 = None

    accuracy = safe_divide(
        tp + tn,
        tp + fp + fn + tn,
    )

    specificity = safe_divide(
        tn,
        tn + fp,
    )

    return {
        "true_positive":
            tp,

        "false_positive":
            fp,

        "false_negative":
            fn,

        "true_negative":
            tn,

        "signal_precision":
            round_metric(
                precision
            ),

        "signal_recall":
            round_metric(
                recall
            ),

        "signal_f1":
            round_metric(
                f1
            ),

        "signal_accuracy":
            round_metric(
                accuracy
            ),

        "signal_specificity":
            round_metric(
                specificity
            ),
    }


# ============================================================
# INTERPRETATION QUALITY
# ============================================================

def calculate_interpretation_metrics(
    df: pd.DataFrame,
) -> dict:

    total = len(df)

    overall_counts = (
        df[
            "human_overall_judgment_canonical"
        ]
        .value_counts()
        .to_dict()
    )

    overall_correct = int(
        overall_counts.get(
            "correct",
            0,
        )
    )

    overall_partial = int(
        overall_counts.get(
            "partial",
            0,
        )
    )

    overall_incorrect = int(
        overall_counts.get(
            "incorrect",
            0,
        )
    )

    # --------------------------------------------------------
    # Exact overall accuracy
    # --------------------------------------------------------

    overall_exact_accuracy = (
        safe_divide(
            overall_correct,
            total,
        )
    )

    # --------------------------------------------------------
    # Acceptable rate:
    # correct + partially correct
    # --------------------------------------------------------

    acceptable_rate = (
        safe_divide(
            overall_correct
            + overall_partial,
            total,
        )
    )

    # --------------------------------------------------------
    # Weighted quality:
    # correct = 1
    # partial = .5
    # incorrect = 0
    # --------------------------------------------------------

    weighted_overall_quality = (
        df[
            "overall_quality_score"
        ]
        .dropna()
        .mean()
    )

    # --------------------------------------------------------
    # Category evaluation
    #
    # Only rows where category judgment exists.
    # --------------------------------------------------------

    category_eval = df[
        df[
            "category_quality_score"
        ]
        .notna()
    ]

    category_exact = (
        (
            category_eval[
                "human_category_judgment_canonical"
            ]
            == "yes"
        )
        .mean()
        if not category_eval.empty
        else None
    )

    category_acceptable = (
        (
            category_eval[
                "human_category_judgment_canonical"
            ]
            .isin(
                [
                    "yes",
                    "partial",
                ]
            )
        )
        .mean()
        if not category_eval.empty
        else None
    )

    category_weighted = (
        category_eval[
            "category_quality_score"
        ]
        .mean()
        if not category_eval.empty
        else None
    )

    # --------------------------------------------------------
    # Label evaluation
    # --------------------------------------------------------

    label_eval = df[
        df[
            "label_quality_score"
        ]
        .notna()
    ]

    label_exact = (
        (
            label_eval[
                "human_label_judgment_canonical"
            ]
            == "yes"
        )
        .mean()
        if not label_eval.empty
        else None
    )

    label_acceptable = (
        (
            label_eval[
                "human_label_judgment_canonical"
            ]
            .isin(
                [
                    "yes",
                    "partial",
                ]
            )
        )
        .mean()
        if not label_eval.empty
        else None
    )

    label_weighted = (
        label_eval[
            "label_quality_score"
        ]
        .mean()
        if not label_eval.empty
        else None
    )

    # --------------------------------------------------------
    # True-positive interpretation quality
    #
    # Important:
    # separates "did AI detect a signal?"
    # from "once signal was detected, did it interpret correctly?"
    # --------------------------------------------------------

    tp_df = df[
        df[
            "signal_detection_result"
        ]
        == "true_positive"
    ]

    tp_overall_correct = (
        (
            tp_df[
                "human_overall_judgment_canonical"
            ]
            == "correct"
        )
        .mean()
        if not tp_df.empty
        else None
    )

    tp_overall_acceptable = (
        (
            tp_df[
                "human_overall_judgment_canonical"
            ]
            .isin(
                [
                    "correct",
                    "partial",
                ]
            )
        )
        .mean()
        if not tp_df.empty
        else None
    )

    return {
        "qc_sample_size":
            total,

        "overall_correct_count":
            overall_correct,

        "overall_partial_count":
            overall_partial,

        "overall_incorrect_count":
            overall_incorrect,

        "overall_exact_accuracy":
            round_metric(
                overall_exact_accuracy
            ),

        "overall_correct_or_partial_rate":
            round_metric(
                acceptable_rate
            ),

        "overall_weighted_quality":
            round_metric(
                weighted_overall_quality
            ),

        "category_evaluated_count":
            len(category_eval),

        "category_exact_accuracy":
            round_metric(
                category_exact
            ),

        "category_correct_or_partial_rate":
            round_metric(
                category_acceptable
            ),

        "category_weighted_quality":
            round_metric(
                category_weighted
            ),

        "label_evaluated_count":
            len(label_eval),

        "label_exact_accuracy":
            round_metric(
                label_exact
            ),

        "label_correct_or_partial_rate":
            round_metric(
                label_acceptable
            ),

        "label_weighted_quality":
            round_metric(
                label_weighted
            ),

        "true_positive_interpretation_count":
            len(tp_df),

        "true_positive_exact_interpretation_accuracy":
            round_metric(
                tp_overall_correct
            ),

        "true_positive_correct_or_partial_rate":
            round_metric(
                tp_overall_acceptable
            ),
    }


# ============================================================
# ZERO-LABEL EVALUATION
# ============================================================

def calculate_zero_label_metrics(
    df: pd.DataFrame,
) -> dict:

    zero_df = df[
        ~df[
            "ai_has_signal"
        ]
    ]

    if zero_df.empty:

        return {
            "zero_label_sample_count":
                0,

            "zero_label_accuracy":
                None,
        }

    correct_zero = (
        ~zero_df[
            "human_has_signal_bool"
        ]
    ).sum()

    accuracy = safe_divide(
        correct_zero,
        len(zero_df),
    )

    return {
        "zero_label_sample_count":
            len(zero_df),

        "correct_zero_label_count":
            int(
                correct_zero
            ),

        "zero_label_accuracy":
            round_metric(
                accuracy
            ),
    }


# ============================================================
# BUILD CONFUSION MATRIX
# ============================================================

def build_confusion_matrix(
    df: pd.DataFrame,
) -> pd.DataFrame:

    tp = int(
        (
            df[
                "signal_detection_result"
            ]
            == "true_positive"
        ).sum()
    )

    fp = int(
        (
            df[
                "signal_detection_result"
            ]
            == "false_positive"
        ).sum()
    )

    fn = int(
        (
            df[
                "signal_detection_result"
            ]
            == "false_negative"
        ).sum()
    )

    tn = int(
        (
            df[
                "signal_detection_result"
            ]
            == "true_negative"
        ).sum()
    )

    matrix = pd.DataFrame(
        {
            "human_signal_yes": [
                tp,
                fn,
            ],

            "human_signal_no": [
                fp,
                tn,
            ],
        },
        index=[
            "ai_signal_yes",
            "ai_signal_no",
        ],
    )

    matrix.index.name = (
        "prediction"
    )

    return matrix.reset_index()


# ============================================================
# ERROR ANALYSIS
# ============================================================

def build_error_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:

    error_df = df[
        (
            df[
                "human_overall_judgment_canonical"
            ]
            != "correct"
        )
        |
        (
            df[
                "human_error_type_canonical"
            ]
            != "none"
        )
    ].copy()

    if error_df.empty:

        return pd.DataFrame(
            columns=[
                "error_type",
                "count",
                "percent_of_qc_sample",
            ]
        )

    error_summary = (
        error_df[
            "human_error_type_canonical"
        ]
        .value_counts()
        .rename_axis(
            "error_type"
        )
        .reset_index(
            name="count"
        )
    )

    error_summary[
        "percent_of_qc_sample"
    ] = (
        error_summary[
            "count"
        ]
        / len(df)
    ).round(4)

    return error_summary


# ============================================================
# SAMPLE GROUP ANALYSIS
# ============================================================

def build_group_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for group_name, group in (
        df.groupby(
            "qc_sample_group"
        )
    ):

        total = len(group)

        correct = int(
            (
                group[
                    "human_overall_judgment_canonical"
                ]
                == "correct"
            ).sum()
        )

        partial = int(
            (
                group[
                    "human_overall_judgment_canonical"
                ]
                == "partial"
            ).sum()
        )

        incorrect = int(
            (
                group[
                    "human_overall_judgment_canonical"
                ]
                == "incorrect"
            ).sum()
        )

        signal_accuracy = (
            (
                group[
                    "ai_has_signal"
                ]
                == group[
                    "human_has_signal_bool"
                ]
            ).mean()
        )

        category_eval = group[
            group[
                "category_quality_score"
            ]
            .notna()
        ]

        label_eval = group[
            group[
                "label_quality_score"
            ]
            .notna()
        ]

        rows.append(
            {
                "qc_sample_group":
                    group_name,

                "sample_size":
                    total,

                "correct_count":
                    correct,

                "partial_count":
                    partial,

                "incorrect_count":
                    incorrect,

                "exact_accuracy":
                    round_metric(
                        safe_divide(
                            correct,
                            total,
                        )
                    ),

                "correct_or_partial_rate":
                    round_metric(
                        safe_divide(
                            correct
                            + partial,
                            total,
                        )
                    ),

                "signal_detection_accuracy":
                    round_metric(
                        signal_accuracy
                    ),

                "category_weighted_quality":
                    (
                        round_metric(
                            category_eval[
                                "category_quality_score"
                            ].mean()
                        )
                        if not category_eval.empty
                        else None
                    ),

                "label_weighted_quality":
                    (
                        round_metric(
                            label_eval[
                                "label_quality_score"
                            ].mean()
                        )
                        if not label_eval.empty
                        else None
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# OPTIONAL: TAXONOMY ERROR DETAIL
# ============================================================

def extract_ai_categories(
    value,
) -> list[str]:

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    return [
        x.strip()
        for x in text.split("|")
        if x.strip()
    ]


def print_category_error_patterns(
    df: pd.DataFrame,
):

    if "ai_categories" not in df.columns:
        return

    problematic = df[
        (
            df[
                "human_category_judgment_canonical"
            ]
            .isin(
                [
                    "no",
                    "partial",
                ]
            )
        )
    ].copy()

    if problematic.empty:

        print(
            "No category errors found."
        )

        return

    exploded = (
        problematic[
            "ai_categories"
        ]
        .apply(
            extract_ai_categories
        )
        .explode()
    )

    exploded = exploded[
        exploded.notna()
        & (
            exploded
            .astype(str)
            .str.strip()
            != ""
        )
    ]

    if exploded.empty:
        return

    print()
    print(
        "Category frequency among "
        "category-error cases:"
    )

    print(
        exploded
        .value_counts()
        .to_string()
    )


# ============================================================
# BUILD METRICS TABLE
# ============================================================

def build_metrics_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    metrics = {}

    metrics.update(
        calculate_signal_metrics(
            df
        )
    )

    metrics.update(
        calculate_interpretation_metrics(
            df
        )
    )

    metrics.update(
        calculate_zero_label_metrics(
            df
        )
    )

    rows = []

    for metric_name, value in (
        metrics.items()
    ):

        rows.append(
            {
                "metric":
                    metric_name,

                "value":
                    value,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PRINT REPORT
# ============================================================

def print_report(
    evaluated_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    error_df: pd.DataFrame,
):

    metrics = dict(
        zip(
            metrics_df[
                "metric"
            ],
            metrics_df[
                "value"
            ],
        )
    )

    print()
    print(
        "=========================================="
    )
    print(
        "3B.1 Human Evaluation Results"
    )
    print(
        "=========================================="
    )

    print(
        f"QC sample size: "
        f"{int(metrics['qc_sample_size'])}"
    )

    print()
    print(
        "Signal Detection"
    )
    print(
        "------------------------------------------"
    )

    print(
        f"TP: "
        f"{int(metrics['true_positive'])}"
    )

    print(
        f"FP: "
        f"{int(metrics['false_positive'])}"
    )

    print(
        f"FN: "
        f"{int(metrics['false_negative'])}"
    )

    print(
        f"TN: "
        f"{int(metrics['true_negative'])}"
    )

    print(
        f"Precision : "
        f"{float(metrics['signal_precision']):.1%}"
    )

    print(
        f"Recall    : "
        f"{float(metrics['signal_recall']):.1%}"
    )

    print(
        f"F1        : "
        f"{float(metrics['signal_f1']):.1%}"
    )

    print(
        f"Accuracy  : "
        f"{float(metrics['signal_accuracy']):.1%}"
    )

    print()
    print(
        "Interpretation Quality"
    )
    print(
        "------------------------------------------"
    )

    print(
        f"Overall exact accuracy       : "
        f"{float(metrics['overall_exact_accuracy']):.1%}"
    )

    print(
        f"Correct + partial rate       : "
        f"{float(metrics['overall_correct_or_partial_rate']):.1%}"
    )

    print(
        f"Category exact accuracy      : "
        f"{float(metrics['category_exact_accuracy']):.1%}"
    )

    print(
        f"Category weighted quality    : "
        f"{float(metrics['category_weighted_quality']):.1%}"
    )

    print(
        f"Label exact accuracy         : "
        f"{float(metrics['label_exact_accuracy']):.1%}"
    )

    print(
        f"Label weighted quality       : "
        f"{float(metrics['label_weighted_quality']):.1%}"
    )

    print(
        f"Zero-label accuracy          : "
        f"{float(metrics['zero_label_accuracy']):.1%}"
    )

    print()
    print(
        "Error Analysis"
    )
    print(
        "------------------------------------------"
    )

    if error_df.empty:

        print(
            "No reviewed errors."
        )

    else:

        print(
            error_df.to_string(
                index=False
            )
        )

    print_category_error_patterns(
        evaluated_df
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading reviewed human QC..."
    )

    raw_df = load_data()

    print(
        f"Loaded {len(raw_df)} reviewed rows."
    )

    evaluated_df = (
        prepare_evaluation_data(
            raw_df
        )
    )

    # ========================================================
    # CALCULATE OUTPUTS
    # ========================================================

    metrics_df = (
        build_metrics_table(
            evaluated_df
        )
    )

    confusion_df = (
        build_confusion_matrix(
            evaluated_df
        )
    )

    error_df = (
        build_error_analysis(
            evaluated_df
        )
    )

    group_df = (
        build_group_analysis(
            evaluated_df
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    metrics_df.to_csv(
        METRICS_OUTPUT,
        index=False,
    )

    confusion_df.to_csv(
        CONFUSION_OUTPUT,
        index=False,
    )

    error_df.to_csv(
        ERROR_OUTPUT,
        index=False,
    )

    group_df.to_csv(
        GROUP_OUTPUT,
        index=False,
    )

    evaluated_df.to_csv(
        EVALUATED_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # REPORT
    # ========================================================

    print_report(
        evaluated_df,
        metrics_df,
        error_df,
    )

    print()
    print(
        "Saved:"
    )

    print(
        f"- {METRICS_OUTPUT}"
    )

    print(
        f"- {CONFUSION_OUTPUT}"
    )

    print(
        f"- {ERROR_OUTPUT}"
    )

    print(
        f"- {GROUP_OUTPUT}"
    )

    print(
        f"- {EVALUATED_OUTPUT}"
    )


if __name__ == "__main__":
    main()