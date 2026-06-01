"""Validate and process annotation data from the data folder."""

from pathlib import Path

import numpy as np
import pandas as pd


VALID_LABELS = {"accept", "reject", "needs_revision"}
VALID_CONFIDENCE = {"low", "medium", "high"}
VALID_ANNOTATORS = [f"annotator_{index}" for index in range(1, 6)]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_PATH = DATA_DIR / "annotations.csv"
VALIDATED_PATH = DATA_DIR / "validated_annotations.csv"
ERRORS_PATH = DATA_DIR / "validation_errors.csv"
ANNOTATOR_STATS_PATH = DATA_DIR / "annotator_stats.csv"



def load_annotations(input_path: Path) -> pd.DataFrame:
    """Load the raw annotations CSV from disk."""

    if not input_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {input_path}")
    return pd.read_csv(input_path)



def normalize_value(value: object) -> str:
    """Convert a value into a stripped string for validation logging."""

    if pd.isna(value):
        return ""
    return str(value).strip()



def validate_score(value: object) -> tuple[bool, str]:
    """Check whether a score is an integer between 1 and 5."""

    if pd.isna(value):
        return False, "value is missing"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False, f"value '{value}' is not numeric"
    if not np.isfinite(numeric_value):
        return False, f"value '{value}' is not finite"
    if numeric_value != int(numeric_value):
        return False, f"value '{value}' is not an integer"
    if not 1 <= int(numeric_value) <= 5:
        return False, f"value '{value}' is outside the allowed range 1-5"
    return True, ""



def validate_timestamp(value: object) -> tuple[bool, str]:
    """Check whether a timestamp can be parsed as a valid datetime."""

    parsed_value = pd.to_datetime([value], errors="coerce")[0]
    if pd.isna(parsed_value):
        return False, f"value '{value}' is not a valid datetime"
    return True, ""



def build_error_record(row: pd.Series, error_type: str, error_detail: str) -> dict[str, str]:
    """Build a standardized validation-error record for one failed rule."""

    return {
        "question_id": normalize_value(row.get("question_id")),
        "annotator_id": normalize_value(row.get("annotator_id")),
        "error_type": error_type,
        "error_detail": error_detail,
    }



def validate_rows(annotations_df: pd.DataFrame) -> pd.DataFrame:
    """Validate every annotation row and return all schema violations."""

    error_records: list[dict[str, str]] = []
    for _, row in annotations_df.iterrows():
        answer_quality_ok, answer_quality_detail = validate_score(row.get("answer_quality"))
        if not answer_quality_ok:
            error_records.append(
                build_error_record(row, "invalid_answer_quality", answer_quality_detail)
            )

        question_clarity_ok, question_clarity_detail = validate_score(row.get("question_clarity"))
        if not question_clarity_ok:
            error_records.append(
                build_error_record(row, "invalid_question_clarity", question_clarity_detail)
            )

        label_value = normalize_value(row.get("label"))
        if label_value not in VALID_LABELS:
            error_records.append(
                build_error_record(
                    row,
                    "invalid_label",
                    f"value '{row.get('label')}' is not one of {sorted(VALID_LABELS)}",
                )
            )

        confidence_value = normalize_value(row.get("confidence"))
        if confidence_value not in VALID_CONFIDENCE:
            error_records.append(
                build_error_record(
                    row,
                    "invalid_confidence",
                    f"value '{row.get('confidence')}' is not one of {sorted(VALID_CONFIDENCE)}",
                )
            )

        annotator_value = normalize_value(row.get("annotator_id"))
        if annotator_value not in VALID_ANNOTATORS:
            error_records.append(
                build_error_record(
                    row,
                    "invalid_annotator_id",
                    f"value '{row.get('annotator_id')}' is not one of {VALID_ANNOTATORS}",
                )
            )

        question_id_value = normalize_value(row.get("question_id"))
        if question_id_value == "":
            error_records.append(
                build_error_record(row, "invalid_question_id", "question_id is null or empty")
            )

        timestamp_ok, timestamp_detail = validate_timestamp(row.get("timestamp"))
        if not timestamp_ok:
            error_records.append(
                build_error_record(row, "invalid_timestamp", timestamp_detail)
            )

    return pd.DataFrame(
        error_records,
        columns=["question_id", "annotator_id", "error_type", "error_detail"],
    )



def calculate_entropy(labels: pd.Series) -> float:
    """Calculate Shannon entropy for a label distribution."""

    value_counts = labels.value_counts(dropna=False)
    probabilities = value_counts / value_counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())



def summarize_question_group(group: pd.DataFrame) -> pd.Series:
    """Summarize disagreement metrics for a single question group."""

    labels = group["label"].fillna("__MISSING__").astype(str).str.strip()
    label_counts = labels.value_counts()
    highest_count = int(label_counts.max())
    top_labels = sorted(label_counts[label_counts == highest_count].index.tolist())
    majority_label = top_labels[0]
    is_conflicted = highest_count <= (len(group) / 2)
    conflict_score = int((labels != majority_label).sum())
    label_entropy = calculate_entropy(labels)
    return pd.Series(
        {
            "majority_label": majority_label,
            "is_conflicted": bool(is_conflicted),
            "conflict_score": conflict_score,
            "label_entropy": label_entropy,
        }
    )



def add_conflict_metrics(annotations_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add per-question conflict metrics back onto the original annotations data."""

    question_mask = (
        annotations_df["question_id"].notna()
        & annotations_df["question_id"].astype(str).str.strip().ne("")
    )
    question_metrics = (
        annotations_df.loc[question_mask]
        .groupby("question_id", sort=True)
        .apply(summarize_question_group)
        .reset_index()
    )
    validated_df = annotations_df.merge(question_metrics, on="question_id", how="left")
    return validated_df, question_metrics



def coerce_valid_score_series(series: pd.Series) -> pd.Series:
    """Convert a score series to numeric values only when the values are valid 1-5 integers."""

    numeric_series = pd.to_numeric(series, errors="coerce")
    valid_mask = (
        numeric_series.notna()
        & np.isfinite(numeric_series)
        & (numeric_series >= 1)
        & (numeric_series <= 5)
        & (numeric_series == np.floor(numeric_series))
    )
    return numeric_series.where(valid_mask)



def calculate_annotator_stats(validated_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate consistency and agreement statistics for each valid annotator."""

    stats_df = validated_df.copy()
    stats_df["annotator_key"] = stats_df["annotator_id"].apply(normalize_value)
    stats_df["answer_quality_numeric"] = coerce_valid_score_series(stats_df["answer_quality"])
    stats_df["question_clarity_numeric"] = coerce_valid_score_series(stats_df["question_clarity"])
    comparable_mask = stats_df["majority_label"].notna()
    label_mismatch = (
        stats_df["label"].fillna("").astype(str).str.strip()
        != stats_df["majority_label"].fillna("").astype(str).str.strip()
    ).astype(float)
    stats_df["differs_from_majority"] = label_mismatch.where(comparable_mask, np.nan)

    grouped = (
        stats_df.loc[stats_df["annotator_key"].isin(VALID_ANNOTATORS)]
        .groupby("annotator_key")
        .agg(
            avg_answer_quality=("answer_quality_numeric", "mean"),
            avg_question_clarity=("question_clarity_numeric", "mean"),
            error_rate=("differs_from_majority", "mean"),
        )
        .reindex(VALID_ANNOTATORS)
        .reset_index()
        .rename(columns={"annotator_key": "annotator_id"})
    )
    grouped["agreement_rate"] = 1 - grouped["error_rate"]

    mean_error_rate = grouped["error_rate"].mean(skipna=True)
    std_error_rate = grouped["error_rate"].std(skipna=True, ddof=0)
    threshold = mean_error_rate + (1.5 * std_error_rate)
    grouped["is_outlier"] = grouped["error_rate"] > threshold

    numeric_columns = [
        "avg_answer_quality",
        "avg_question_clarity",
        "error_rate",
        "agreement_rate",
    ]
    grouped[numeric_columns] = grouped[numeric_columns].round(4)
    grouped["is_outlier"] = grouped["is_outlier"].fillna(False)
    return grouped



def save_outputs(
    validated_df: pd.DataFrame,
    validation_errors_df: pd.DataFrame,
    annotator_stats_df: pd.DataFrame,
) -> None:
    """Save all processed CSV outputs to the data folder."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    validated_df.to_csv(VALIDATED_PATH, index=False)
    validation_errors_df.to_csv(ERRORS_PATH, index=False)
    annotator_stats_df.to_csv(ANNOTATOR_STATS_PATH, index=False)



def print_report(
    annotations_df: pd.DataFrame,
    question_metrics_df: pd.DataFrame,
    validation_errors_df: pd.DataFrame,
    annotator_stats_df: pd.DataFrame,
) -> None:
    """Print the requested annotation pipeline summary report."""

    total_annotations = len(annotations_df)
    total_questions = len(question_metrics_df)
    conflicted_questions = int(question_metrics_df["is_conflicted"].sum()) if total_questions else 0
    clean_questions = total_questions - conflicted_questions
    conflicted_pct = conflicted_questions / total_questions if total_questions else 0
    clean_pct = clean_questions / total_questions if total_questions else 0
    flagged_annotators = annotator_stats_df.loc[
        annotator_stats_df["is_outlier"], "annotator_id"
    ].tolist()

    print("=== ANNOTATION PIPELINE REPORT ===")
    print(f"Total annotations processed: {total_annotations}")
    print(f"Total questions covered: {total_questions}")
    print(f"Schema validation errors: {len(validation_errors_df)}")
    print(f"Conflicted questions: {conflicted_questions} ({conflicted_pct:.2%})")
    print(f"Clean questions: {clean_questions} ({clean_pct:.2%})")
    print(f"Flagged annotators: {flagged_annotators}")
    print("===================================")



def main() -> None:
    """Run the full annotation validation and processing pipeline."""

    print("Loading annotation data...")
    annotations_df = load_annotations(INPUT_PATH)

    print("Running schema validation...")
    validation_errors_df = validate_rows(annotations_df)

    print("Computing conflict metrics...")
    validated_df, question_metrics_df = add_conflict_metrics(annotations_df)

    print("Calculating annotator consistency statistics...")
    annotator_stats_df = calculate_annotator_stats(validated_df)

    print("Saving processed outputs...")
    save_outputs(validated_df, validation_errors_df, annotator_stats_df)

    print_report(
        annotations_df=annotations_df,
        question_metrics_df=question_metrics_df,
        validation_errors_df=validation_errors_df,
        annotator_stats_df=annotator_stats_df,
    )


if __name__ == "__main__":
    main()

