"""Compute inter-annotator agreement metrics for validated annotation data.

Cohen's Kappa measures pairwise agreement between two annotators while accounting
for agreement that could happen by chance. Fleiss Kappa extends this idea to
multiple annotators rating the same items at the same time. Percent agreement is
simpler: it measures how often two annotators picked the exact same label.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


VALID_ANNOTATORS = [f"annotator_{index}" for index in range(1, 6)]
VALID_LABELS = ["accept", "reject", "needs_revision"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
VALIDATED_PATH = DATA_DIR / "validated_annotations.csv"
ANNOTATOR_STATS_PATH = DATA_DIR / "annotator_stats.csv"
COHEN_KAPPA_PATH = DATA_DIR / "cohen_kappa_scores.csv"
PERCENT_AGREEMENT_PATH = DATA_DIR / "percent_agreement.csv"
CONTESTED_QUESTIONS_PATH = DATA_DIR / "contested_questions.csv"
AGREEMENT_SUMMARY_PATH = DATA_DIR / "agreement_summary.json"



def load_validated_annotations(input_path: Path) -> pd.DataFrame:
    """Load validated annotations that already include conflict metrics."""

    if not input_path.exists():
        raise FileNotFoundError(f"Validated annotations file not found: {input_path}")
    return pd.read_csv(input_path)



def load_annotator_stats(input_path: Path) -> pd.DataFrame:
    """Load per-annotator statistics used for the final report and analysis."""

    if not input_path.exists():
        raise FileNotFoundError(f"Annotator stats file not found: {input_path}")
    return pd.read_csv(input_path)



def normalize_text(value: object) -> str:
    """Normalize a text-like value for safe agreement calculations."""

    if pd.isna(value):
        return ""
    return str(value).strip()



def prepare_metric_rows(validated_df: pd.DataFrame) -> pd.DataFrame:
    """Filter validated annotations down to rows suitable for agreement metrics."""

    metric_df = validated_df.copy()
    metric_df["question_id"] = metric_df["question_id"].apply(normalize_text)
    metric_df["annotator_id"] = metric_df["annotator_id"].apply(normalize_text)
    metric_df["label"] = metric_df["label"].apply(normalize_text)
    metric_df = metric_df.loc[
        metric_df["question_id"].ne("")
        & metric_df["annotator_id"].isin(VALID_ANNOTATORS)
        & metric_df["label"].isin(VALID_LABELS)
    ].copy()
    metric_df = metric_df.drop_duplicates(subset=["question_id", "annotator_id"], keep="first")
    return metric_df



def build_label_matrix(metric_df: pd.DataFrame) -> pd.DataFrame:
    """Build a question-by-annotator label matrix for pairwise comparisons."""

    return metric_df.pivot(index="question_id", columns="annotator_id", values="label").reindex(
        columns=VALID_ANNOTATORS
    )



def interpret_kappa(score: float) -> str:
    """Map a kappa score to a standard qualitative agreement interpretation."""

    if pd.isna(score):
        return "Unavailable"
    if score < 0.00:
        return "Poor"
    if score <= 0.20:
        return "Slight"
    if score <= 0.40:
        return "Fair"
    if score <= 0.60:
        return "Moderate"
    if score <= 0.80:
        return "Substantial"
    return "Almost Perfect"



def compute_pairwise_cohen_kappas(label_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute Cohen's Kappa for every annotator pair on shared question IDs only."""

    records: list[dict[str, object]] = []
    for annotator_a, annotator_b in combinations(VALID_ANNOTATORS, 2):
        aligned = label_matrix[[annotator_a, annotator_b]].dropna()
        if aligned.empty:
            kappa_score = np.nan
        else:
            kappa_score = float(cohen_kappa_score(aligned[annotator_a], aligned[annotator_b]))
        records.append(
            {
                "annotator_a": annotator_a,
                "annotator_b": annotator_b,
                "kappa_score": kappa_score,
                "agreement_level": interpret_kappa(kappa_score),
            }
        )
    return pd.DataFrame(records)



def compute_percent_agreement_matrix(label_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute simple percent agreement for every annotator pair as a matrix."""

    agreement_matrix = pd.DataFrame(index=VALID_ANNOTATORS, columns=VALID_ANNOTATORS, dtype=float)
    for annotator in VALID_ANNOTATORS:
        agreement_matrix.loc[annotator, annotator] = 100.0
    for annotator_a, annotator_b in combinations(VALID_ANNOTATORS, 2):
        aligned = label_matrix[[annotator_a, annotator_b]].dropna()
        if aligned.empty:
            agreement_pct = np.nan
        else:
            agreement_pct = float((aligned[annotator_a] == aligned[annotator_b]).mean() * 100)
        agreement_matrix.loc[annotator_a, annotator_b] = agreement_pct
        agreement_matrix.loc[annotator_b, annotator_a] = agreement_pct
    agreement_matrix.index.name = "annotator_id"
    return agreement_matrix.round(2)



def build_fleiss_count_matrix(label_matrix: pd.DataFrame) -> pd.DataFrame:
    """Build the question-by-category count matrix required for Fleiss Kappa."""

    complete_matrix = label_matrix.dropna()
    count_matrix = pd.DataFrame(index=complete_matrix.index)
    for label in VALID_LABELS:
        count_matrix[label] = (complete_matrix == label).sum(axis=1)
    return count_matrix



def compute_fleiss_kappa(count_matrix: pd.DataFrame) -> float:
    """Compute Fleiss Kappa from scratch for multi-annotator agreement.

    Fleiss Kappa measures how much five annotators agree across the full set of
    questions after accounting for the agreement expected by chance.
    """

    if count_matrix.empty:
        return float("nan")

    matrix = count_matrix.to_numpy(dtype=float)
    num_questions = matrix.shape[0]
    ratings_per_question = matrix.sum(axis=1)
    if np.any(ratings_per_question != ratings_per_question[0]):
        raise ValueError("Fleiss Kappa requires the same number of ratings for each question.")

    total_annotators = ratings_per_question[0]
    if total_annotators <= 1:
        return float("nan")

    p_i = ((matrix * (matrix - 1)).sum(axis=1)) / (total_annotators * (total_annotators - 1))
    p_j = matrix.sum(axis=0) / (num_questions * total_annotators)
    p_bar = p_i.mean()
    p_e_bar = np.square(p_j).sum()
    denominator = 1 - p_e_bar
    if denominator == 0:
        return float("nan")
    return float((p_bar - p_e_bar) / denominator)



def build_agreement_summary(fleiss_kappa: float, total_questions: int) -> dict[str, object]:
    """Build the Fleiss Kappa summary payload for JSON export."""

    return {
        "fleiss_kappa": None if pd.isna(fleiss_kappa) else round(float(fleiss_kappa), 4),
        "interpretation": interpret_kappa(fleiss_kappa),
        "total_questions": int(total_questions),
        "total_annotators": len(VALID_ANNOTATORS),
    }



def select_contested_questions(validated_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Select the most contested questions using conflict score and entropy."""

    question_df = validated_df.copy()
    question_df["question_id"] = question_df["question_id"].apply(normalize_text)
    question_df = question_df.loc[question_df["question_id"].ne("")].copy()
    columns = ["question_id", "majority_label", "is_conflicted", "conflict_score", "label_entropy"]
    question_df = question_df[columns].drop_duplicates(subset=["question_id"], keep="first")
    question_df = question_df.sort_values(
        by=["conflict_score", "label_entropy", "question_id"],
        ascending=[False, False, True],
    )
    return question_df.head(top_n).reset_index(drop=True)



def save_pairwise_kappas(kappa_df: pd.DataFrame, output_path: Path) -> None:
    """Save pairwise Cohen's Kappa scores to CSV."""

    kappa_output = kappa_df.copy()
    kappa_output["kappa_score"] = kappa_output["kappa_score"].round(4)
    kappa_output.to_csv(output_path, index=False)



def save_percent_agreement(agreement_matrix: pd.DataFrame, output_path: Path) -> None:
    """Save the percent-agreement matrix to CSV."""

    agreement_matrix.to_csv(output_path)



def save_contested_questions(contested_df: pd.DataFrame, output_path: Path) -> None:
    """Save the top contested questions to CSV."""

    contested_output = contested_df.copy()
    contested_output["label_entropy"] = contested_output["label_entropy"].round(4)
    contested_output.to_csv(output_path, index=False)



def save_agreement_summary(summary: dict[str, object], output_path: Path) -> None:
    """Save the Fleiss Kappa summary to a JSON file."""

    with output_path.open("w", encoding="ascii") as output_file:
        json.dump(summary, output_file, indent=2)



def choose_best_pair(kappa_df: pd.DataFrame) -> pd.Series:
    """Select the annotator pair with the highest Cohen's Kappa score."""

    scored_df = kappa_df.dropna(subset=["kappa_score"]).sort_values(
        by=["kappa_score", "annotator_a", "annotator_b"], ascending=[False, True, True]
    )
    return scored_df.iloc[0] if not scored_df.empty else pd.Series(dtype=object)



def choose_worst_pair(kappa_df: pd.DataFrame) -> pd.Series:
    """Select the annotator pair with the lowest Cohen's Kappa score."""

    scored_df = kappa_df.dropna(subset=["kappa_score"]).sort_values(
        by=["kappa_score", "annotator_a", "annotator_b"], ascending=[True, True, True]
    )
    return scored_df.iloc[0] if not scored_df.empty else pd.Series(dtype=object)



def format_pair_summary(pair_row: pd.Series) -> str:
    """Format a best-pair or worst-pair summary line for the final report."""

    if pair_row.empty:
        return "Unavailable"
    return (
        f"{pair_row['annotator_a']} vs {pair_row['annotator_b']} = "
        f"{float(pair_row['kappa_score']):.2f}"
    )



def print_report(fleiss_kappa: float, kappa_df: pd.DataFrame, annotator_stats_df: pd.DataFrame) -> None:
    """Print a formatted agreement report for quick inspection.

    The report highlights overall agreement across all five annotators,
    pairwise agreement strength, and each annotator's observed error rate.
    """

    best_pair = choose_best_pair(kappa_df)
    worst_pair = choose_worst_pair(kappa_df)
    average_kappa = kappa_df["kappa_score"].dropna().mean()

    print("=== AGREEMENT METRICS REPORT ===")
    print(f"Fleiss Kappa: {fleiss_kappa:.2f} ({interpret_kappa(fleiss_kappa)})")
    print()
    print("Pairwise Cohen's Kappa Summary:")
    print(f"Best pair: {format_pair_summary(best_pair)}")
    print(f"Worst pair: {format_pair_summary(worst_pair)}")
    print(f"Average kappa: {average_kappa:.2f}")
    print()
    print("Annotator Error Rates:")
    stats_lookup = annotator_stats_df.set_index("annotator_id") if not annotator_stats_df.empty else pd.DataFrame()
    for annotator_id in VALID_ANNOTATORS:
        if annotator_id in getattr(stats_lookup, "index", []):
            error_rate = pd.to_numeric(stats_lookup.loc[annotator_id, "error_rate"], errors="coerce")
        else:
            error_rate = np.nan
        error_text = f"{error_rate * 100:.1f}%" if pd.notna(error_rate) else "nan%"
        print(f"{annotator_id}: {error_text}")
    print("================================")



def main() -> None:
    """Run the full inter-annotator agreement analysis workflow."""

    print("Loading validated annotations...")
    validated_df = load_validated_annotations(VALIDATED_PATH)

    print("Loading annotator statistics...")
    annotator_stats_df = load_annotator_stats(ANNOTATOR_STATS_PATH)

    print("Preparing aligned label matrix...")
    metric_df = prepare_metric_rows(validated_df)
    label_matrix = build_label_matrix(metric_df)

    print("Computing pairwise Cohen's Kappa scores...")
    kappa_df = compute_pairwise_cohen_kappas(label_matrix)

    print("Computing Fleiss Kappa across all five annotators...")
    count_matrix = build_fleiss_count_matrix(label_matrix)
    fleiss_kappa = compute_fleiss_kappa(count_matrix)
    agreement_summary = build_agreement_summary(fleiss_kappa, len(count_matrix))

    print("Computing percent agreement matrix...")
    percent_agreement_df = compute_percent_agreement_matrix(label_matrix)

    print("Selecting the most contested questions...")
    contested_questions_df = select_contested_questions(validated_df, top_n=20)

    print("Saving agreement outputs...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_pairwise_kappas(kappa_df, COHEN_KAPPA_PATH)
    save_percent_agreement(percent_agreement_df, PERCENT_AGREEMENT_PATH)
    save_contested_questions(contested_questions_df, CONTESTED_QUESTIONS_PATH)
    save_agreement_summary(agreement_summary, AGREEMENT_SUMMARY_PATH)

    print_report(fleiss_kappa, kappa_df, annotator_stats_df)


if __name__ == "__main__":
    main()
