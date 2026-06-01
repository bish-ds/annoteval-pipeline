"""Run the full LLM annotation quality pipeline end to end."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TOTAL_STEPS = 6



def load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    """Load a CSV file if it exists, otherwise return None."""

    if not path.exists():
        return None
    return pd.read_csv(path)



def load_json_if_exists(path: Path) -> dict | None:
    """Load a JSON file if it exists, otherwise return None."""

    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)



def safe_mean(series: pd.Series | None) -> float | None:
    """Return a numeric mean or None when a metric series is unavailable."""

    if series is None:
        return None
    numeric_series = pd.to_numeric(series, errors="coerce")
    if numeric_series.dropna().empty:
        return None
    return float(numeric_series.mean())



def build_run_stats(step_results: list[dict[str, str]]) -> tuple[dict[str, object], pd.DataFrame]:
    """Build the database payload for a completed pipeline run."""

    validated_df = load_csv_if_exists(DATA_DIR / "validated_annotations.csv")
    validation_errors_df = load_csv_if_exists(DATA_DIR / "validation_errors.csv")
    annotator_stats_df = load_csv_if_exists(DATA_DIR / "annotator_stats.csv")
    llm_eval_df = load_csv_if_exists(DATA_DIR / "llm_eval_scores.csv")
    agreement_summary = load_json_if_exists(DATA_DIR / "agreement_summary.json") or {}

    total_questions = 0
    total_annotations = 0
    conflicted_questions = 0
    conflict_rate = None
    if validated_df is not None and not validated_df.empty:
        total_annotations = int(len(validated_df))
        question_metrics = validated_df[["question_id", "is_conflicted"]].copy()
        question_metrics["question_id"] = question_metrics["question_id"].astype(str).str.strip()
        question_metrics = question_metrics.loc[question_metrics["question_id"] != ""]
        question_metrics = question_metrics.drop_duplicates(subset=["question_id"])
        total_questions = int(len(question_metrics))
        conflicted_questions = int(question_metrics["is_conflicted"].fillna(False).astype(bool).sum())
        conflict_rate = float(conflicted_questions / total_questions) if total_questions else None

    validation_errors = 0 if validation_errors_df is None else int(len(validation_errors_df))

    avg_faithfulness = None
    avg_relevance = None
    avg_completeness = None
    avg_llm_score = None
    if llm_eval_df is not None and not llm_eval_df.empty and "eval_status" in llm_eval_df.columns:
        success_df = llm_eval_df.loc[llm_eval_df["eval_status"] == "success"].copy()
        avg_faithfulness = safe_mean(success_df.get("faithfulness"))
        avg_relevance = safe_mean(success_df.get("relevance"))
        avg_completeness = safe_mean(success_df.get("completeness"))
        avg_llm_score = safe_mean(success_df.get("avg_score"))

    failed_steps = [result["module_name"] for result in step_results if result["status"] == "failed"]
    notes = (
        "All pipeline steps completed successfully"
        if not failed_steps
        else f"Failed steps: {', '.join(failed_steps)}"
    )

    stats_dict = {
        "run_timestamp": pd.Timestamp.now().isoformat(),
        "total_questions": total_questions,
        "total_annotations": total_annotations,
        "validation_errors": validation_errors,
        "conflicted_questions": conflicted_questions,
        "conflict_rate": conflict_rate,
        "fleiss_kappa": agreement_summary.get("fleiss_kappa"),
        "kappa_interpretation": agreement_summary.get("interpretation"),
        "avg_faithfulness": avg_faithfulness,
        "avg_relevance": avg_relevance,
        "avg_completeness": avg_completeness,
        "avg_llm_score": avg_llm_score,
        "notes": notes,
    }

    if annotator_stats_df is None:
        annotator_stats_df = pd.DataFrame()
    return stats_dict, annotator_stats_df



def run_step(step_number: int, module_name: str, module_path: str) -> dict[str, str]:
    """Run one pipeline module, report success or failure, and continue."""

    try:
        module = importlib.import_module(module_path)
        module.main()
        print(f"✅ Step {step_number}/{TOTAL_STEPS} complete: {module_name}")
        return {"module_name": module_name, "status": "success"}
    except Exception as exc:
        print(f"❌ Step {step_number}/{TOTAL_STEPS} failed: {module_name} - {exc}")
        return {"module_name": module_name, "status": "failed"}



def log_pipeline_run(step_results: list[dict[str, str]]) -> None:
    """Log the latest pipeline run and annotator statistics to SQLite."""

    try:
        db_module = importlib.import_module("src.db")
        db_module.init_db()
        stats_dict, annotator_stats_df = build_run_stats(step_results)
        run_id = db_module.log_run(stats_dict)
        db_module.log_annotator_stats(run_id, annotator_stats_df)
        print(f"Logged pipeline run to SQLite with run_id={run_id}")
    except Exception as exc:
        print(f"Failed to log pipeline run: {exc}")



def main() -> None:
    """Execute every pipeline module in order and persist the run summary."""

    print("🚀 Starting LLM Annotation Quality Pipeline...")
    steps = [
        ("generate_data", "src.generate_data"),
        ("annotation_pipeline", "src.annotation_pipeline"),
        ("agreement_metrics", "src.agreement_metrics"),
        ("llm_evaluator", "src.llm_evaluator"),
        ("aws_storage", "src.aws_storage"),
        ("db", "src.db"),
    ]

    step_results = []
    for index, (module_name, module_path) in enumerate(steps, start=1):
        step_results.append(run_step(index, module_name, module_path))

    log_pipeline_run(step_results)
    print("🎉 Pipeline complete! Run streamlit run dashboard/app.py to view results")


if __name__ == "__main__":
    main()
