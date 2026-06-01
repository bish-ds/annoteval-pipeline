"""Handle SQLite logging for pipeline runs and annotator statistics."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "pipeline_logs.db"
PIPELINE_RUN_COLUMNS = [
    "run_timestamp",
    "total_questions",
    "total_annotations",
    "validation_errors",
    "conflicted_questions",
    "conflict_rate",
    "fleiss_kappa",
    "kappa_interpretation",
    "avg_faithfulness",
    "avg_relevance",
    "avg_completeness",
    "avg_llm_score",
    "notes",
]



def get_connection() -> sqlite3.Connection:
    """Create and return a SQLite connection for the pipeline log database."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection



def init_db() -> None:
    """Create the pipeline logging tables if they do not already exist."""

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp TEXT,
                total_questions INTEGER,
                total_annotations INTEGER,
                validation_errors INTEGER,
                conflicted_questions INTEGER,
                conflict_rate REAL,
                fleiss_kappa REAL,
                kappa_interpretation TEXT,
                avg_faithfulness REAL,
                avg_relevance REAL,
                avg_completeness REAL,
                avg_llm_score REAL,
                notes TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS annotator_stats_log (
                stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                annotator_id TEXT,
                avg_answer_quality REAL,
                avg_question_clarity REAL,
                error_rate REAL,
                agreement_rate REAL,
                is_outlier INTEGER,
                FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
            )
            """
        )
        connection.commit()



def log_run(stats_dict: dict[str, object]) -> int:
    """Insert one pipeline-run record and return the newly created run_id."""

    values = [stats_dict.get(column) for column in PIPELINE_RUN_COLUMNS]
    placeholders = ", ".join(["?"] * len(PIPELINE_RUN_COLUMNS))
    columns = ", ".join(PIPELINE_RUN_COLUMNS)

    with get_connection() as connection:
        cursor = connection.execute(
            f"INSERT INTO pipeline_runs ({columns}) VALUES ({placeholders})",
            values,
        )
        connection.commit()
        return int(cursor.lastrowid)



def log_annotator_stats(run_id: int, annotator_df: pd.DataFrame) -> None:
    """Insert annotator statistics for a specific run into the log table."""

    if annotator_df.empty:
        return

    records = []
    for row in annotator_df.to_dict(orient="records"):
        records.append(
            (
                run_id,
                row.get("annotator_id"),
                row.get("avg_answer_quality"),
                row.get("avg_question_clarity"),
                row.get("error_rate"),
                row.get("agreement_rate"),
                int(bool(row.get("is_outlier"))) if pd.notna(row.get("is_outlier")) else 0,
            )
        )

    with get_connection() as connection:
        connection.executemany(
            """
            INSERT INTO annotator_stats_log (
                run_id,
                annotator_id,
                avg_answer_quality,
                avg_question_clarity,
                error_rate,
                agreement_rate,
                is_outlier
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        connection.commit()



def get_run_history() -> pd.DataFrame:
    """Return all logged pipeline runs ordered by run_timestamp."""

    with get_connection() as connection:
        return pd.read_sql_query(
            "SELECT * FROM pipeline_runs ORDER BY run_timestamp",
            connection,
        )



def get_latest_run() -> dict[str, object]:
    """Return the most recent pipeline run as a dictionary."""

    with get_connection() as connection:
        latest_df = pd.read_sql_query(
            "SELECT * FROM pipeline_runs ORDER BY run_timestamp DESC, run_id DESC LIMIT 1",
            connection,
        )
    if latest_df.empty:
        return {}
    latest_record = latest_df.iloc[0].to_dict()
    return latest_record



def get_annotator_history(annotator_id: str) -> pd.DataFrame:
    """Return all logged statistics for one annotator across pipeline runs."""

    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM annotator_stats_log
            WHERE annotator_id = ?
            ORDER BY run_id
            """,
            connection,
            params=[annotator_id],
        )



def main() -> None:
    """Initialize the database, print run history if present, and confirm setup."""

    init_db()
    run_history_df = get_run_history()
    if not run_history_df.empty:
        print(run_history_df.to_string(index=False))
    print("Database initialized successfully")


if __name__ == "__main__":
    main()
