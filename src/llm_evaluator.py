"""Generate LLM answers with OpenAI and score them with an LLM-as-judge pass.

Faithfulness measures whether the answer matches the reference facts.
Relevance measures whether the answer directly addresses the question.
Completeness measures whether the answer covers the key information needed.
"""

import json
import os
import time

import openai
import pandas as pd
import dotenv


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESPONSES_PATH = os.path.join(DATA_DIR, "llm_responses.csv")
EVAL_SCORES_PATH = os.path.join(DATA_DIR, "llm_eval_scores.csv")
MODEL_NAME = "gpt-4o-mini"



def normalize_text(value: object) -> str:
    """Convert a value to a stripped string for prompt and status checks."""

    if pd.isna(value):
        return ""
    return str(value).strip()



def load_environment_and_client() -> openai.OpenAI:
    """Load environment variables from .env and initialize the OpenAI client."""

    dotenv.load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Update the .env file before running this script.")
    return openai.OpenAI(api_key=api_key)



def load_llm_responses(input_path: str) -> pd.DataFrame:
    """Load the LLM response placeholder dataset from CSV."""

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"LLM responses file not found: {input_path}")
    llm_df = pd.read_csv(input_path)
    if "generated_at" not in llm_df.columns:
        llm_df["generated_at"] = ""
    llm_df["generated_at"] = llm_df["generated_at"].fillna("")
    return llm_df



def build_generation_prompt(context: str, question: str) -> str:
    """Build the exact prompt used to generate an answer from the provided context."""

    return (
        "Answer the following question based only on the given context. \n"
        "Be concise — answer in 1-2 sentences maximum.\n\n"
        f"Context: {context}\n"
        f"Question: {question}\n\n"
        "Answer:"
    )



def request_model_text(client: openai.OpenAI, prompt: str) -> str:
    """Send a prompt to the model and return the text output."""

    response = client.responses.create(model=MODEL_NAME, input=prompt)
    return response.output_text.strip()



def run_generation_phase(client: openai.OpenAI, llm_df: pd.DataFrame) -> pd.DataFrame:
    """Generate concise answers for rows that still have placeholder values."""

    updated_df = llm_df.copy()
    total_rows = len(updated_df)
    processed = 0

    for row_index in updated_df.index:
        current_answer = normalize_text(updated_df.at[row_index, "llm_answer"])
        if current_answer != "TO_BE_GENERATED":
            continue

        prompt = build_generation_prompt(
            normalize_text(updated_df.at[row_index, "context"]),
            normalize_text(updated_df.at[row_index, "question"]),
        )
        try:
            generated_answer = request_model_text(client, prompt)
            updated_df.at[row_index, "llm_answer"] = generated_answer or "GENERATION_FAILED"
            if generated_answer:
                updated_df.at[row_index, "generated_at"] = pd.Timestamp.utcnow().isoformat()
        except Exception as exc:
            updated_df.at[row_index, "llm_answer"] = "GENERATION_FAILED"
            print(
                f"Generation failed for {normalize_text(updated_df.at[row_index, 'question_id'])}: {exc}"
            )
        processed += 1
        if processed % 10 == 0:
            print(f"Generated {processed}/{total_rows} answers...")
        time.sleep(0.3)

    return updated_df



def save_llm_responses(llm_df: pd.DataFrame, output_path: str) -> None:
    """Save the updated LLM responses dataset back to CSV."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    llm_df.to_csv(output_path, index=False)



def build_judge_prompt(question: str, reference_answer: str, llm_answer: str) -> str:
    """Build the exact judge prompt used for LLM-as-judge scoring."""

    return (
        "You are an expert QA evaluator. Evaluate the LLM answer against \n"
        "the reference answer. Return ONLY a valid JSON object, \n"
        "no other text, no markdown backticks.\n\n"
        f"Question: {question}\n"
        f"Reference Answer: {reference_answer}\n"
        f"LLM Answer: {llm_answer}\n\n"
        "Return exactly this JSON:\n"
        "{\n"
        '  "faithfulness": <integer 1-5, does LLM answer match reference facts>,\n'
        '  "relevance": <integer 1-5, does LLM answer address the question>,\n'
        '  "completeness": <integer 1-5, does LLM answer cover key information>,\n'
        '  "reasoning": <one sentence explaining your scores>\n'
        "}"
    )



def parse_judge_scores(raw_text: str) -> dict[str, object]:
    """Parse the judge model output as JSON and return normalized evaluation fields."""

    cleaned_text = raw_text.strip()
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.strip("`")
        if cleaned_text.lower().startswith("json"):
            cleaned_text = cleaned_text[4:].strip()
    try:
        parsed = json.loads(cleaned_text)
    except json.JSONDecodeError:
        normalized_text = cleaned_text.replace("'", '"')
        parsed = json.loads(normalized_text)
    return {
        "faithfulness": parsed.get("faithfulness"),
        "relevance": parsed.get("relevance"),
        "completeness": parsed.get("completeness"),
        "reasoning": parsed.get("reasoning"),
    }



def safe_score(value: object) -> object:
    """Keep only integer scores from 1 to 5; otherwise return null."""

    if value is None or pd.isna(value):
        return None
    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= numeric_value <= 5:
        return numeric_value
    return None



def evaluate_answers_with_judge(client: openai.OpenAI, llm_df: pd.DataFrame) -> pd.DataFrame:
    """Score generated answers for faithfulness, relevance, and completeness."""

    records = []
    evaluable_indices = []
    for row_index in llm_df.index:
        llm_answer = normalize_text(llm_df.at[row_index, "llm_answer"])
        if llm_answer not in {"", "GENERATION_FAILED", "TO_BE_GENERATED"}:
            evaluable_indices.append(row_index)

    processed = 0
    total_evaluable = len(evaluable_indices)

    for row_index in llm_df.index:
        question_id = normalize_text(llm_df.at[row_index, "question_id"])
        question = normalize_text(llm_df.at[row_index, "question"])
        reference_answer = normalize_text(llm_df.at[row_index, "reference_answer"])
        llm_answer = normalize_text(llm_df.at[row_index, "llm_answer"])

        record = {
            "question_id": question_id,
            "question": question,
            "reference_answer": reference_answer,
            "llm_answer": llm_answer,
            "faithfulness": None,
            "relevance": None,
            "completeness": None,
            "avg_score": None,
            "reasoning": None,
            "eval_status": "failed",
        }

        if llm_answer == "GENERATION_FAILED" or llm_answer in {"", "TO_BE_GENERATED"}:
            records.append(record)
            continue

        judge_prompt = build_judge_prompt(question, reference_answer, llm_answer)
        try:
            judge_text = request_model_text(client, judge_prompt)
            parsed_scores = parse_judge_scores(judge_text)
            record["faithfulness"] = safe_score(parsed_scores.get("faithfulness"))
            record["relevance"] = safe_score(parsed_scores.get("relevance"))
            record["completeness"] = safe_score(parsed_scores.get("completeness"))
            record["reasoning"] = parsed_scores.get("reasoning")
            if None not in (
                record["faithfulness"],
                record["relevance"],
                record["completeness"],
            ):
                record["avg_score"] = round(
                    float(
                        (
                            record["faithfulness"]
                            + record["relevance"]
                            + record["completeness"]
                        )
                        / 3
                    ),
                    4,
                )
                record["eval_status"] = "success"
            else:
                print(f"Evaluation parsing failed for {question_id}: invalid score values returned")
        except json.JSONDecodeError as exc:
            print(f"Evaluation parsing failed for {question_id}: {exc}")
        except Exception as exc:
            print(f"Evaluation failed for {question_id}: {exc}")

        records.append(record)
        processed += 1
        if processed % 10 == 0:
            print(f"Evaluated {processed}/{total_evaluable} answers...")
        time.sleep(0.3)

    return pd.DataFrame(
        records,
        columns=[
            "question_id",
            "question",
            "reference_answer",
            "llm_answer",
            "faithfulness",
            "relevance",
            "completeness",
            "avg_score",
            "reasoning",
            "eval_status",
        ],
    )



def save_eval_scores(eval_df: pd.DataFrame, output_path: str) -> None:
    """Save LLM evaluation scores to CSV."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    eval_df.to_csv(output_path, index=False)



def print_report(llm_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    """Print the final LLM evaluation summary report."""

    failed_generations = int((llm_df["llm_answer"].astype(str).str.strip() == "GENERATION_FAILED").sum())
    failed_evaluations = int(
        (
            (eval_df["eval_status"] == "failed")
            & (eval_df["llm_answer"].astype(str).str.strip() != "GENERATION_FAILED")
        ).sum()
    )
    successful_df = eval_df.loc[eval_df["eval_status"] == "success"].copy()

    faithfulness_avg = successful_df["faithfulness"].mean()
    relevance_avg = successful_df["relevance"].mean()
    completeness_avg = successful_df["completeness"].mean()
    overall_avg = successful_df["avg_score"].mean()

    high_quality = int((successful_df["avg_score"] >= 4.0).sum())
    medium_quality = int(((successful_df["avg_score"] >= 3.0) & (successful_df["avg_score"] < 4.0)).sum())
    low_quality = int((successful_df["avg_score"] < 3.0).sum())

    print("=== LLM EVALUATION REPORT ===")
    print(f"Total evaluated: {len(eval_df)}")
    print(f"Failed generations: {failed_generations}")
    print(f"Failed evaluations: {failed_evaluations}")
    print()
    print("Average Scores:")
    print(f"Faithfulness: {faithfulness_avg:.2f} / 5")
    print(f"Relevance: {relevance_avg:.2f} / 5")
    print(f"Completeness: {completeness_avg:.2f} / 5")
    print(f"Overall Average: {overall_avg:.2f} / 5")
    print()
    print("Score Distribution:")
    print(f"4.0-5.0 (High quality): {high_quality} questions")
    print(f"3.0-3.9 (Medium quality): {medium_quality} questions")
    print(f"Below 3.0 (Low quality): {low_quality} questions")
    print("==============================")



def main() -> None:
    """Run answer generation first, then evaluate the answers with an LLM judge."""

    print("Loading environment and OpenAI client...")
    client = load_environment_and_client()

    print("Loading LLM response dataset...")
    llm_df = load_llm_responses(RESPONSES_PATH)

    print("Starting Phase 1 - Generate LLM answers...")
    llm_df = run_generation_phase(client, llm_df)
    save_llm_responses(llm_df, RESPONSES_PATH)
    print(f"Saved updated LLM responses to {RESPONSES_PATH}")

    print("Starting Phase 2 - LLM-as-judge evaluation...")
    eval_df = evaluate_answers_with_judge(client, llm_df)
    save_eval_scores(eval_df, EVAL_SCORES_PATH)
    print(f"Saved LLM evaluation scores to {EVAL_SCORES_PATH}")

    print_report(llm_df, eval_df)


if __name__ == "__main__":
    main()


