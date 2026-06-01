"""Generate synthetic QA, annotation, and LLM response datasets."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
TOPICS = ["science", "history", "technology", "health", "sports"]
DIFFICULTIES = ["easy", "medium", "hard"]
ANNOTATORS = [f"annotator_{index}" for index in range(1, 6)]
LABELS = ["accept", "reject", "needs_revision"]
CONFIDENCE_LEVELS = ["low", "medium", "high"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

TOPIC_BANK = {
    "science": [
        {
            "subject": "photosynthesis",
            "intro": "Photosynthesis is one of the most important processes in plant biology.",
            "fact": "During photosynthesis, chlorophyll absorbs sunlight so plants can produce sugars.",
            "impact": "The process supports plant growth and contributes oxygen to the atmosphere.",
            "answer": "chlorophyll",
            "question": "What pigment absorbs sunlight during photosynthesis?",
        },
        {
            "subject": "penicillin",
            "intro": "The discovery of penicillin changed the way doctors treated bacterial infections.",
            "fact": "Alexander Fleming identified penicillin after observing mold killing bacteria in a laboratory dish.",
            "impact": "Its later development made many once-dangerous infections far easier to treat.",
            "answer": "penicillin",
            "question": "What antibiotic did Fleming's observation lead to?",
        },
        {
            "subject": "plate tectonics",
            "intro": "Geologists study the movement of Earth's surface to explain natural events over long time periods.",
            "fact": "The theory of plate tectonics explains why continents shift and why many earthquakes happen near plate boundaries.",
            "impact": "It helps scientists interpret mountains, volcanoes, and ocean trenches.",
            "answer": "plate tectonics",
            "question": "What theory explains why continents shift over time?",
        },
        {
            "subject": "DNA",
            "intro": "DNA carries the instructions that cells use to build and maintain living organisms.",
            "fact": "The molecule has a double helix structure formed by two intertwined strands.",
            "impact": "Its stored information is passed from parents to offspring during reproduction.",
            "answer": "double helix",
            "question": "What structure does the DNA molecule have?",
        },
    ],
    "history": [
        {
            "subject": "Magna Carta",
            "intro": "The Magna Carta is one of the most famous legal documents in medieval English history.",
            "fact": "It was sealed in 1215 during a dispute between King John and his barons.",
            "impact": "The document later influenced ideas about constitutional government and the rule of law.",
            "answer": "1215",
            "question": "In what year was the Magna Carta sealed?",
        },
        {
            "subject": "printing press",
            "intro": "The movable-type printing press transformed how information spread across Europe.",
            "fact": "Johannes Gutenberg developed the press, making books cheaper and easier to distribute.",
            "impact": "As printed material spread, literacy and the exchange of ideas increased.",
            "answer": "Johannes Gutenberg",
            "question": "Who developed the movable-type printing press mentioned in the passage?",
        },
        {
            "subject": "Berlin Wall",
            "intro": "The Berlin Wall became a powerful symbol of Cold War division in Europe.",
            "fact": "The barrier stood from 1961 until 1989, when its fall marked a major political shift.",
            "impact": "Its collapse helped speed German reunification and changed European politics.",
            "answer": "1989",
            "question": "In what year did the Berlin Wall fall?",
        },
        {
            "subject": "Underground Railroad",
            "intro": "Harriet Tubman is remembered for her leadership in the fight against slavery.",
            "fact": "She used the Underground Railroad, a secret network of routes and safe houses, to guide enslaved people to freedom.",
            "impact": "Her repeated missions made her one of the most respected abolitionists in American history.",
            "answer": "the Underground Railroad",
            "question": "What secret network did Harriet Tubman use to guide people to freedom?",
        },
    ],
    "technology": [
        {
            "subject": "cloud computing",
            "intro": "Cloud computing changed how organizations manage software, storage, and infrastructure.",
            "fact": "It allows teams to rent computing power and storage over the internet instead of owning all of their hardware.",
            "impact": "This model helps companies scale systems more quickly and reduce on-site maintenance.",
            "answer": "computing power and storage",
            "question": "What does cloud computing allow organizations to rent over the internet?",
        },
        {
            "subject": "encryption",
            "intro": "Encryption is a core tool for protecting digital communication and stored information.",
            "fact": "Only someone with the proper key can turn encrypted text back into its original readable form.",
            "impact": "Banks, messaging apps, and websites rely on this protection for sensitive data.",
            "answer": "the proper key",
            "question": "What is needed to decode encrypted data?",
        },
        {
            "subject": "machine learning",
            "intro": "Machine learning systems improve at tasks by learning from examples.",
            "fact": "They identify patterns in data and use those patterns to make predictions or classifications.",
            "impact": "Performance depends heavily on training data quality and evaluation practices.",
            "answer": "patterns",
            "question": "What do machine learning systems identify in data to improve at a task?",
        },
        {
            "subject": "APIs",
            "intro": "APIs help software teams connect applications without rebuilding the same functionality each time.",
            "fact": "An API allows systems to exchange data and functions in a structured and documented way.",
            "impact": "This makes services easier to integrate, extend, and maintain over time.",
            "answer": "data and functions",
            "question": "What does an API allow different software systems to exchange?",
        },
    ],
    "health": [
        {
            "subject": "aerobic exercise",
            "intro": "Regular aerobic exercise is widely recommended as part of a healthy lifestyle.",
            "fact": "Activities such as brisk walking, cycling, and swimming can lower the risk of cardiovascular disease.",
            "impact": "Consistent exercise also improves circulation and supports long-term heart health.",
            "answer": "cardiovascular disease",
            "question": "What kind of disease can regular aerobic exercise help lower the risk of?",
        },
        {
            "subject": "sleep",
            "intro": "Sleep is essential for both physical recovery and mental performance.",
            "fact": "During healthy sleep, the brain consolidates memories and supports attention for the next day.",
            "impact": "Too little sleep can reduce concentration and slow reaction times.",
            "answer": "memories",
            "question": "What mental process does sleep help the brain consolidate?",
        },
        {
            "subject": "vitamin D",
            "intro": "Vitamin D plays an important role in bone health and overall wellness.",
            "fact": "It helps the body absorb calcium, which is needed to build and maintain strong bones.",
            "impact": "People can get vitamin D from sunlight, certain foods, and supplements.",
            "answer": "calcium",
            "question": "What mineral does vitamin D help the body absorb?",
        },
        {
            "subject": "handwashing",
            "intro": "Hand hygiene is one of the simplest ways to reduce the spread of illness.",
            "fact": "Washing hands with soap removes dirt, oils, and many harmful microbes from the skin.",
            "impact": "Public health experts recommend this habit before eating and after using the restroom.",
            "answer": "soap",
            "question": "What should people use when washing hands to remove harmful microbes?",
        },
    ],
    "sports": [
        {
            "subject": "soccer",
            "intro": "Soccer is played under a simple structure, but small tactical changes can shape an entire match.",
            "fact": "A standard match is divided into two halves of forty-five minutes each.",
            "impact": "Teams often adjust strategy at halftime depending on the score and player fitness.",
            "answer": "forty-five minutes",
            "question": "How many minutes is each half of a standard soccer match?",
        },
        {
            "subject": "marathon",
            "intro": "A marathon is one of the best-known endurance events in athletics.",
            "fact": "The official race distance is 42.195 kilometers, which tests pacing, hydration, and focus.",
            "impact": "Runners often train for months to prepare for the physical demands of the event.",
            "answer": "42.195 kilometers",
            "question": "What distance does a marathon cover?",
        },
        {
            "subject": "tennis tiebreak",
            "intro": "Tiebreaks are high-pressure moments that can decide an entire tennis set.",
            "fact": "A player must win a tiebreak by two points once both sides reach six points.",
            "impact": "Serving accuracy and composure matter even more during these short sequences.",
            "answer": "two points",
            "question": "By how many points must a tennis player win a tiebreak?",
        },
        {
            "subject": "relay race",
            "intro": "Relay races combine individual speed with coordinated teamwork.",
            "fact": "Each runner completes part of the course and passes a baton to the next teammate.",
            "impact": "Clean exchanges are crucial because slow handoffs waste valuable time.",
            "answer": "a baton",
            "question": "What item is passed between teammates in a relay race?",
        },
    ],
}

TOPIC_FILLERS = {
    "science": [
        "Researchers continue to test these ideas with measurements and experiments.",
        "This topic is often used in classrooms to explain how evidence supports a scientific claim.",
        "Small details in the process can affect how scientists interpret the results.",
    ],
    "history": [
        "Historians compare written records and artifacts to understand the period more clearly.",
        "The event is still discussed because its effects lasted well beyond its original moment.",
        "Public memory of the topic has been shaped by books, museums, and education.",
    ],
    "technology": [
        "Engineers often balance speed, reliability, security, and cost when applying this idea.",
        "Real-world adoption depends on both technical design and user needs.",
        "Organizations usually combine the approach with careful testing and documentation.",
    ],
    "health": [
        "Doctors and public health teams often emphasize prevention alongside treatment.",
        "Daily habits can make a measurable difference when practiced consistently over time.",
        "Clear communication helps people apply this guidance in ordinary routines.",
    ],
    "sports": [
        "Coaches often focus on discipline and timing because small mistakes can change the result.",
        "Fans usually notice the final score, but preparation and technique matter just as much.",
        "Athletes train repeatedly so that key movements feel automatic under pressure.",
    ],
}

ANNOTATOR_BIASES = {
    "annotator_1": {"quality": 0.1, "clarity": 0.0},
    "annotator_2": {"quality": 0.3, "clarity": 0.2},
    "annotator_3": {"quality": -0.2, "clarity": -0.1},
    "annotator_4": {"quality": 0.0, "clarity": 0.1},
    "annotator_5": {"quality": -0.1, "clarity": 0.2},
}


def set_random_seed(seed: int = SEED) -> None:
    """Set all random seeds used in this script for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)



def lower_first(text: str) -> str:
    """Lowercase the first character of a string for question templating."""

    return text[:1].lower() + text[1:] if text else text



def build_question(base_question: str, subject: str, difficulty: str) -> str:
    """Create a question variant that reflects the chosen difficulty."""

    if difficulty == "easy":
        return base_question
    if difficulty == "medium":
        return f"According to the passage, {lower_first(base_question)}"
    return f"Based on the context about {subject}, {lower_first(base_question)}"



def choose_topic() -> str:
    """Select a topic for a synthetic QA record."""

    return random.choice(TOPICS)



def choose_difficulty() -> str:
    """Select a difficulty level for a synthetic QA record."""

    return random.choices(DIFFICULTIES, weights=[0.4, 0.4, 0.2], k=1)[0]



def build_context(topic: str, difficulty: str) -> tuple[str, str, str]:
    """Build a 3-5 sentence context paragraph plus its question and answer."""

    entry = random.choice(TOPIC_BANK[topic])
    extra_count = {"easy": 0, "medium": 1, "hard": 2}[difficulty]
    extras = random.sample(TOPIC_FILLERS[topic], k=extra_count)
    sentences = [entry["intro"], entry["fact"], entry["impact"], *extras]
    context = " ".join(sentences)
    question = build_question(entry["question"], entry["subject"], difficulty)
    return context, question, entry["answer"]



def generate_raw_qa_data(num_rows: int = 500) -> pd.DataFrame:
    """Generate the synthetic raw QA dataset in SQuAD-style format."""

    records = []
    for index in range(1, num_rows + 1):
        topic = choose_topic()
        difficulty = choose_difficulty()
        context, question, answer = build_context(topic, difficulty)
        records.append(
            {
                "question_id": f"QA_{index:04d}",
                "context": context,
                "question": question,
                "reference_answer": answer,
                "topic": topic,
                "difficulty": difficulty,
            }
        )
    return pd.DataFrame(records)



def choose_consensus_label(difficulty: str) -> str:
    """Choose a likely consensus label based on item difficulty."""

    weights = {
        "easy": [0.75, 0.05, 0.20],
        "medium": [0.52, 0.12, 0.36],
        "hard": [0.25, 0.28, 0.47],
    }
    return random.choices(LABELS, weights=weights[difficulty], k=1)[0]



def assign_conflict_profiles(num_questions: int) -> list[str]:
    """Assign exact conflict-profile ratios across all questions."""

    profiles = ["mostly_agree"] * int(num_questions * 0.60)
    profiles += ["moderate_conflict"] * int(num_questions * 0.30)
    profiles += ["high_conflict"] * (num_questions - len(profiles))
    random.shuffle(profiles)
    return profiles



def sample_other_label(primary_label: str, excluded: set[str] | None = None) -> str:
    """Sample a label that differs from the primary label."""

    excluded = excluded or set()
    options = [label for label in LABELS if label != primary_label and label not in excluded]
    return random.choice(options)



def build_votes(consensus_label: str, profile: str) -> list[str]:
    """Create five labels that satisfy the requested disagreement profile."""

    if profile == "mostly_agree":
        votes = [consensus_label] * 5 if random.random() < 0.65 else [consensus_label] * 4 + [sample_other_label(consensus_label)]
    elif profile == "moderate_conflict":
        if random.random() < 0.7:
            alt = sample_other_label(consensus_label)
            votes = [consensus_label] * 3 + [alt] * 2
        else:
            alt_one = sample_other_label(consensus_label)
            alt_two = sample_other_label(consensus_label, excluded={alt_one})
            votes = [consensus_label] * 3 + [alt_one, alt_two]
    else:
        alt_one = sample_other_label(consensus_label)
        alt_two = sample_other_label(consensus_label, excluded={alt_one})
        votes = [consensus_label] * 2 + [alt_one] * 2 + [alt_two]
    random.shuffle(votes)
    return votes



def base_scores(label: str, difficulty: str) -> tuple[float, float]:
    """Return baseline answer-quality and question-clarity scores."""

    quality_map = {
        "easy": {"accept": 4.5, "needs_revision": 3.4, "reject": 2.3},
        "medium": {"accept": 4.0, "needs_revision": 3.2, "reject": 2.1},
        "hard": {"accept": 3.5, "needs_revision": 2.9, "reject": 1.9},
    }
    clarity_map = {
        "easy": {"accept": 4.4, "needs_revision": 3.5, "reject": 2.5},
        "medium": {"accept": 3.9, "needs_revision": 3.1, "reject": 2.1},
        "hard": {"accept": 3.3, "needs_revision": 2.8, "reject": 1.9},
    }
    return quality_map[difficulty][label], clarity_map[difficulty][label]



def clamp_score(value: float) -> int:
    """Round and clamp a score into the inclusive 1-5 range."""

    return int(max(1, min(5, round(value))))



def choose_confidence(label: str, consensus_label: str, difficulty: str, profile: str) -> str:
    """Choose a confidence band using difficulty and disagreement level."""

    if label != consensus_label:
        weights = [0.35, 0.50, 0.15] if profile != "high_conflict" else [0.28, 0.52, 0.20]
    elif difficulty == "easy":
        weights = [0.05, 0.35, 0.60]
    elif difficulty == "medium":
        weights = [0.10, 0.50, 0.40]
    else:
        weights = [0.18, 0.52, 0.30]
    return random.choices(CONFIDENCE_LEVELS, weights=weights, k=1)[0]



def random_timestamp(start: datetime, end: datetime) -> str:
    """Generate a random timestamp string between two datetimes."""

    offset_seconds = random.randint(0, int((end - start).total_seconds()))
    return (start + timedelta(seconds=offset_seconds)).strftime("%Y-%m-%d %H:%M:%S")



def generate_annotations(raw_qa_df: pd.DataFrame) -> pd.DataFrame:
    """Generate five annotator records per question with realistic disagreement."""

    profiles = assign_conflict_profiles(len(raw_qa_df))
    rows = []
    start = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 12, 31, 23, 59, 59)

    for qa_row, profile in zip(raw_qa_df.itertuples(index=False), profiles):
        consensus_label = choose_consensus_label(qa_row.difficulty)
        votes = build_votes(consensus_label, profile)
        for annotator_id, label in zip(ANNOTATORS, votes):
            quality_base, clarity_base = base_scores(label, qa_row.difficulty)
            bias = ANNOTATOR_BIASES[annotator_id]
            rows.append(
                {
                    "question_id": qa_row.question_id,
                    "annotator_id": annotator_id,
                    "answer_quality": clamp_score(np.random.normal(quality_base + bias["quality"], 0.45)),
                    "question_clarity": clamp_score(np.random.normal(clarity_base + bias["clarity"], 0.45)),
                    "label": label,
                    "confidence": choose_confidence(label, consensus_label, qa_row.difficulty, profile),
                    "timestamp": random_timestamp(start, end),
                }
            )
    return pd.DataFrame(rows)



def generate_llm_responses(raw_qa_df: pd.DataFrame, num_rows: int = 100) -> pd.DataFrame:
    """Create the placeholder LLM response dataset from the first QA rows."""

    llm_df = raw_qa_df.head(num_rows).copy()
    llm_df["llm_answer"] = "TO_BE_GENERATED"
    llm_df["model_used"] = "gpt-4o-mini"
    llm_df["generated_at"] = ""
    return llm_df[["question_id", "question", "context", "reference_answer", "llm_answer", "model_used", "generated_at"]]



def save_dataframe(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Save a dataframe to CSV, creating parent directories when needed."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)



def calculate_conflict_rate(annotations_df: pd.DataFrame) -> float:
    """Estimate question-level conflict as the share with more than one label."""

    return float((annotations_df.groupby("question_id")["label"].nunique() > 1).mean())



def print_summary(raw_qa_df: pd.DataFrame, annotations_df: pd.DataFrame, llm_df: pd.DataFrame) -> None:
    """Print dataset row counts, label distribution, and conflict estimate."""

    print("\nSynthetic data generation summary")
    print(f"raw_qa.csv rows: {len(raw_qa_df)}")
    print(f"annotations.csv rows: {len(annotations_df)}")
    print(f"llm_responses.csv rows: {len(llm_df)}")
    print("\nLabel distribution:")
    for label, count in annotations_df["label"].value_counts().sort_index().items():
        print(f"  {label}: {count}")
    print(f"\nConflict rate estimate: {calculate_conflict_rate(annotations_df):.2%} of questions have label disagreement")



def main() -> None:
    """Generate all synthetic datasets, save them to disk, and print a summary."""

    set_random_seed()
    print("Starting synthetic annotation data generation...")
    print("Generating raw QA dataset...")
    raw_qa_df = generate_raw_qa_data(500)
    save_dataframe(raw_qa_df, DATA_DIR / "raw_qa.csv")
    print(f"Saved raw QA dataset to {DATA_DIR / 'raw_qa.csv'}")

    print("Generating annotator labels with realistic disagreement...")
    annotations_df = generate_annotations(raw_qa_df)
    save_dataframe(annotations_df, DATA_DIR / "annotations.csv")
    print(f"Saved annotation dataset to {DATA_DIR / 'annotations.csv'}")

    print("Generating placeholder LLM response dataset...")
    llm_df = generate_llm_responses(raw_qa_df, 100)
    save_dataframe(llm_df, DATA_DIR / "llm_responses.csv")
    print(f"Saved LLM response dataset to {DATA_DIR / 'llm_responses.csv'}")

    print_summary(raw_qa_df, annotations_df, llm_df)


if __name__ == "__main__":
    main()
