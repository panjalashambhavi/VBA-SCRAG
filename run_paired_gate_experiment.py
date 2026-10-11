
import os
import re
import string
import pandas as pd
import numpy as np
from datasets import load_dataset

from src.retriever import Retriever
from src.generator import Generator
from src.verifier import Verifier
from src.correction import Corrector

# Experiment settings
NUM_QUESTIONS = 50
THRESHOLD = 0.75
MARGIN = 0.001
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_answer(text):
    """Normalize answers for simple Exact Match and token F1."""
    text = str(text).lower()
    text = "".join(
        ch for ch in text if ch not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def answer_f1(prediction, reference):
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1

    overlap = 0
    for token in ref_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

print("Loading HotpotQA...")
dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split=f"validation[:{NUM_QUESTIONS}]"
)

questions = []
references = []
documents = []

for item in dataset:
    questions.append(item["question"])
    references.append(item["answer"])

    for title, sentences in zip(
        item["context"]["title"],
        item["context"]["sentences"]
    ):
        documents.append(title + ". " + " ".join(sentences))

documents = list(dict.fromkeys(documents))

print("Questions:", len(questions))
print("Unique documents:", len(documents))
print("Loading models...")

retriever = Retriever(documents)
generator = Generator()
verifier = Verifier()
corrector = Corrector()

paired_results = []

for i, (question, reference) in enumerate(
    zip(questions, references), start=1
):
    # 1. Generate the initial answer ONCE.
    initial_evidence = retriever.retrieve(question, top_k=3)
    initial_answer = generator.generate(question, initial_evidence)
    initial_score = verifier.calculate_grounding(
        initial_answer, initial_evidence
    )

    # Defaults: both systems retain the initial answer.
    candidate_answer = None
    candidate_score = None
    attempted = initial_score < THRESHOLD

    if attempted:
        # 2. Generate the correction candidate ONCE.
        corrected_query = corrector.correct_query(
            question, initial_answer
        )
        candidate_evidence = retriever.retrieve(
            corrected_query, top_k=3
        )
        candidate_answer = generator.generate(
            question, candidate_evidence
        )
        candidate_score = verifier.calculate_grounding(
            candidate_answer, candidate_evidence
        )

    # 3. Apply two different decisions to the SAME candidate.
    if attempted:
        improvement = candidate_score - initial_score

        # Ungated system accepts every triggered candidate.
        ungated_answer = candidate_answer
        ungated_score = candidate_score

        # Gated system accepts only a sufficient score improvement.
        accepted = improvement >= MARGIN

        if accepted:
            gated_answer = candidate_answer
            gated_score = candidate_score
        else:
            gated_answer = initial_answer
            gated_score = initial_score
    else:
        improvement = 0.0
        accepted = False
        ungated_answer = initial_answer
        ungated_score = initial_score
        gated_answer = initial_answer
        gated_score = initial_score

    paired_results.append({
        "question": question,
        "reference": reference,
        "initial_answer": initial_answer,
        "initial_grounding": initial_score,
        "candidate_answer": candidate_answer,
        "candidate_grounding": candidate_score,
        "candidate_improvement": improvement,
        "correction_attempted": attempted,
        "gate_accepted": accepted,
        "ungated_answer": ungated_answer,
        "ungated_grounding": ungated_score,
        "gated_answer": gated_answer,
        "gated_grounding": gated_score,
        "ungated_f1": answer_f1(ungated_answer, reference),
        "gated_f1": answer_f1(gated_answer, reference),
        "ungated_exact_match": float(
            normalize_answer(ungated_answer)
            == normalize_answer(reference)
        ),
        "gated_exact_match": float(
            normalize_answer(gated_answer)
            == normalize_answer(reference)
        ),
    })

    print(
        f"[{i}/{len(questions)}] "
        f"initial={initial_score:.4f} | "
        f"candidate={candidate_score if candidate_score is not None else 'N/A'} | "
        f"accepted={accepted}"
    )

df = pd.DataFrame(paired_results)
output_path = os.path.join(
    OUTPUT_DIR, "paired_gate_experiment.csv"
)
df.to_csv(output_path, index=False)

print("\nPAIRED EXPERIMENT SUMMARY")
print("Questions:", len(df))
print("Correction attempts:", int(df["correction_attempted"].sum()))
print("Gate accepted:", int(df["gate_accepted"].sum()))
print(
    "Gate rejected:",
    int(
        (
            df["correction_attempted"]
            & ~df["gate_accepted"]
        ).sum()
    )
)

summary = pd.DataFrame([
    {
        "System": "Correction without gate",
        "Mean grounding": df["ungated_grounding"].mean(),
        "Answer F1 (%)": 100 * df["ungated_f1"].mean(),
        "Exact Match (%)": 100 * df["ungated_exact_match"].mean(),
    },
    {
        "System": "VBA-SCRAG gate",
        "Mean grounding": df["gated_grounding"].mean(),
        "Answer F1 (%)": 100 * df["gated_f1"].mean(),
        "Exact Match (%)": 100 * df["gated_exact_match"].mean(),
    },
])

print("\n", summary.to_string(index=False))
print("\nSaved paired question-level results to:", output_path)

# Direct paired differences: gated minus ungated.
print("\nPAIRED DIFFERENCES (gate minus no gate)")
print(
    "Mean grounding difference:",
    round(
        (df["gated_grounding"] - df["ungated_grounding"]).mean(),
        6
    )
)
print(
    "Mean Answer F1 difference (percentage points):",
    round(
        100 * (df["gated_f1"] - df["ungated_f1"]).mean(),
        4
    )
)
print(
    "Exact Match difference (percentage points):",
    round(
        100 * (
            df["gated_exact_match"]
            - df["ungated_exact_match"]
        ).mean(),
        4
    )
)
