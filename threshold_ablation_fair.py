import os
import pandas as pd
from datasets import load_dataset

from src.retriever import Retriever
from src.generator import Generator
from src.verifier import Verifier
from src.correction import Corrector
from src.vba_scrag_fixed import VBASCRAG

NUM_QUESTIONS = 50
THRESHOLDS = [0.60, 0.70, 0.75, 0.80, 0.90]
MARGIN = 0.001

print("Loading HotpotQA...")
dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split=f"validation[:{NUM_QUESTIONS}]",
)

questions = []
documents = []

for item in dataset:
    questions.append(item["question"])

    for title, sentences in zip(
        item["context"]["title"],
        item["context"]["sentences"],
    ):
        documents.append(title + ". " + " ".join(sentences))

documents = list(dict.fromkeys(documents))

print(f"Questions: {len(questions)}")
print(f"Unique documents: {len(documents)}")

retriever = Retriever(documents)
generator = Generator()
verifier = Verifier()
corrector = Corrector()

os.makedirs("results", exist_ok=True)
summary_rows = []

for threshold in THRESHOLDS:
    print(f"\nRunning threshold = {threshold:.2f}")

    system = VBASCRAG(
        retriever,
        generator,
        verifier,
        corrector,
        threshold=threshold,
        improvement_margin=MARGIN,
    )

    result_rows = []

    for i, question in enumerate(questions, start=1):
        result = system.run(question)
        result_rows.append(result)

        print(
            f"[{i}/{len(questions)}] "
            f"attempted={result['correction_attempted']} "
            f"accepted={result['correction_accepted']} "
            f"rejected={result['correction_rejected']}"
        )

    df = pd.DataFrame(result_rows)

    attempts = int(df["correction_attempted"].sum())
    accepted = int(df["correction_accepted"].sum())
    rejected = int(df["correction_rejected"].sum())

    summary_rows.append({
        "Threshold": threshold,
        "Questions": len(df),
        "Average Initial Grounding": df["initial_grounding"].mean(),
        "Average Final Grounding": df["grounding"].mean(),
        "Average Grounding Improvement": df["grounding_improvement"].mean(),
        "Average Iterations": df["iterations"].mean(),
        "Correction Attempts": attempts,
        "Accepted Corrections": accepted,
        "Rejected Corrections": rejected,
        "Acceptance Rate (%)": (
            100 * accepted / attempts if attempts else 0.0
        ),
        "Rejection Rate (%)": (
            100 * rejected / attempts if attempts else 0.0
        ),
    })

    df.to_csv(
        f"results/fixed_threshold_{threshold:.2f}_details.csv",
        index=False,
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv("results/fixed_threshold_ablation.csv", index=False)

print("\nTHRESHOLD ABLATION RESULTS")
print(summary.round(6).to_string(index=False))
print("\nSaved results/fixed_threshold_ablation.csv")