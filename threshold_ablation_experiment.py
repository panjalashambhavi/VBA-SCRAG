import os
import pandas as pd
from datasets import load_dataset

from src.retriever import Retriever
from src.generator import Generator
from src.verifier import Verifier
from src.correction import Corrector
from src.vba_scrag import VBASCRAG


# ============================================================
# THRESHOLD ABLATION CONFIGURATION
# ============================================================

NUM_QUESTIONS = 50

# Only tau is varied in this experiment.
THRESHOLDS = [0.60, 0.70, 0.75, 0.80, 0.90]

# Keep the remaining parameters identical to the main experiment.
MARGIN = 0.00
MAX_ITERATIONS = 3


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading HotpotQA...")

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split=f"validation[:{NUM_QUESTIONS}]"
)

questions = []
documents = []

for item in dataset:

    questions.append(item["question"])

    for title, sentences in zip(
        item["context"]["title"],
        item["context"]["sentences"]
    ):
        documents.append(
            title + ". " + " ".join(sentences)
        )

# Remove duplicate documents.
documents = list(dict.fromkeys(documents))

print("Questions:", len(questions))
print("Documents:", len(documents))


# ============================================================
# INITIALIZE SHARED COMPONENTS
# ============================================================

print("\nInitializing retriever, generator, verifier, and corrector...")

retriever = Retriever(documents)
generator = Generator()
verifier = Verifier()
corrector = Corrector()


# ============================================================
# RUN THRESHOLD ABLATION
# ============================================================

summary_results = []

for threshold in THRESHOLDS:

    print("\n" + "=" * 80)
    print(f"RUNNING THRESHOLD = {threshold}")
    print("=" * 80)

    system = VBASCRAG(
        retriever=retriever,
        generator=generator,
        verifier=verifier,
        corrector=corrector,
        threshold=threshold,
        improvement_margin=MARGIN,
        max_iterations=MAX_ITERATIONS
    )

    results = []

    for i, question in enumerate(questions, start=1):
        result = system.run(question)
        results.append(result)

        print(
            f"[{i:02d}/{NUM_QUESTIONS}] "
            f"initial={result['initial_grounding']:.6f} | "
            f"final={result['grounding']:.6f} | "
            f"improvement={result['grounding_improvement']:.6f} | "
            f"iterations={result['iterations']}"
        )

    df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Derive correction behavior from the actual results.
    #
    # A correction was attempted if iterations > 0.
    # If the final grounding improved (or stayed equal when
    # epsilon = 0), the candidate was accepted.
    # Otherwise it was rejected.
    # --------------------------------------------------------

    attempts = int(
        (df["iterations"] > 0).sum()
    )

    accepted = int(
        (
            (df["iterations"] > 0)
            &
            (df["grounding_improvement"] >= MARGIN)
        ).sum()
    )

    rejected = attempts - accepted

    acceptance_rate = (
        accepted / attempts * 100
        if attempts > 0 else 0.0
    )

    rejection_rate = (
        rejected / attempts * 100
        if attempts > 0 else 0.0
    )

    summary = {
        "Threshold": threshold,
        "Questions": len(df),
        "Average Initial Grounding":
            df["initial_grounding"].mean(),
        "Average Final Grounding":
            df["grounding"].mean(),
        "Average Grounding Improvement":
            df["grounding_improvement"].mean(),
        "Average Iterations":
            df["iterations"].mean(),
        "Correction Attempts":
            attempts,
        "Accepted Corrections":
            accepted,
        "Rejected Corrections":
            rejected,
        "Acceptance Rate (%)":
            acceptance_rate,
        "Rejection Rate (%)":
            rejection_rate
    }

    summary_results.append(summary)

    print("\nSummary for threshold", threshold)

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs("results", exist_ok=True)

ablation_df = pd.DataFrame(summary_results)

output_file = "results/threshold_ablation.csv"

ablation_df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 80)
print("FINAL THRESHOLD ABLATION RESULTS")
print("=" * 80)

print(
    ablation_df.to_string(index=False)
)

print("\nSaved to:")
print(output_file)
