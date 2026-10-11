import pandas as pd
from datasets import load_dataset

from src.retriever import Retriever
from src.generator import Generator
from src.verifier import Verifier
from src.correction import Corrector
from src.vba_scrag import VBASCRAG


NUM_QUESTIONS = 50
THRESHOLD = 0.75
MARGIN = 0.00


# =========================================================
# LOAD DATASET
# =========================================================

print("Loading HotpotQA...")

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split=f"validation[:{NUM_QUESTIONS}]"
)

questions = []
gold_answers = []
documents = []

for item in dataset:

    questions.append(item["question"])
    gold_answers.append(item["answer"])

    for title, sentences in zip(
        item["context"]["title"],
        item["context"]["sentences"]
    ):

        documents.append(
            title + ". " + " ".join(sentences)
        )

documents = list(dict.fromkeys(documents))

print("Questions:", len(questions))
print("Documents:", len(documents))


# =========================================================
# INITIALIZE
# =========================================================

retriever = Retriever(documents)
generator = Generator()
verifier = Verifier()
corrector = Corrector()


# =========================================================
# SYSTEM 1 — VANILLA RAG
# =========================================================

print("\nRunning Vanilla RAG...")

vanilla = []

for question in questions:

    evidence = retriever.retrieve(
        question,
        top_k=3
    )

    answer = generator.generate(
        question,
        evidence
    )

    grounding = verifier.calculate_grounding(
        answer,
        evidence
    )

    vanilla.append({
        "question": question,
        "answer": answer,
        "grounding": grounding,
        "iterations": 0
    })


# =========================================================
# SYSTEM 2 — SELF-CORRECTING RAG
# Automatically accepts correction
# =========================================================

print("\nRunning Self-Correcting RAG...")

self_correcting = []

for question in questions:

    evidence = retriever.retrieve(
        question,
        top_k=3
    )

    answer = generator.generate(
        question,
        evidence
    )

    initial_grounding = verifier.calculate_grounding(
        answer,
        evidence
    )

    corrected_query = corrector.correct_query(
        question,
        answer
    )

    new_evidence = retriever.retrieve(
        corrected_query,
        top_k=3
    )

    corrected_answer = generator.generate(
        question,
        new_evidence
    )

    corrected_grounding = verifier.calculate_grounding(
        corrected_answer,
        new_evidence
    )

    self_correcting.append({
        "question": question,
        "initial_grounding": initial_grounding,
        "final_grounding": corrected_grounding,
        "grounding_improvement":
            corrected_grounding - initial_grounding,
        "iterations": 1
    })


# =========================================================
# SYSTEM 3 — VBA-SCRAG
# =========================================================

print("\nRunning VBA-SCRAG...")

vba = []

system = VBASCRAG(
    retriever=retriever,
    generator=generator,
    verifier=verifier,
    corrector=corrector,
    threshold=THRESHOLD,
    improvement_margin=MARGIN,
    max_iterations=3
)

for question in questions:

    result = system.run(question)

    vba.append(result)


# =========================================================
# DATAFRAMES
# =========================================================

vanilla_df = pd.DataFrame(vanilla)

self_df = pd.DataFrame(self_correcting)

vba_df = pd.DataFrame(vba)


# =========================================================
# SUMMARY
# =========================================================

accepted = vba_df[
    "correction_accepted"
].sum()

rejected = vba_df[
    "correction_rejected"
].sum()

attempts = vba_df[
    "correction_attempted"
].sum()


comparison = pd.DataFrame({

    "System": [
        "Vanilla RAG",
        "Self-Correcting RAG",
        "VBA-SCRAG"
    ],

    "Average Initial Grounding": [
        vanilla_df["grounding"].mean(),
        self_df["initial_grounding"].mean(),
        vba_df["initial_grounding"].mean()
    ],

    "Average Final Grounding": [
        vanilla_df["grounding"].mean(),
        self_df["final_grounding"].mean(),
        vba_df["grounding"].mean()
    ],

    "Average Improvement": [
        0,
        self_df[
            "grounding_improvement"
        ].mean(),
        (
            vba_df["grounding"]
            -
            vba_df["initial_grounding"]
        ).mean()
    ],

    "Average Iterations": [
        0,
        self_df["iterations"].mean(),
        vba_df["iterations"].mean()
    ]
})


# =========================================================
# PRINT
# =========================================================

print("\n")
print("=" * 80)
print("FINAL SYSTEM COMPARISON")
print("=" * 80)

print(
    comparison.to_string(index=False)
)

print("\nVBA-SCRAG")

print("Correction Attempts:", attempts)
print("Accepted Corrections:", accepted)
print("Rejected Corrections:", rejected)

if attempts > 0:

    print(
        "Acceptance Rate:",
        round(accepted / attempts, 4)
    )

    print(
        "Rejection Rate:",
        round(rejected / attempts, 4)
    )


# =========================================================
# SAVE
# =========================================================

comparison.to_csv(
    "results/final_system_comparison.csv",
    index=False
)

vanilla_df.to_csv(
    "results/final_vanilla_rag.csv",
    index=False
)

self_df.to_csv(
    "results/final_self_correcting_rag.csv",
    index=False
)

vba_df.to_csv(
    "results/final_vba_scrag.csv",
    index=False
)

print("\nFiles saved.")