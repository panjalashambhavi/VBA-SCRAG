import os
import pandas as pd
from datasets import load_dataset

from src.retriever import Retriever
from src.generator import Generator
from src.verifier import Verifier
from src.correction import Corrector
from src.vba_scrag_fixed import VBASCRAG

NUM_QUESTIONS = 50
THRESHOLD = 0.75
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

vanilla_rows = []
ungated_rows = []
gated_rows = []

system = VBASCRAG(
    retriever,
    generator,
    verifier,
    corrector,
    threshold=THRESHOLD,
    improvement_margin=MARGIN,
)

for i, question in enumerate(questions, start=1):
    # Shared initial answer and grounding score.
    initial_evidence = retriever.retrieve(question, top_k=3)
    initial_answer = generator.generate(question, initial_evidence)
    initial_score = float(
        verifier.calculate_grounding(initial_answer, initial_evidence)
    )

    vanilla_rows.append({
        "question": question,
        "answer": initial_answer,
        "initial_grounding": initial_score,
        "grounding": initial_score,
        "grounding_improvement": 0.0,
        "iterations": 0,
        "correction_attempted": False,
        "correction_accepted": False,
        "correction_rejected": False,
    })

    # Both correction systems use the same trigger and candidate.
    if initial_score < THRESHOLD:
        corrected_query = corrector.correct_query(
            question, initial_answer
        )
        corrected_evidence = retriever.retrieve(
            corrected_query, top_k=3
        )
        candidate_answer = generator.generate(
            question, corrected_evidence
        )
        candidate_score = float(
            verifier.calculate_grounding(
                candidate_answer, corrected_evidence
            )
        )

        improvement = candidate_score - initial_score

        # No-gate baseline accepts the candidate unconditionally.
        ungated_rows.append({
            "question": question,
            "answer": candidate_answer,
            "initial_grounding": initial_score,
            "grounding": candidate_score,
            "grounding_improvement": improvement,
            "candidate_grounding": candidate_score,
            "iterations": 1,
            "correction_attempted": True,
            "correction_accepted": True,
            "correction_rejected": False,
        })
    else:
        # No correction is triggered for this question.
        ungated_rows.append({
            "question": question,
            "answer": initial_answer,
            "initial_grounding": initial_score,
            "grounding": initial_score,
            "grounding_improvement": 0.0,
            "candidate_grounding": None,
            "iterations": 0,
            "correction_attempted": False,
            "correction_accepted": False,
            "correction_rejected": False,
        })

    # Verify-before-accept system.
    gated_result = system.run(question)
    gated_rows.append(gated_result)

    print(
        f"[{i}/{len(questions)}] "
        f"initial={initial_score:.4f} | "
        f"no_gate={ungated_rows[-1]['grounding']:.4f} | "
        f"VBA={gated_result['grounding']:.4f}"
    )

vanilla_df = pd.DataFrame(vanilla_rows)
ungated_df = pd.DataFrame(ungated_rows)
gated_df = pd.DataFrame(gated_rows)

os.makedirs("results", exist_ok=True)

vanilla_df.to_csv("results/fixed_vanilla_rag.csv", index=False)
ungated_df.to_csv("results/fixed_self_correcting_rag.csv", index=False)
gated_df.to_csv("results/fixed_vba_scrag.csv", index=False)


def summarize(name, df):
    attempted = int(df["correction_attempted"].sum())
    accepted = int(df["correction_accepted"].sum())
    rejected = int(df["correction_rejected"].sum())

    return {
        "System": name,
        "Questions": len(df),
        "Average Initial Grounding": df["initial_grounding"].mean(),
        "Average Final Grounding": df["grounding"].mean(),
        "Average Improvement": df["grounding_improvement"].mean(),
        "Correction Questions": attempted,
        "Accepted": accepted,
        "Rejected": rejected,
        "Average Iterations": df["iterations"].mean(),
    }


comparison = pd.DataFrame([
    summarize("Vanilla RAG", vanilla_df),
    summarize("Correction without gate", ungated_df),
    summarize("VBA-SCRAG", gated_df),
])

comparison.to_csv("results/fixed_system_comparison.csv", index=False)

print("\nFINAL COMPARISON")
print(comparison.round(6).to_string(index=False))
print("\nSaved CSV files in the results folder.")
print(
    "Reminder: grounding is semantic similarity to evidence, "
    "not a direct measure of factual correctness."
)