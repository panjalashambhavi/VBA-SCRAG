
import pandas as pd
import numpy as np

files = {
    "Vanilla RAG": "results/fixed_vanilla_rag_correctness_semantic.csv",
    "Correction without gate": "results/fixed_self_correcting_rag_correctness_semantic.csv",
    "VBA-SCRAG": "results/fixed_vba_scrag_correctness_semantic.csv",
}

data = {name: pd.read_csv(path) for name, path in files.items()}

# Align rows by question before comparing systems
questions = set.intersection(
    *(set(df["question"]) for df in data.values())
)

print("Shared questions:", len(questions))

metrics = [
    ("semantic_similarity", "Semantic similarity"),
    ("answer_f1", "Answer F1"),
    ("exact_match", "Exact Match"),
]

rng = np.random.default_rng(42)
n_bootstrap = 10000

for column, label in metrics:
    print(f"\n--- {label} ---")

    aligned = {}
    for name, df in data.items():
        indexed = df.drop_duplicates("question").set_index("question")
        aligned[name] = indexed.loc[sorted(questions), column].astype(float)

    baseline = aligned["Correction without gate"]

    for other_name in ["Vanilla RAG", "VBA-SCRAG"]:
        differences = (
            baseline.to_numpy() - aligned[other_name].to_numpy()
        )

        observed = differences.mean()
        samples = rng.choice(
            differences,
            size=(n_bootstrap, len(differences)),
            replace=True
        ).mean(axis=1)

        low, high = np.percentile(samples, [2.5, 97.5])

        print(
            f"Correction without gate - {other_name}: "
            f"{observed:.4f}; bootstrap 95% CI "
            f"[{low:.4f}, {high:.4f}]"
        )
