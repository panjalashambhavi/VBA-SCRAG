
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

RESULTS_DIR = "results"

files = {
    "Vanilla RAG": "fixed_vanilla_rag_correctness.csv",
    "Correction without gate": "fixed_self_correcting_rag_correctness.csv",
    "VBA-SCRAG": "fixed_vba_scrag_correctness.csv",
}

print("Loading sentence embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

summary = []

for system, filename in files.items():
    path = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(path):
        print(f"Missing file: {path}")
        continue

    df = pd.read_csv(path)

    required = {"answer", "reference"}
    if not required.issubset(df.columns):
        print(f"Missing answer/reference columns in {filename}")
        continue

    answers = df["answer"].fillna("").astype(str).tolist()
    references = df["reference"].fillna("").astype(str).tolist()

    answer_embeddings = model.encode(
        answers, convert_to_numpy=True
    )
    reference_embeddings = model.encode(
        references, convert_to_numpy=True
    )

    scores = [
        cosine_similarity(
            answer_embeddings[i].reshape(1, -1),
            reference_embeddings[i].reshape(1, -1)
        )[0][0]
        for i in range(len(answers))
    ]

    df["semantic_similarity"] = scores

    output_path = os.path.join(
        RESULTS_DIR,
        filename.replace(".csv", "_semantic.csv")
    )
    df.to_csv(output_path, index=False)

    summary.append({
        "System": system,
        "Questions": len(df),
        "Mean Semantic Similarity": sum(scores) / len(scores)
            if scores else 0,
        "Answer F1 (%)": (
            df["answer_f1"].mean() * 100
            if "answer_f1" in df.columns else None
        ),
        "Exact Match (%)": (
            df["exact_match"].mean() * 100
            if "exact_match" in df.columns else None
        ),
    })

    print(f"Saved: {output_path}")

print("\nSEMANTIC EVALUATION RESULTS")
print(pd.DataFrame(summary).to_string(index=False))
