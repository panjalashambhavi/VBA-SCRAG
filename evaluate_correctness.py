
import re
import string
import pandas as pd
from datasets import load_dataset

# Load the same 50 HotpotQA validation questions
dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split="validation[:50]",
)

references = {
    row["question"]: row["answer"]
    for row in dataset
}

def normalize(text):
    text = str(text).lower()
    text = "".join(
        ch for ch in text if ch not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def exact_match(prediction, reference):
    return float(normalize(prediction) == normalize(reference))

def token_f1(prediction, reference):
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()

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

files = {
    "Vanilla RAG": "results/fixed_vanilla_rag.csv",
    "Correction without gate": (
        "results/fixed_self_correcting_rag.csv"
    ),
    "VBA-SCRAG": "results/fixed_vba_scrag.csv",
}

summary = []
scored_data = {}

for system_name, path in files.items():
    df = pd.read_csv(path)

    if "answer" not in df.columns:
        raise ValueError(
            f"'answer' column missing in {path}. "
            "Check that this is the latest experiment output."
        )

    df["reference"] = df["question"].map(references)

    if df["reference"].isna().any():
        raise ValueError(
            f"Some questions in {path} did not match "
            "the HotpotQA reference dataset."
        )

    df["exact_match"] = df.apply(
        lambda row: exact_match(
            row["answer"], row["reference"]
        ),
        axis=1,
    )

    df["answer_f1"] = df.apply(
        lambda row: token_f1(
            row["answer"], row["reference"]
        ),
        axis=1,
    )

    scored_data[system_name] = df

    summary.append({
        "System": system_name,
        "Questions": len(df),
        "Exact Match (%)": 100 * df["exact_match"].mean(),
        "Answer F1 (%)": 100 * df["answer_f1"].mean(),
        "Mean Grounding": df["grounding"].mean(),
    })

    df.to_csv(
        path.replace(".csv", "_correctness.csv"),
        index=False,
    )

result = pd.DataFrame(summary)

print("\nINDEPENDENT ANSWER EVALUATION")
print(result.round(3).to_string(index=False))

print("\nPAIRED ANSWER F1 DIFFERENCES (percentage points)")

vanilla = scored_data["Vanilla RAG"].set_index("question")
vba = scored_data["VBA-SCRAG"].set_index("question")
ungated = scored_data["Correction without gate"].set_index("question")

print(
    "VBA-SCRAG minus Vanilla RAG:",
    round(
        100 * (
            vba["answer_f1"] - vanilla["answer_f1"]
        ).mean(),
        3,
    ),
)

print(
    "VBA-SCRAG minus Correction without gate:",
    round(
        100 * (
            vba["answer_f1"] - ungated["answer_f1"]
        ).mean(),
        3,
    ),
)

result.to_csv(
    "results/independent_correctness_comparison.csv",
    index=False,
)

print("\nSaved results/independent_correctness_comparison.csv")
