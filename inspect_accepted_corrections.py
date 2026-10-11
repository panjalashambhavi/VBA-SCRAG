
import pandas as pd

path = "results/paired_gate_experiment.csv"
df = pd.read_csv(path)

accepted = df[df["gate_accepted"] == True].copy()

print("Accepted corrections:", len(accepted))

columns = [
    "question",
    "reference",
    "initial_answer",
    "candidate_answer",
    "initial_grounding",
    "candidate_grounding",
    "ungated_f1",
    "gated_f1",
]

print(accepted[columns].to_string(index=False))

print("\nF1 change for accepted candidates:")
accepted["f1_change"] = (
    accepted["gated_f1"] - accepted["ungated_f1"]
)
print(accepted[["question", "f1_change"]].to_string(index=False))
