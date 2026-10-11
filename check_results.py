
import pandas as pd

files = {
    "0.60": "results/fixed_threshold_0.60_details.csv",
    "0.70": "results/fixed_threshold_0.70_details.csv",
    "0.75": "results/fixed_threshold_0.75_details.csv",
    "0.80": "results/fixed_threshold_0.80_details.csv",
    "0.90": "results/fixed_threshold_0.90_details.csv",
}

frames = {
    threshold: pd.read_csv(path).set_index("question")
    for threshold, path in files.items()
}

for threshold, df in frames.items():
    attempted = df[df["correction_attempted"]]
    accepted = df[df["correction_accepted"]]
    rejected = df[df["correction_rejected"]]

    print(f"\nTHRESHOLD {threshold}")
    print("Attempted:", len(attempted))
    print("Accepted:", len(accepted))
    print("Rejected:", len(rejected))
    print("Accepted with zero improvement:",
          (accepted["grounding_improvement"].abs() < 1e-8).sum())
    print("Accepted with positive improvement:",
          (accepted["grounding_improvement"] > 1e-8).sum())
    print("Mean candidate score:",
          attempted["candidate_grounding"].mean())

print("\nCOMPARING ACCEPTED QUESTIONS AT 0.60 AND 0.90")

a = frames["0.60"]
b = frames["0.90"]

comparison = pd.DataFrame({
    "score_060": a["grounding"],
    "score_090": b["grounding"],
    "accepted_060": a["correction_accepted"],
    "accepted_090": b["correction_accepted"],
})

print(comparison[
    comparison["accepted_060"] != comparison["accepted_090"]
].to_string())
