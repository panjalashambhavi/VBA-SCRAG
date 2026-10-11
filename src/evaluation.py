import pandas as pd


def evaluate_results(results):

    df = pd.DataFrame(results)

    total = len(df)

    accepted = df["accepted_correction"].sum()

    rejected = df["rejected_correction"].sum()

    abstained = df["abstained"].sum()

    correction_attempts = (
        accepted + rejected
    )

    if correction_attempts > 0:
        correction_success_rate = (
            accepted / correction_attempts
        )
    else:
        correction_success_rate = 0

    rejection_rate = (
        rejected / correction_attempts
        if correction_attempts > 0
        else 0
    )

    summary = {
        "Number of Questions": total,
        "Average Grounding":
            df["grounding"].mean(),
        "Average Iterations":
            df["iterations"].mean(),
        "Correction Success Rate":
            correction_success_rate,
        "Correction Rejection Rate":
            rejection_rate,
        "Abstention Rate":
            abstained / total
    }

    return summary