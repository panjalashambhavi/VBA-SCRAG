import pandas as pd


def run_vanilla_rag(
    questions,
    retriever,
    generator,
    verifier
):

    results = []

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

        results.append({
            "question": question,
            "answer": answer,
            "grounding": grounding,
            "iterations": 0
        })

    return results


def run_correction_without_gate(
    questions,
    retriever,
    generator,
    verifier,
    corrector
):

    results = []

    for question in questions:

        evidence = retriever.retrieve(
            question,
            top_k=3
        )

        answer = generator.generate(
            question,
            evidence
        )

        initial_grounding = (
            verifier.calculate_grounding(
                answer,
                evidence
            )
        )

        # Always apply correction
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

        corrected_grounding = (
            verifier.calculate_grounding(
                corrected_answer,
                new_evidence
            )
        )

        # IMPORTANT:
        # This baseline accepts the correction
        # without checking whether grounding improved.

        results.append({
            "question": question,
            "answer": corrected_answer,
            "grounding": corrected_grounding,
            "initial_grounding": initial_grounding,
            "iterations": 1
        })

    return results


def summarize(results):

    df = pd.DataFrame(results)

    return {
        "Average Grounding":
            df["grounding"].mean(),

        "Average Iterations":
            df["iterations"].mean()
    }