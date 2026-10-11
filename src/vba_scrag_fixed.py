class VBASCRAG:
    """Verify-before-accept RAG with one correction candidate per question."""

    def __init__(
        self,
        retriever,
        generator,
        verifier,
        corrector,
        threshold=0.75,
        improvement_margin=0.0,
        max_iterations=1,
    ):
        self.retriever = retriever
        self.generator = generator
        self.verifier = verifier
        self.corrector = corrector
        self.threshold = threshold
        self.improvement_margin = improvement_margin

        # The current corrector produces the same query each time,
        # so repeating a rejected candidate would not be meaningful.
        self.max_iterations = 1

    def run(self, question):
        evidence = self.retriever.retrieve(question, top_k=3)
        answer = self.generator.generate(question, evidence)

        initial_grounding = float(
            self.verifier.calculate_grounding(answer, evidence)
        )
        final_grounding = initial_grounding

        attempted = False
        accepted = False
        rejected = False
        candidate_grounding = None
        iterations = 0

        if initial_grounding < self.threshold:
            attempted = True
            iterations = 1

            corrected_query = self.corrector.correct_query(
                question, answer
            )
            candidate_evidence = self.retriever.retrieve(
                corrected_query, top_k=3
            )
            candidate_answer = self.generator.generate(
                question, candidate_evidence
            )
            candidate_grounding = float(
                self.verifier.calculate_grounding(
                    candidate_answer, candidate_evidence
                )
            )

            improvement = candidate_grounding - initial_grounding

            if improvement >= self.improvement_margin:
                answer = candidate_answer
                evidence = candidate_evidence
                final_grounding = candidate_grounding
                accepted = True
            else:
                # Keep the original answer if the candidate is rejected.
                rejected = True

        return {
            "question": question,
            "answer": answer,
            "initial_grounding": initial_grounding,
            "grounding": final_grounding,
            "grounding_improvement": final_grounding - initial_grounding,
            "candidate_grounding": candidate_grounding,
            "iterations": iterations,
            "correction_attempted": attempted,
            "correction_accepted": accepted,
            "correction_rejected": rejected,
        }