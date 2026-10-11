class VBASCRAG:

    def __init__(
        self,
        retriever,
        generator,
        verifier,
        corrector,
        threshold=0.75,
        improvement_margin=0.0,
        max_iterations=3
    ):

        self.retriever = retriever
        self.generator = generator
        self.verifier = verifier
        self.corrector = corrector

        self.threshold = threshold
        self.improvement_margin = improvement_margin
        self.max_iterations = max_iterations


    def run(self, question):

        evidence = self.retriever.retrieve(
            question,
            top_k=3
        )

        answer = self.generator.generate(
            question,
            evidence
        )

        current_grounding = (
            self.verifier.calculate_grounding(
                answer,
                evidence
            )
        )

        initial_grounding = current_grounding

        iterations = 0
        correction_attempted = False
        correction_accepted = False
        correction_rejected = False

        while (
            current_grounding < self.threshold
            and iterations < self.max_iterations
        ):

            iterations += 1
            correction_attempted = True

            corrected_query = (
                self.corrector.correct_query(
                    question,
                    answer
                )
            )

            new_evidence = (
                self.retriever.retrieve(
                    corrected_query,
                    top_k=3
                )
            )

            candidate_answer = (
                self.generator.generate(
                    question,
                    new_evidence
                )
            )

            candidate_grounding = (
                self.verifier.calculate_grounding(
                    candidate_answer,
                    new_evidence
                )
            )

            improvement = (
                candidate_grounding
                - current_grounding
            )

            # VERIFY-BEFORE-ACCEPT
            if improvement >= self.improvement_margin:

                answer = candidate_answer
                evidence = new_evidence
                current_grounding = candidate_grounding

                correction_accepted = True
                correction_rejected = False

                break

            else:

                correction_rejected = True

        return {

            "question": question,

            "initial_grounding":
                initial_grounding,

            "grounding":
                current_grounding,

            "grounding_improvement":
                current_grounding
                - initial_grounding,

            "iterations":
                iterations,

            "correction_attempted":
                correction_attempted,

            "correction_accepted":
                correction_accepted,

            "correction_rejected":
                correction_rejected
        }