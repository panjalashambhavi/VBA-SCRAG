class Corrector:

    def correct_query(self, question, previous_answer):

        # Reformulate the query so that retrieval
        # focuses on explicit factual evidence.
        corrected_query = (
            f"{question} "
            f"Give a precise factual explanation and supporting evidence."
        )

        return corrected_query