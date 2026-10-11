from sentence_transformers import SentenceTransformer
import numpy as np


class Verifier:
    def __init__(self):
        print("Loading verification model...")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def calculate_grounding(self, answer, evidence):

        if not evidence:
            return 0.0

        answer_embedding = self.model.encode(
            [answer],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        evidence_text = [
            item["text"] for item in evidence
        ]

        evidence_embeddings = self.model.encode(
            evidence_text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        similarities = np.dot(
            evidence_embeddings,
            answer_embedding
        )

        return float(np.max(similarities))