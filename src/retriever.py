from sentence_transformers import SentenceTransformer
import numpy as np


class Retriever:
    def __init__(self, documents):
        self.documents = documents

        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Creating document embeddings...")
        self.embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def retrieve(self, query, top_k=3):
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        scores = np.dot(self.embeddings, query_embedding)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:
            results.append({
                "text": self.documents[index],
                "score": float(scores[index])
            })

        return results