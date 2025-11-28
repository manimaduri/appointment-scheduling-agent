import os
from typing import List
from sentence_transformers import SentenceTransformer



class HFEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        token = os.getenv("HUGGINGFACE_TOKEN")

        if token:
            self.model = SentenceTransformer(model_name, use_auth_token=token)
        else:
            self.model = SentenceTransformer(model_name)
            
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.
        """
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            print(f"Error embedding documents: {e}")
            return [[0.0] * 384 for _ in texts]  # fallback vector size

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single text query.
        """
        try:
            embedding = self.model.encode([text], convert_to_numpy=True)[0]
            return embedding.tolist()
        except Exception as e:
            print(f"Error embedding query: {e}")
            return [0.0] * 384

    def __call__(self, text: str):
        return self.embed_query(text)


def get_embeddings() -> HFEmbeddings:
    return HFEmbeddings()
