import torch
from sentence_transformers import SentenceTransformer
from typing import List

class LocalEmbeddingEngine:
    def __init__(self,model_name: str ="BAAI/bge-m3", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"Loading embedding model '{model_name}' on device: {self.device}")
        self.model = SentenceTransformer(model_name,device=self.device)
        
    def embed_texts(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_query(self,query:str)->List[float]:
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.tolist()
        

        