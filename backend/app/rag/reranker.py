import torch
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class LocalCrossEncoderReranker:
    def __init__(self, model_name: str="BAAI/bge-reranker-base", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading cross-encoder reranker '{model_name}' on device: {self.device}...")
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int=3) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        
        pairs = [(query, c['text']) for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        reranked = []
        for candidate, score in zip(candidates, scores):
            item = candidate.copy()
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda x:x["rerank_score"], reverse=True)
        return reranked[:top_n]