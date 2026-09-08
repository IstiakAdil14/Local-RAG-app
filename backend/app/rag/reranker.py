import os
from typing import List, Dict, Any

class LocalCrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = None):
        self.model_name = model_name
        self.model = None

        try:
            import torch
            from sentence_transformers import CrossEncoder
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading cross-encoder reranker '{model_name}' on device: {device}...")
            self.model = CrossEncoder(model_name, device=device)
        except Exception as e:
            print(f"⚠️ CrossEncoder loading skipped ({e}). Using candidate RRF score fallback.")
            self.model = None

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        if self.model is not None:
            try:
                pairs = [(query, c['text']) for c in candidates]
                scores = self.model.predict(pairs, show_progress_bar=False)

                reranked = []
                for candidate, score in zip(candidates, scores):
                    item = candidate.copy()
                    item["rerank_score"] = float(score)
                    reranked.append(item)

                reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
                return reranked[:top_n]
            except Exception:
                pass

        # Fallback to existing candidate ordering (RRF score)
        return candidates[:top_n]