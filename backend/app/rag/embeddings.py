import os
from pathlib import Path

# Ensure HuggingFace cache targets project models directory on D: drive
_project_root = Path(__file__).resolve().parent.parent.parent.parent
_models_dir = os.path.join(_project_root, "models", "huggingface")
os.makedirs(_models_dir, exist_ok=True)
os.environ.setdefault("HF_HOME", _models_dir)
os.environ.setdefault("TRANSFORMERS_CACHE", _models_dir)

import torch
import sentence_transformers.sentence_transformer.modules as st_modules
from sentence_transformers import SentenceTransformer
from typing import List

# Compatibility fix for newer sentence-transformers versions with legacy HF model configs
_original_pooling_load = st_modules.Pooling.load

@classmethod
def _patched_pooling_load(cls, model_path: str, **kwargs):
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if "word_embedding_dimension" in config and "embedding_dimension" not in config:
            config["embedding_dimension"] = config["word_embedding_dimension"]
        return cls(**config)
    return _original_pooling_load(model_path, **kwargs)

st_modules.Pooling.load = _patched_pooling_load

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
        

        