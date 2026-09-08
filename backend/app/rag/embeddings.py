import os
import requests
from pathlib import Path
from typing import List

# Ensure HuggingFace cache targets project models directory on D: drive
_project_root = Path(__file__).resolve().parent.parent.parent.parent
_models_dir = os.path.join(_project_root, "models", "huggingface")
os.makedirs(_models_dir, exist_ok=True)
os.environ.setdefault("HF_HOME", _models_dir)
os.environ.setdefault("TRANSFORMERS_CACHE", _models_dir)

class LocalEmbeddingEngine:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = None):
        self.model_name = model_name
        self.model = None
        self.use_api = False

        # Attempt to load local SentenceTransformer model
        try:
            import torch
            import sentence_transformers.sentence_transformer.modules as st_modules
            from sentence_transformers import SentenceTransformer

            # Patch Pooling load for legacy configs
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

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading embedding model '{model_name}' on device: {device}")
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            print(f"⚠️ Local PyTorch embedding loading skipped ({e}). Using Hugging Face Serverless API fallback.")
            self.use_api = True

    def _api_embed(self, texts: List[str]) -> List[List[float]]:
        headers = {}
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        results = []
        for text in texts:
            try:
                url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"
                res = requests.post(url, headers=headers, json={"inputs": text}, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    if isinstance(data, list) and isinstance(data[0], list):
                        # Mean pooling over token embeddings
                        mean_vec = [sum(col) / len(col) for col in zip(*data)]
                        results.append(mean_vec)
                    elif isinstance(data, list) and isinstance(data[0], (int, float)):
                        results.append(data)
                    else:
                        results.append([0.0] * 1024)
                else:
                    results.append([0.0] * 1024)
            except Exception:
                results.append([0.0] * 1024)
        return results

    def embed_texts(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        if not texts:
            return []
        if self.model is not None and not self.use_api:
            try:
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                return embeddings.tolist()
            except Exception:
                self.use_api = True

        return self._api_embed(texts)

    def embed_query(self, query: str) -> List[float]:
        res = self.embed_texts([query])
        return res[0] if res else [0.0] * 1024