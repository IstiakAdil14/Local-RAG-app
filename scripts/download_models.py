import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_all_models():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f">>> 1/2 Downloading/Caching local LLM({model_id})...")
    AutoTokenizer.from_pretrained(model_id)
    AutoModelForCausalLM.from_pretrained(model_id)
    print(" Local LLM cached.")

    print(">>> 2/2 Pre-caching BAAI/bge-m3 embedding model..") 
    SentenceTransformer("BAAI/bge-m3")
    print(" BGE-M3 cached locally.")   

if __name__ == "__main__":
    download_all_models()

        