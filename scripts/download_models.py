import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_all_models():
    llm_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f">>> 1/3 Pre-caching local LLM ({llm_id})...")
    AutoTokenizer.from_pretrained(llm_id)
    AutoModelForCausalLM.from_pretrained(llm_id)
    print(" [OK] Local LLM cached.")

    print(">>> 2/3 Pre-caching BAAI/bge-m3 embedding model...") 
    SentenceTransformer("BAAI/bge-m3")
    print(" [OK] BGE-M3 cached locally.")   

    print(">>> 3/3 Pre-caching BAAI/bge-reranker-base reranker model...")
    CrossEncoder("BAAI/bge-reranker-base")
    print(" [OK] BGE-Reranker-Base cached locally.")

if __name__ == "__main__":
    download_all_models()