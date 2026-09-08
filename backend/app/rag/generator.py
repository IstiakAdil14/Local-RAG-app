import os
import requests
from pathlib import Path
from typing import List, Dict, Any

def is_cloud_environment() -> bool:
    # Detect Streamlit Cloud, Render, or low RAM environments (< 3GB RAM)
    if os.getenv("STREAMLIT_SERVER_PORT") or os.getenv("HOME") == "/home/adminuser" or "/mount/src" in str(Path.cwd()):
        return True
    try:
        import psutil
        if psutil.virtual_memory().total < 3 * 1024**3:
            return True
    except Exception:
        pass
    return False

class LocalGenerator:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = None):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.use_api = False

        if is_cloud_environment():
            print(f"☁️ Cloud environment detected. Using Serverless HF API for generator ({model_id}).")
            self.use_api = True
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            torch.set_num_threads(os.cpu_count() or 8)

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"Loading generator model '{model_id}' on device: {device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float32,
                low_cpu_mem_usage=True
            ).to(device)
            self.model.eval()
            self.device = device
        except Exception as e:
            print(f"⚠️ Local generator loading skipped ({e}). Using Serverless API.")
            self.use_api = True

    def _api_generate(self, prompt_text: str, max_tokens: int = 250) -> str:
        headers = {}
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(model=self.model_id, token=hf_token)
            res = client.text_generation(prompt_text, max_new_tokens=max_tokens)
            if res:
                return res.strip()
        except Exception:
            pass

        try:
            url = f"https://api-inference.huggingface.co/models/{self.model_id}"
            payload = {
                "inputs": prompt_text,
                "parameters": {"max_new_tokens": max_tokens, "return_full_text": False}
            }
            res = requests.post(url, headers=headers, json=payload, timeout=20)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0].get("generated_text", "").strip()
                elif isinstance(data, dict):
                    return data.get("generated_text", "").strip()
        except Exception:
            pass

        return "I could not generate a response from the document context."

    def generate_grounded_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        max_tokens: int = 250
    ) -> str:
        if not context_chunks:
            return "I could not find this information in the provided documents."

        formatted_context = "\n\n".join([
            f"--- Document: {c.get('metadata', {}).get('document_name', 'Document')} | Section: {c.get('metadata', {}).get('section', 'General')} (Page {c.get('metadata', {}).get('page_number', 1)}) ---\n{c.get('text', '')}"
            for c in context_chunks
        ])

        system_instruction = (
            "You are a strictly grounded, accurate, and concise RAG assistant.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Answer the user query using ONLY explicit facts found directly in the Context below.\n"
            "2. Be direct, precise, and concise. Provide ONLY the specific information requested.\n"
            "3. Do NOT include conversational filler, greetings, introductory boilerplate, or unasked background information.\n"
            "4. Do NOT make assumptions, speculate, or draw from external knowledge outside the text.\n"
            "5. If the Context does not explicitly contain the answer, reply EXACTLY: 'I could not find this information in the provided documents.'"
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Context:\n{formatted_context}\n\nQuestion: {query}\n\nDirect Answer:"}
        ]

        if not self.use_api and self.model is not None and self.tokenizer is not None:
            try:
                import torch
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

                with torch.inference_mode():
                    output_tokens = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_tokens,
                        use_cache=True,
                        do_sample=False,
                        repetition_penalty=1.15,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                gen_tokens = output_tokens[0][inputs["input_ids"].shape[1]:]
                raw_answer = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

                for prefix in ["Direct Answer:", "Answer:", "Summary:", "Based on the context,", "According to the provided documents,"]:
                    if raw_answer.startswith(prefix):
                        raw_answer = raw_answer[len(prefix):].strip()

                return raw_answer
            except Exception:
                self.use_api = True

        prompt_fallback = f"{system_instruction}\n\nContext:\n{formatted_context}\n\nQuestion: {query}\n\nDirect Answer:"
        raw_answer = self._api_generate(prompt_fallback, max_tokens=max_tokens)
        for prefix in ["Direct Answer:", "Answer:", "Summary:", "Based on the context,", "According to the provided documents,"]:
            if raw_answer.startswith(prefix):
                raw_answer = raw_answer[len(prefix):].strip()
        return raw_answer