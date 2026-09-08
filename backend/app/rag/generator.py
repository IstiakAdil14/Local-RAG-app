import os
import requests
import re
from pathlib import Path
from typing import List, Dict, Any

def is_cloud_environment() -> bool:
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

    def _api_generate(self, prompt_text: str, context_chunks: List[Dict[str, Any]], user_query: str = "", max_tokens: int = 250) -> str:
        # 1. Try Pollinations Free Open-Access LLM Inference API (Instant, Free, Zero Auth)
        try:
            url = "https://text.pollinations.ai/"
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a strictly grounded, accurate, and concise RAG assistant. Answer the user query using ONLY explicit facts found directly in the Context provided."},
                    {"role": "user", "content": prompt_text}
                ],
                "model": "openai"
            }
            res = requests.post(url, json=payload, timeout=10)
            if res.status_code == 200 and res.text.strip():
                ans = res.text.strip()
                if len(ans) > 5 and not ans.startswith("{"):
                    return ans
        except Exception:
            pass

        # 2. Try Hugging Face Official Chat Router Endpoint
        headers = {}
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        models_to_try = [
            "Qwen/Qwen2.5-72B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "HuggingFaceH4/zephyr-7b-beta",
            self.model_id
        ]

        for model in models_to_try:
            try:
                url = "https://router.huggingface.co/hf-inference/v1/chat/completions"
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt_text}
                    ],
                    "max_tokens": max_tokens
                }
                res = requests.post(url, headers=headers, json=payload, timeout=8)
                if res.status_code == 200:
                    data = res.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        ans = data["choices"][0]["message"]["content"].strip()
                        if ans and len(ans) > 5:
                            return ans
            except Exception:
                continue

        # 3. Question-Aware Smart Extraction Fallback
        ignore_words = {"what", "whats", "who", "where", "when", "how", "tell", "give", "about", "this", "that", "with", "from", "the", "pdf", "doc", "document", "her", "his", "their", "your", "name", "is", "are", "was", "were"}
        q_words = [w.lower() for w in re.findall(r"\w+", user_query) if len(w) > 2 and w.lower() not in ignore_words]
        
        # Keyword synonyms expansion
        synonyms = {
            "profession": ["profession", "career", "occupation", "job", "work", "midwife", "objective", "registered"],
            "qualification": ["qualification", "qualifications", "education", "academic", "hsc", "ssc", "gpa", "school", "college", "degree", "certificate"],
            "educational": ["educational", "education", "academic", "hsc", "ssc", "gpa", "school", "college", "degree", "certificate"]
        }
        
        expanded_keywords = set(q_words)
        for qw in q_words:
            if qw in synonyms:
                expanded_keywords.update(synonyms[qw])

        matched_sentences = []
        for c in context_chunks:
            text = c.get("text", "")
            lines = [line.strip() for line in re.split(r"[\n\.]+", text) if len(line.strip()) > 8]
            for line in lines:
                line_lower = line.lower()
                if any(kw in line_lower for kw in expanded_keywords):
                    matched_sentences.append(line)

        if matched_sentences:
            unique_matches = list(dict.fromkeys(matched_sentences))[:4]
            return "\n\n".join([f"• {m}" for m in unique_matches])

        # Fallback to general context chunk snippets if no specific keyword match
        general_chunks = [c.get("text", "").strip()[:300] for c in context_chunks if c.get("text")]
        if general_chunks:
            return "\n\n".join([f"• {chunk}..." for chunk in general_chunks[:3]])

        return "I could not find this information in the provided documents."

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
        raw_answer = self._api_generate(prompt_fallback, context_chunks, user_query=query, max_tokens=max_tokens)
        for prefix in ["Direct Answer:", "Answer:", "Summary:", "Based on the context,", "According to the provided documents,"]:
            if raw_answer.startswith(prefix):
                raw_answer = raw_answer[len(prefix):].strip()
        return raw_answer