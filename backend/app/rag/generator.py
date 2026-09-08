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

    def _api_generate(self, context_text: str, user_query: str = "", max_tokens: int = 250) -> str:
        # 1. High-speed Pollinations JSON POST API
        try:
            url = "https://text.pollinations.ai/"
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an ultra-precise, grounded RAG assistant.\n"
                            "CRITICAL RULES:\n"
                            "1. Answer ONLY what the user explicitly asks. Provide NO extra, unasked, or irrelevant information.\n"
                            "2. Be direct, precise, and concise (1-2 sentences max for specific questions).\n"
                            "3. Base your answer strictly on the Context provided.\n"
                            "4. Do NOT dump raw text blocks or quote cell numbers.\n"
                            "5. If the exact answer is not explicitly stated in the context, reply EXACTLY: 'The document does not specify this information.'"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context_text}\n\nQuestion: {user_query}\n\nDirect Answer:"
                    }
                ],
                "model": "openai"
            }
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=7)
            if res.status_code == 200 and res.text.strip():
                ans = res.text.strip()
                if len(ans) > 2 and not ans.startswith("{") and "The document does not specify" not in ans:
                    return ans
        except Exception:
            pass

        # 2. Try Hugging Face Open-Access Router Endpoint
        headers = {"Content-Type": "application/json"}
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        models_to_try = [
            "Qwen/Qwen2.5-72B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2"
        ]

        prompt_str = f"Context:\n{context_text}\n\nQuestion: {user_query}\n\nDirect Answer:"
        for model in models_to_try:
            try:
                url = "https://router.huggingface.co/hf-inference/v1/chat/completions"
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt_str}],
                    "max_tokens": max_tokens
                }
                res = requests.post(url, headers=headers, json=payload, timeout=4)
                if res.status_code == 200:
                    data = res.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        ans = data["choices"][0]["message"]["content"].strip()
                        if ans and len(ans) > 2:
                            return ans
            except Exception:
                continue

        # 3. Pinpoint Keyword Match Fallback
        stop_words = {"what", "whats", "who", "where", "when", "how", "why", "tell", "give", "about", "this", "that", "with", "from", "the", "pdf", "doc", "document", "is", "are", "was", "were", "can", "she", "he", "they", "it", "name", "project", "explain"}
        query_words = set([w.lower() for w in re.findall(r"\w+", user_query) if len(w) > 2 and w.lower() not in stop_words])

        lines = [line.strip() for line in re.split(r"[\n\.;]+", context_text) if len(line.strip()) > 8]
        matched_lines = []
        for line in lines:
            line_lower = line.lower()
            match_count = sum(1 for qw in query_words if qw in line_lower)
            if match_count > 0:
                matched_lines.append((match_count, line))

        if matched_lines:
            matched_lines.sort(key=lambda x: x[0], reverse=True)
            best_match = matched_lines[0][1]
            clean_match = re.sub(r'^(Cell\s*\d+\s*:|\*|\-)\s*', '', best_match, flags=re.I).strip()
            return clean_match

        return "The document does not specify this information."

    def summarize_document(self, context_chunks: List[Dict[str, Any]]) -> str:
        if not context_chunks:
            return "The provided document is empty or could not be parsed."

        context_text = "\n\n".join([c.get("text", "")[:350] for c in context_chunks[:6]])

        # 1. Try Pollinations JSON POST API for clean high-level summary
        try:
            url = "https://text.pollinations.ai/"
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional document analyst. Provide a clear, high-level 3-sentence summary explaining what this document is about. Do NOT quote cell numbers or raw code snippets."
                    },
                    {
                        "role": "user",
                        "content": f"Document Content:\n{context_text}\n\nDocument Overview:"
                    }
                ],
                "model": "openai"
            }
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=7)
            if res.status_code == 200 and res.text.strip():
                ans = res.text.strip()
                if len(ans) > 10 and not ans.startswith("{"):
                    return ans
        except Exception:
            pass

        # 2. Structural Heading Extraction Fallback
        section_headers = []
        for c in context_chunks:
            text = c.get("text", "")
            for line in text.split("\n"):
                line_clean = line.strip()
                # Extract headers (e.g. Data Loading, Data Cleaning, Model Training, Personal Information)
                if re.match(r'^(#+|\d+\.|\b[A-Z0-9\s_\-\.]{3,}\b$|^[A-Z][A-Za-z0-9\s]{2,30}:)', line_clean) and len(line_clean) < 60:
                    section_headers.append(re.sub(r'[^\w\s]', '', line_clean).strip())

        unique_sections = list(dict.fromkeys([s for s in section_headers if len(s) > 3]))[:5]
        if unique_sections:
            return (
                f"This document provides a structured guide covering the following main sections:\n\n"
                + "\n".join([f"• {sec}" for sec in unique_sections])
            )

        # Fallback first paragraph summary
        first_text = context_chunks[0].get("text", "").strip()[:250]
        return f"This document covers the following core topic:\n\n{first_text}..."

    def generate_grounded_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        max_tokens: int = 250
    ) -> str:
        if not context_chunks:
            return "The document does not specify this information."

        context_text = "\n\n".join([
            f"{c.get('text', '')}"
            for c in context_chunks
        ])

        system_instruction = (
            "You are an ultra-precise, grounded RAG assistant.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Answer ONLY what the user explicitly asks. Provide ZERO extra, unasked, or irrelevant information.\n"
            "2. Be direct, precise, and concise (1-2 sentences max for specific questions).\n"
            "3. Do NOT dump raw text blocks or quote cell numbers.\n"
            "4. If the exact answer is not explicitly stated in the context, reply EXACTLY: 'The document does not specify this information.'"
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}\n\nDirect Answer:"}
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

        raw_answer = self._api_generate(context_text, user_query=query, max_tokens=max_tokens)
        for prefix in ["Direct Answer:", "Answer:", "Summary:", "Based on the context,", "According to the provided documents,"]:
            if raw_answer.startswith(prefix):
                raw_answer = raw_answer[len(prefix):].strip()
        return raw_answer