import os
import requests
import re
import urllib.parse
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
        # 1. Pollinations GET API (100% Instant, Zero Auth, Returns direct LLM text)
        try:
            clean_instruction = (
                "System: You are an accurate RAG assistant. Answer the user question directly in concise, professional natural language using ONLY the context provided. "
                "Do NOT quote raw context chunks. If information is missing, state what is missing.\n\n"
            )
            full_prompt = clean_instruction + prompt_text
            encoded_prompt = urllib.parse.quote(full_prompt[:2500])
            res = requests.get(f"https://text.pollinations.ai/{encoded_prompt}", timeout=7)
            if res.status_code == 200 and res.text.strip():
                ans = res.text.strip()
                if len(ans) > 5 and not ans.startswith("{") and "Relevant excerpts" not in ans:
                    return ans
        except Exception:
            pass

        # 2. Try Hugging Face Open-Access Router Endpoint
        headers = {}
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        models_to_try = [
            "Qwen/Qwen2.5-72B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2"
        ]

        for model in models_to_try:
            try:
                url = "https://router.huggingface.co/hf-inference/v1/chat/completions"
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "max_tokens": max_tokens
                }
                res = requests.post(url, headers=headers, json=payload, timeout=4)
                if res.status_code == 200:
                    data = res.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        ans = data["choices"][0]["message"]["content"].strip()
                        if ans and len(ans) > 5:
                            return ans
            except Exception:
                continue

        # 3. Universal Extractive Answer Synthesis (Never outputs raw chunk dump header)
        stop_words = {"what", "whats", "who", "where", "when", "how", "why", "tell", "give", "about", "this", "that", "with", "from", "the", "pdf", "doc", "document", "is", "are", "was", "were", "can", "she", "he", "they", "it"}
        query_words = set([w.lower() for w in re.findall(r"\w+", user_query) if len(w) > 2 and w.lower() not in stop_words])

        scored_lines = []
        for c in context_chunks:
            text = c.get("text", "")
            lines = [line.strip() for line in re.split(r"[\n\.;]+", text) if len(line.strip()) > 8]
            for line in lines:
                line_lower = line.lower()
                match_count = sum(1 for qw in query_words if qw in line_lower)
                if match_count > 0:
                    scored_lines.append((match_count, line))

        if scored_lines:
            scored_lines.sort(key=lambda x: x[0], reverse=True)
            best_lines = list(dict.fromkeys([line for _, line in scored_lines]))[:3]
            return "Based on the provided document:\n\n• " + "\n• ".join(best_lines)

        # Fallback snippet summary if no keyword overlap
        general_snippets = [c.get("text", "").strip()[:200] for c in context_chunks if c.get("text")]
        if general_snippets:
            return "Summary based on the document context:\n\n• " + "\n• ".join(general_snippets[:2])

        return "The provided document does not contain sufficient details regarding your query."

    def summarize_document(self, context_chunks: List[Dict[str, Any]]) -> str:
        if not context_chunks:
            return "The provided document is empty or could not be parsed."

        sections = list(dict.fromkeys([c.get("metadata", {}).get("section", "General") for c in context_chunks if c.get("metadata")]))
        snippets = "\n\n".join([f"[{c.get('metadata', {}).get('section', 'Section')}] {c.get('text', '')[:250]}" for c in context_chunks[:6]])

        prompt = (
            "Provide a clean, direct 3-bullet point summary explaining what this document is about based on the context:\n\n"
            f"Sections Found: {', '.join(sections[:5])}\n\nContext:\n{snippets}\n\nSummary:"
        )

        return self._api_generate(prompt, context_chunks, user_query="overview summary", max_tokens=300)

    def generate_grounded_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        max_tokens: int = 250
    ) -> str:
        if not context_chunks:
            return "The provided document does not contain information regarding your query."

        formatted_context = "\n\n".join([
            f"--- Section: {c.get('metadata', {}).get('section', 'General')} (Page {c.get('metadata', {}).get('page_number', 1)}) ---\n{c.get('text', '')}"
            for c in context_chunks
        ])

        system_instruction = (
            "You are a professional, accurate, and concise RAG assistant.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Synthesize a clean, natural-language answer to the user's question using ONLY the facts explicitly provided in the Context below.\n"
            "2. NEVER dump raw unformatted chunk blocks or list cell numbers.\n"
            "3. If the user asks for specific information (such as job placement or company names) and the Context only lists general qualifications or background, explicitly state what is in the document and clarify what is missing.\n"
            "4. Be direct, precise, and professional."
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