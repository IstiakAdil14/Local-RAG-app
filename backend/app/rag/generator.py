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

    def _api_generate(self, context_text: str, user_query: str = "", max_tokens: int = 250) -> str:
        # 1. High-speed Pollinations GET API
        try:
            prompt = f"Context:\n{context_text[:1500]}\n\nQuestion: {user_query}\n\nProvide a direct 1-2 sentence answer based ONLY on the context:"
            url = f"https://text.pollinations.ai/{urllib.parse.quote(prompt)}"
            res = requests.get(url, timeout=7)
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
        stop_words = {"what", "whats", "who", "where", "when", "how", "why", "tell", "give", "about", "this", "that", "with", "from", "the", "pdf", "doc", "document", "is", "are", "was", "were", "can", "she", "he", "they", "it", "name", "project", "explain", "ive", "please", "show", "me", "summarize", "overview"}
        query_words = set([w.lower() for w in re.findall(r"\w+", user_query) if len(w) > 2 and w.lower() not in stop_words])

        if not query_words or any(w in user_query.lower() for w in ["explain", "overview", "summary", "summarize", "about"]):
            fake_chunks = [{"text": context_text}]
            return self.summarize_document(fake_chunks)

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

        full_context = "\n\n".join([c.get("text", "") for c in context_chunks])
        context_summary_text = full_context[:2500]

        # 1. Try Pollinations JSON POST API for clean 3-sentence summary
        try:
            url = "https://text.pollinations.ai/"
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a senior document analyst. Summarize the main purpose, dataset/topic, and key steps outlined in this document in a clear 3-4 sentence paragraph. Do not quote cell numbers or raw chunk IDs."
                    },
                    {
                        "role": "user",
                        "content": f"Document Text:\n{context_summary_text}\n\nSummary:"
                    }
                ],
                "model": "openai"
            }
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=6)
            if res.status_code == 200 and res.text.strip():
                ans = res.text.strip()
                if len(ans) > 25 and not ans.startswith("{") and "The document does not specify" not in ans:
                    return ans
        except Exception:
            pass

        # 2. Universal Section & Header Extraction Fallback across ALL chunks
        section_headers = []
        for c in context_chunks:
            sec_meta = c.get("metadata", {}).get("section", "")
            if sec_meta and sec_meta not in ["General", "General Section"] and len(sec_meta) < 75:
                clean_sec = re.sub(r'^[^\w\s]+', '', sec_meta).strip()
                clean_sec = re.sub(r'^Cell\s*\d+[\:\-]?\s*', '', clean_sec, flags=re.I).strip()
                if clean_sec and len(clean_sec) > 3:
                    section_headers.append(clean_sec)

            text = c.get("text", "")
            for line in text.split("\n"):
                line_clean = line.strip()
                if not line_clean or len(line_clean) > 80:
                    continue

                clean_no_symbols = re.sub(r'^[^\w\s]+', '', line_clean).strip()

                is_header = (
                    bool(re.search(r'\(Cells?\s*\d+', line_clean, re.I)) or
                    bool(re.match(r'^(📌|🧹|🔍|⚙️|📊|#+|\d+[\.\:]|Cell\s*\d+[\:\-])', line_clean, re.I)) or
                    bool(re.match(r'^(Data|Model|Feature|Clean|Missing|Outlier|Prediction|Evaluation|Personal|Education|Experience|Skills|Summary|Introduction|Conclusion)\b', clean_no_symbols, re.I))
                )

                if is_header:
                    display_header = re.sub(r'^[^\w\s]+', '', line_clean).strip()
                    display_header = re.sub(r'^Cell\s*\d+[\:\-]?\s*', '', display_header, flags=re.I).strip()
                    if display_header and len(display_header) > 3:
                        section_headers.append(display_header)

        unique_sections = list(dict.fromkeys(section_headers))[:8]
        if len(unique_sections) >= 2:
            return (
                f"This document provides a structured guide covering the following main sections:\n\n"
                + "\n".join([f"• {sec}" for sec in unique_sections])
            )

        # 3. Excerpt Paragraph Fallback
        lines = [l.strip() for l in full_context.split("\n") if len(l.strip()) > 15 and not l.strip().startswith("Cell")]
        excerpt = " ".join(lines[:4])[:350]
        if excerpt:
            return f"This document outlines the following topic:\n\n{excerpt}..."

        return f"This document covers the following core content:\n\n{full_context[:300]}..."

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