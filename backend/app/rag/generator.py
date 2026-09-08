import os
import requests
import re
import urllib.parse
from pathlib import Path
from typing import List, Dict, Any, Optional

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
                            "You are an expert AI document assistant.\n"
                            "CRITICAL INSTRUCTIONS:\n"
                            "1. Answer the user's question directly and concisely based ONLY on the provided context.\n"
                            "2. Write in complete, professional natural sentences. Do NOT output raw title lines or cell numbers alone."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context_text[:2000]}\n\nQuestion: {user_query}\n\nAnswer:"
                    }
                ],
                "model": "openai"
            }
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=6)
            if res.status_code == 200 and res.text.strip():
                ans = res.text.strip()
                if len(ans) > 5 and not ans.startswith("{") and "The document does not specify" not in ans:
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

        # 3. Pinpoint Keyword Match Fallback with Articulate Sentence Synthesis
        stop_words = {"what", "whats", "who", "where", "when", "how", "why", "tell", "give", "this", "that", "with", "from", "the", "pdf", "doc", "document", "is", "are", "was", "were", "can", "she", "he", "they", "it", "name", "project", "explain", "ive", "please", "show", "me", "summarize", "overview"}
        query_words = set([w.lower() for w in re.findall(r"\w+", user_query) if len(w) > 2 and w.lower() not in stop_words])

        is_summary_query = any(w in user_query.lower() for w in ["overview", "summary", "summarize"]) or bool(re.search(r'\babout\s+(this|the)\s+(document|pdf|file|doc)\b', user_query.lower()))
        if not query_words or is_summary_query:
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
            top_matches = [m[1] for m in matched_lines[:2]]
            
            cleaned_snippets = []
            for line in top_matches:
                clean_line = re.sub(r'^(Cell\s*\d+\s*:|\*|\-)\s*', '', line, flags=re.I).strip()
                cleaned_snippets.append(clean_line)

            combined_info = " ".join(cleaned_snippets)
            if not combined_info.endswith("."):
                combined_info += "."

            return f"According to the document: {combined_info}"

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
        raw_headers = []
        major_category_headers = []

        def is_sensitive_contact_field(text: str) -> bool:
            t_lower = text.lower()
            return any(k in t_lower for k in [
                "phone", "mobile", "cell", "tel", "+88", "email", "@", "gmail",
                "nid", "passport", "dob", "birth", "religion", "father", "mother"
            ])

        for c in context_chunks:
            sec_meta = c.get("metadata", {}).get("section", "")
            if sec_meta and sec_meta not in ["General", "General Section"] and len(sec_meta) < 75:
                if not is_sensitive_contact_field(sec_meta):
                    raw_headers.append(sec_meta.strip())

            text = c.get("text", "")
            # Split text by line breaks AND bullet symbols (•), pipes (|), or semicolons
            split_parts = re.split(r'[•|\;\n]', text)

            for part in split_parts:
                line_clean = part.strip()
                if not line_clean or len(line_clean) > 75 or is_sensitive_contact_field(line_clean):
                    continue

                if any(emoji in line_clean for emoji in ["📌", "🧹", "🔍", "⚙️", "📊"]) or "(Cells" in line_clean:
                    major_category_headers.append(line_clean)

                clean_no_symbols = re.sub(r'^[^\w\s]+', '', line_clean).strip()
                clean_no_symbols = re.sub(r'[\:\-]$', '', clean_no_symbols).strip()

                is_header = (
                    bool(re.search(r'\(Cells?\s*\d+', line_clean, re.I)) or
                    bool(re.match(r'^(📌|🧹|🔍|⚙️|📊|#+|\d+[\.\:]|Cell\s*\d+[\:\-])', line_clean, re.I)) or
                    bool(re.match(r'^(CURRICULUM|CV|CAREER|ACADEMIC|QUALIFICATION|PERSONAL|EXPERIENCE|COMPUTER|LITERACY|LANGUAGE|SKILLS|PROJECTS|Data|Model|Feature|Clean|Missing|Outlier|Prediction|Evaluation|Summary|Introduction|Conclusion)\b', clean_no_symbols, re.I)) or
                    (clean_no_symbols.isupper() and len(clean_no_symbols) > 4)
                )

                if is_header:
                    display_header = re.sub(r'^[^\w\s]+', '', line_clean).strip()
                    display_header = re.sub(r'^Cell\s*\d+[\:\-]?\s*', '', display_header, flags=re.I).strip()
                    display_header = re.sub(r'[\:\-]$', '', display_header).strip()
                    if display_header and len(display_header) > 3 and not is_sensitive_contact_field(display_header):
                        raw_headers.append(display_header)

        # Prioritize major category section headers if found
        if major_category_headers:
            unique_majors = list(dict.fromkeys(major_category_headers))[:8]
            return (
                f"This document provides a structured guide covering the following main sections:\n\n"
                + "\n".join([f"• {sec}" for sec in unique_majors])
            )

        unique_sections = list(dict.fromkeys(raw_headers))[:8]
        if len(unique_sections) >= 2:
            return (
                f"This document provides a structured overview covering the following key sections:\n\n"
                + "\n".join([f"• {sec}" for sec in unique_sections])
            )

        # 3. Excerpt Paragraph Fallback
        lines = [l.strip() for l in full_context.split("\n") if len(l.strip()) > 15 and not l.strip().startswith("Cell")]
        excerpt = " ".join(lines[:4])[:350]
        if excerpt:
            return f"This document outlines the following topic:\n\n{excerpt}..."

        return f"This document covers the following core content:\n\n{full_context[:300]}..."

    def generate_metadata_answer(self, query: str, doc_metadata: Optional[Any] = None, fallback_chunks: List[Dict[str, Any]] = None) -> str:
        q_lower = query.lower()
        title = getattr(doc_metadata, "title", "") if doc_metadata else ""
        doc_name = getattr(doc_metadata, "document_name", "") if doc_metadata else ""
        total_pages = getattr(doc_metadata, "total_pages", None) if doc_metadata else None
        first_page = getattr(doc_metadata, "first_page_text", "") if doc_metadata else ""

        if not first_page and fallback_chunks:
            first_page = fallback_chunks[0].get("text", "")

        # 1. Title requests
        if any(k in q_lower for k in ["title", "called", "subject", "course"]):
            from app.ingestion.parser import DocumentParser

            # If stored title looks like an institutional/exam header, try re-extracting from first page
            is_institutional = any(re.search(pat, title, re.I) for pat in [
                r'\b(college|university|school|institute|academy|department|faculty)\b',
                r'\b(midterm|examination|final exam|test|question paper)\b'
            ]) if title else False

            if title and not is_institutional and len(title) > 3:
                return f"The title of this document is '{title}'."

            if first_page:
                extracted = DocumentParser.extract_document_title("", [{"text": first_page}])
                if extracted and len(extracted) > 3 and not any(re.search(pat, extracted, re.I) for pat in [
                    r'\b(college|university|school|institute|academy)\b',
                    r'\b(midterm|examination|final exam)\b'
                ]):
                    return f"The title of this document is '{extracted}'."

            if title and len(title) > 3:
                return f"The title of this document is '{title}'."

            if first_page:
                extracted = DocumentParser.extract_document_title("", [{"text": first_page}])
                if extracted and len(extracted) > 3:
                    return f"The title of this document is '{extracted}'."

            if doc_name:
                clean_name = os.path.splitext(doc_name)[0].replace("_", " ").replace("-", " ")
                return f"The title of this document is '{clean_name}'."

        # 2. Page count requests
        if any(k in q_lower for k in ["page", "pages", "length"]):
            if total_pages:
                return f"This document contains {total_pages} page(s)."

        # 3. Filename requests
        if any(k in q_lower for k in ["filename", "file name"]):
            if doc_name:
                return f"The filename of this document is '{doc_name}'."

        # 4. General metadata response fallback
        if title or doc_name:
            t_str = f"'{title}'" if title else f"'{doc_name}'"
            p_str = f" ({total_pages} pages)" if total_pages else ""
            return f"This document is titled {t_str}{p_str}."

        return "The document does not specify this metadata."

    def _extract_targeted_attribute(self, query: str, context_text: str) -> Optional[str]:
        q_lower = query.lower()

        # Title Extraction Fallback
        if any(k in q_lower for k in ["title of", "what is the title", "document title", "course title"]):
            from app.ingestion.parser import DocumentParser
            extracted = DocumentParser.extract_document_title("", [{"text": context_text}])
            if extracted and len(extracted) > 3:
                return extracted
            match = re.search(r"(?:Title|Course|Subject)\s*[:\-]\s*(.+?)(?=\s*[\n;]|$)", context_text, re.I)
            if match:
                val = match.group(1).strip()
                if val and len(val) > 3:
                    return val

        # Experience / Nursing Experience Extraction
        if any(k in q_lower for k in ["experience", "experiences", "worked", "duty", "ward", "position"]):
            exp_match = re.search(r"(?:EXPERIENCE|WORK\s*EXPERIENCE|PROFESSIONAL\s*EXPERIENCE|NURSING\s*EXPERIENCE)\s*[:\-]?\s*([\s\S]+?)(?=\s*(?:EDUCATION|QUALIFICATION|ACADEMIC|REFERENCE|DECLARATION|SKILLS|PROJECTS|TRAINING|HOBBIES|\b[A-Z\s]{4,}\:|$))", context_text, re.I)
            if exp_match:
                exp_text = exp_match.group(1).strip()
                exp_items = [re.sub(r'^[^\w]+', '', item).strip() for item in re.split(r'[\*\•\n\|]+', exp_text) if len(item.strip()) > 2]
                if exp_items:
                    formatted_exp = ", ".join(exp_items[:10])
                    return f"The document lists the following experience: {formatted_exp}."
                if exp_text:
                    return f"The document lists the following experience: {exp_text}."

        # Candidate Name / CV Owner Extraction
        if any(k in q_lower for k in ["whose", "whos", "candidate", "who is this", "owner", "who is she", "who is he", "her name", "his name", "full name", "applicant"]):
            decl_match = re.search(r"\bI,\s*([A-Z][A-Za-z\s\.]+?),\s*hereby\s*declare", context_text, re.I)
            if decl_match:
                return decl_match.group(1).strip()

            name_match = re.search(r"(?:Full\s*Name|Candidate\s*Name|Name)\s*[:\-]\s*([A-Za-z\s\.]+?)(?=\s*(?:Father|Mother|Present|Permanent|DECLARATION|Date|Religion|NID|Mobile|Phone|Address|[\:\-\n]|$))", context_text, re.I)
            if name_match:
                val = name_match.group(1).strip()
                if val and len(val) > 2 and "VITAE" not in val.upper():
                    return val

            cv_match = re.search(r"(?:CURRICULUM\s*VITAE|CV)\s*(?:OF)?\s*[:\-]?\s*([A-Z][A-Za-z\s\.]+?)(?=\s*(?:Cell|Mobile|Phone|Email|CAREER|ACADEMIC|PERSONAL|[\:\-\n]|$))", context_text, re.I)
            if cv_match:
                val = cv_match.group(1).strip()
                if val and len(val) > 2 and "VITAE" not in val.upper():
                    return val

        field_regex = None
        if "father" in q_lower:
            field_regex = r"(?:Father[’']?s?\s*Name|Father)\s*[:\-]\s*([A-Za-z\s\.]+?)(?=\s*(?:Mother|Present|Permanent|DECLARATION|Date|Religion|NID|Mobile|Phone|Address|[\:\-\n]|$))"
        elif "mother" in q_lower:
            field_regex = r"(?:Mother[’']?s?\s*Name|Mother)\s*[:\-]\s*([A-Za-z\s\.]+?)(?=\s*(?:Father|Present|Permanent|DECLARATION|Date|Religion|NID|Mobile|Phone|Address|[\:\-\n]|$))"
        elif "address" in q_lower:
            field_regex = r"(?:Present\s*Address|Permanent\s*Address|Address)\s*[:\-]\s*(.+?)(?=\s*(?:DECLARATION|Date|Religion|NID|Mobile|Phone|[\n]|$))"
        elif "email" in q_lower:
            field_regex = r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
        elif "phone" in q_lower or "mobile" in q_lower or "contact" in q_lower:
            field_regex = r"(\+?\d[\d\s\-\(\)]{8,})"

        if field_regex:
            match = re.search(field_regex, context_text, re.IGNORECASE)
            if match:
                extracted_val = match.group(1).strip()
                extracted_val = re.sub(r'[\:\-]$', '', extracted_val).strip()
                if extracted_val and len(extracted_val) > 1:
                    return extracted_val
        return None

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

        # Pinpoint targeted attribute extraction (e.g. title, father's name, mother's name, address, email)
        attr_match = self._extract_targeted_attribute(query, context_text)
        if attr_match:
            return attr_match

        system_instruction = (
            "You are an ultra-precise, grounded RAG assistant.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Answer ONLY what the user explicitly asks based on the provided context.\n"
            "2. Be direct, precise, and concise (1-5 words max for specific fact lookups).\n"
            "3. Do NOT output unasked fields or extra sentences.\n"
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