import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any

# Maximize CPU core utilization for high-speed inference
torch.set_num_threads(os.cpu_count() or 8)

class LocalGenerator:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading generator model '{model_id}' on device: {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()

    def generate_grounded_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        max_tokens: int = 250
    ) -> str:
        if not context_chunks:
            return "I could not find this information in the provided documents."

        # Format retrieved context sections with explicit document name and metadata
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

        # Clean up common model prefixes and boilerplate
        for prefix in ["Direct Answer:", "Answer:", "Summary:", "Based on the context,", "According to the provided documents,"]:
            if raw_answer.startswith(prefix):
                raw_answer = raw_answer[len(prefix):].strip()

        return raw_answer