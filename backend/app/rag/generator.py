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
            "You are a strictly grounded RAG assistant. Your task is to answer the user query based ONLY and EXCLUSIVELY on the clear facts in the provided Context below. "
            "Do NOT use external pre-trained knowledge or make assumptions beyond the text. "
            "If the Context does not explicitly contain the answer, reply EXACTLY: 'I could not find this information in the provided documents.'"
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Context:\n{formatted_context}\n\nQuestion: {query}\n\nAnswer:"}
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
                pad_token_id=self.tokenizer.eos_token_id
            )

        gen_tokens = output_tokens[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()