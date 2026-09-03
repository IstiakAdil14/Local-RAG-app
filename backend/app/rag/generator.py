import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any

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

    def generate_grounded_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        if not context_chunks:
            return "I could not find this information in the provided documents."

        formatted_context = "\n".join([
            f"[{c.get('metadata', {}).get('section', 'General')}]: {c.get('text', '')}"
            for c in context_chunks
        ])

        system_instruction = (
            "You are a factual assistant. Answer the question in 1 or 2 concise sentences using ONLY the provided Context. "
            "If the answer is not in the context, respond strictly with: "
            "'I could not find this information in the provided documents.'"
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Context:\n{formatted_context}\n\nQuestion: {query}\nAnswer:"}
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_tokens = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        gen_tokens = output_tokens[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()