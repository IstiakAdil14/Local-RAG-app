import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_local_llm():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading local model: ({model_id})...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16, 
        low_cpu_mem_usage=True
    ).to(device)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "State 'phase 1 operational' and nothing else."}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer([text], return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=30)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    print("\n--- LLM Output ---")
    print(response)

    print("------------------\n")
    print("Phase 1 envirnment verified successfully.")

if __name__ == "__main__":
    test_local_llm()