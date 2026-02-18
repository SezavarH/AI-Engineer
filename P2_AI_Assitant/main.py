import torch
import gradio as gr
from threading import Thread
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TextIteratorStreamer
)

# --- CONFIGURATION ---
MODEL_REPO = "Qwen/Qwen2.5-7B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)

# 4-bit Quantization to fit comfortably on RTX 4060 Ti
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading model onto GPU...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="sdpa" # Speed boost for RTX 40-series
)

def predict(message, history):
    # 1. Start with the system prompt
    messages = [{"role": "system", "content": "You are a helpful assistant. You speak fluent Persian."}]
    
    # 2. DYNAMIC HISTORY MAPPING
    # This loop handles both Gradio 4.x (lists) and Gradio 5.x (dicts)
    for turn in history:
        if isinstance(turn, dict):
            # Already in dict format {'role': '...', 'content': '...'}
            # We force content to string to avoid the TypeError
            messages.append({"role": turn["role"], "content": str(turn["content"])})
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            # Old format: [user_text, bot_text]
            messages.append({"role": "user", "content": str(turn[0])})
            messages.append({"role": "assistant", "content": str(turn[1])})
    
    # 3. Add the current user message
    messages.append({"role": "user", "content": str(message)})

    # 4. Apply chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    # 5. Setup Streaming (For real-time Persian response)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
    )

    # 6. Run generation in a background thread
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    # 7. Yield text chunks as they arrive
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text

# --- Gradio UI ---
view = gr.ChatInterface(
    fn=predict,
    title="Qwen 2.5 Persian Assistant",
    description="Optimized for RTX 4060 Ti. Supports real-time streaming.",
    examples=["چطور می‌توانم پایتون یاد بگیرم؟", "یک شعر کوتاه درباره باران بگو."]
)

if __name__ == "__main__":
    view.launch()