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
# These models are OPEN ACCESS (No gated permission required)
MODELS = {
    "Qwen 2.5-7B (Best Persian Support)": "Qwen/Qwen2.5-7B-Instruct",
    "Gemma-2-9B (Most Intelligent)": "google/gemma-2-9b-it",
    "Mistral-7B-v0.3 (Fast & Stable)": "mistralai/Mistral-7B-Instruct-v0.3"
}

current_model = None
current_tokenizer = None
current_name = ""

def load_selected_model(name):
    global current_model, current_tokenizer, current_name
    repo_id = MODELS[name]
    if current_name == repo_id:
        return f"Model {name} is already active."
    
    # Cleanup memory before loading new model
    if current_model is not None:
        print("Cleaning up VRAM...")
        del current_model
        del current_tokenizer
        torch.cuda.empty_cache()

    print(f"Loading {repo_id}...")
    
    # 4-bit quantization ensures these 7B-9B models run fast on 4060 Ti
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    current_tokenizer = AutoTokenizer.from_pretrained(repo_id)
    current_model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa" # Hardware acceleration for 40-series
    )
    current_name = repo_id
    return f"Active Model: {name}"

def check_system_role_support(model_id):
    """Check if the model supports system role in chat template"""
    models_without_system = ["google/gemma-2-9b-it"]
    return model_id not in models_without_system

def predict(message, history, model_choice):
    global current_model, current_tokenizer, current_name
    
    if current_name != MODELS[model_choice]:
        load_selected_model(model_choice)

    # Build messages based on model's system role support
    messages = []
    
    # Add system prompt if model supports it
    if check_system_role_support(current_name):
        messages.append({"role": "system", "content": "You are a helpful assistant. You speak and write fluent, accurate Persian (Farsi)."})
    
    # Rebuild history (Handles multiple Gradio versions)
    for turn in history:
        if isinstance(turn, dict):
            messages.append(turn)
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            messages.append({"role": "user", "content": str(turn[0])})
            messages.append({"role": "assistant", "content": str(turn[1])})

    messages.append({"role": "user", "content": str(message)})

    # For models without system role support, prepend system instruction to first user message
    if not check_system_role_support(current_name):
        system_msg = "You are a helpful assistant. You speak and write fluent, accurate Persian (Farsi). "
        if messages and messages[0]["role"] == "user":
            messages[0]["content"] = system_msg + messages[0]["content"]
        else:
            # If no user message exists (shouldn't happen), create one
            messages.insert(0, {"role": "user", "content": system_msg})

    try:
        # Apply Template with error handling
        inputs = current_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(current_model.device)
    except Exception as e:
        # Fallback: manually format if template fails
        print(f"Template error: {e}, using fallback formatting")
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"
        prompt += "Assistant: "
        
        inputs = current_tokenizer(prompt, return_tensors="pt").to(current_model.device)

    # Streamer
    streamer = TextIteratorStreamer(current_tokenizer, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=1024,
        temperature=0.6, # Slightly lower for better accuracy
        do_sample=True,
        pad_token_id=current_tokenizer.eos_token_id
    )

    # Threaded Generation
    thread = Thread(target=current_model.generate, kwargs=generate_kwargs)
    thread.start()

    partial_text = ""
    for new_text in streamer:
        # Clean up any leftover artifacts from specific model templates
        clean_text = new_text.replace("<|im_end|>", "").replace("<|end|>", "").replace("<|eot_id|>", "")
        partial_text += clean_text
        yield partial_text

# --- GRADIO UI ---
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("## 🇮🇷 RTX 4060 Ti Persian Assistant (Open Access Models)")
    
    with gr.Row():
        model_select = gr.Dropdown(
            choices=list(MODELS.keys()), 
            value="Qwen 2.5-7B (Best Persian Support)", 
            label="Switch Model"
        )
        load_btn = gr.Button("Switch Model", variant="secondary")
        status_bar = gr.Textbox(label="Status", interactive=False)

    # Note: We don't use type="messages" here to stay compatible with your Gradio version
    chat = gr.ChatInterface(
        fn=predict,
        additional_inputs=[model_select]
    )
    
    load_btn.click(load_selected_model, inputs=[model_select], outputs=[status_bar])

if __name__ == "__main__":
    # Initial load of the default model
    load_selected_model("Qwen 2.5-7B (Best Persian Support)")
    demo.launch()