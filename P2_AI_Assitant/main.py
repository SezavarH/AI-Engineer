import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"           
# MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"



# CACHE_DIR = r""   


# ------------------ tokenizer -------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    trust_remote_code = True,
    # cache_dir=CACHE_DIR
)

# ------------------ config and Model  -------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    trust_remote_code = True,
    device_map = "auto",
    dtype = torch.float16,
    quantization_config=bnb_config,
    # cache_dir=CACHE_DIR
)

# ------------------- Pipeline: connet model and tokenizer --------------
pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    return_full_text=False,  # avoids repeating prompt
)


# -------------------- wrap the pipeline with the langchain ------------------
llm = HuggingFacePipeline(pipeline= pipe) 

# -------------------- call the model ------------------
response = llm.invoke("سلام خودت را معرفی کن")
print(response)