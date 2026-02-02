import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


# define tokenizer, model and pipeline
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    trust_remote_code = True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    trust_remote_code = True,
    device_map = "auto",
    load_in_4bit = True,
    dtype = torch.float16
)

# connect both in pipeline
pipe = pipeline(
    "text_generation",
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 256,
    temprature = 0.7
)

# wrap the pipeline with the langchain
from langchain_community.llms import HuggigngFacePipeline

llm = HuggigngFacePipeline(pipeline= pipe) 

# call the model
response = llm.invoke({"سلام خودت را معرفی کن"})
print(response)