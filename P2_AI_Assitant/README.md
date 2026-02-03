## AI Assitant

* Chat Only

* Model: Qwen2.5-7B-Instruct

* Qunatization: using "bitsandbytes"

### Step 1

    - Transformers (simple, flexible)

        Load Qwen with transformers

        Wrap with LangChain HuggingFacePipeline

        Best for:

            Custom prompts

            Fast prototyping

### Step 2

    - vLLM (recommended for production)

        Much faster

        Better batching

        OpenAI-compatible API

        Plug into LangChain via ChatOpenAI-style interface


### Setup Enviornment

    * python -m venv venv
    * venv\Scripts\Activate
    If having any error:
    * Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted -Force
    then activate again

### Install requirments

    pip install -r requirementx.txt (CPU usage)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 (Cuda)


### Huggingface step

    use AutoTokenizer, AutoModelForCausalLM and connect them using pipeline

### Langchain
    
    use HuggigngFacePipeline to wrap the pipeline with Langchain

### Call the Model

    use invoke 

### Speed up

    pip install vllm

    python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --dtype float16 \
    --max-model-len 4096 \
    --port 8000