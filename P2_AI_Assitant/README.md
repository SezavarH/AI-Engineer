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
