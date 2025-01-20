
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify model name and local save path
model_name = "meta-llama/Llama-3.2-3B-Instruct"
local_path = "/gpfs/scratch/wz1492/llama_models"

# Download the model and tokenizer to the specified path
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_path, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=local_path,
    torch_dtype=torch.bfloat16,  # Using bf16 for efficiency
    device_map="auto",          # Automatically distribute the model across devices
    use_auth_token=True         # For private models, ensure auth is provided
)

# Create a text generation pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",         # Use device mapping for optimal performance
)

# Messages for the chatbot
system_prompt = "You are a pirate chatbot who always responds in pirate speak! "
user_prompt = "Who are you?"

# Format input as a single string for causal models
formatted_input = f"{system_prompt}\nUser: {user_prompt}\nPirateBot:"

# Generate a response
outputs = pipeline(
    formatted_input,
    max_new_tokens=256,  # Maximum number of tokens to generate
)

# Print the generated response
generated_text = outputs[0]["generated_text"]
print(generated_text)