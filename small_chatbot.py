# pirate_chatbot_langchain.py
from dotenv import load_dotenv
import os
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Updated import for HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline

# Updated imports for prompts and chains
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferWindowMemory

import logging

# ================================
# 1. Setup Logging
# ================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# 2. Environment Variables and Authentication
# ================================
# Load environment variables from the .env file
load_dotenv()

# Retrieve the token from the environment variables
HF_AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN')

if HF_AUTH_TOKEN is None:
    raise ValueError("HF_AUTH_TOKEN is not set in the environment variables.")

# ================================
# 3. Model and Tokenizer Configuration
# ================================
# Specify the model name and local cache path
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LOCAL_CACHE_DIR = "/gpfs/scratch/wz1492/llama_models"  # Ensure this path exists and is writable

# Create the local cache directory if it doesn't exist
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

# ================================
# 4. Load Tokenizer and Model
# ================================
try:
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=LOCAL_CACHE_DIR,
        use_auth_token=HF_AUTH_TOKEN,
        truncation=True  # Enable truncation to address the warning
    )
    logger.info("Tokenizer loaded successfully.")

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=LOCAL_CACHE_DIR,
        torch_dtype=torch.bfloat16,    # Use bfloat16 for efficiency; ensure hardware supports it
        device_map="auto",             # Automatically map model to available devices (e.g., GPUs)
        use_auth_token=HF_AUTH_TOKEN
    )
    logger.info("Model loaded successfully.")

except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    raise e

# ================================
# 5. Create HuggingFace Pipeline
# ================================
try:
    logger.info("Creating HuggingFace pipeline...")
    generation_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",               # Utilize device mapping for optimal performance
        torch_dtype=torch.bfloat16,      # Ensure dtype matches model
        max_length=256,                  # Maximum number of tokens to generate
        max_new_tokens=50,               # Limit the number of new tokens generated
        do_sample=True,                  # Enable sampling for diverse outputs
        top_p=0.95,                      # Top-p (nucleus) sampling
        temperature=0.7,                 # Temperature to control randomness
        # Removed use_auth_token=HF_AUTH_TOKEN
    )
    logger.info("Pipeline created successfully.")

except Exception as e:
    logger.error(f"Error creating pipeline: {e}")
    raise e

# ================================
# 6. Integrate with LangChain
# ================================
try:
    logger.info("Wrapping pipeline with LangChain's HuggingFacePipeline...")
    llm = HuggingFacePipeline(pipeline=generation_pipeline)
    logger.info("HuggingFacePipeline wrapped successfully.")

    # Define the prompt template
    prompt_template = PromptTemplate(
        template="{history}\nUser: {input}\nPirateBot:",
        input_variables=["history", "input"]
    )
    logger.info("Prompt template defined.")

    # Create the LLMChain (deprecated; consider updating to RunnableSequence)
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=True  # Set to True for detailed logs; set to False in production
    )
    logger.info("LLMChain created successfully.")

    # Optional: Set up a ConversationChain with memory for multi-turn conversations
    memory = ConversationBufferWindowMemory()
    conversation_chain = ConversationChain(
        llm=llm,
        prompt=prompt_template,
        memory=memory,
        verbose=True
    )
    logger.info("ConversationChain with memory created successfully.")

except Exception as e:
    logger.error(f"Error integrating with LangChain: {e}")
    raise e

# ================================
# 7. Define System and User Prompts
# ================================
system_prompt = "You are a pirate chatbot who always responds in pirate speak!"
initial_user_prompt = "Who are you?"

# ================================
# 8. Generate and Print Responses
# ================================
def generate_single_turn_response():
    """
    Generates a single-turn response using LLMChain.
    """
    try:
        logger.info("Generating single-turn response...")
        # Provide both 'history' and 'input' as expected by the prompt template
        response = llm_chain.invoke({
            "history": "",
            "input": initial_user_prompt
        })
        print("PirateBot:", response["text"])
    except Exception as e:
        logger.error(f"Error generating single-turn response: {e}")

def generate_multi_turn_response():
    """
    Engages in a multi-turn conversation using ConversationChain with memory.
    """
    try:
        logger.info("Generating multi-turn responses...")

        # 1st turn
        response1 = conversation_chain.run(initial_user_prompt)
        print("User: " + initial_user_prompt)
        print("PirateBot:", response1)

        # 2nd turn
        user_prompt2 = "Tell me a pirate joke."
        response2 = conversation_chain.run(user_prompt2)
        print("User: " + user_prompt2)
        print("PirateBot:", response2)

        # 3rd turn
        user_prompt3 = "That was great! What's the weather like today?"
        response3 = conversation_chain.run(user_prompt3)
        print("User: " + user_prompt3)
        print("PirateBot:", response3)

    except Exception as e:
        logger.error(f"Error generating multi-turn response: {e}")

def interactive_chat():
    """
    Allows the user to converse interactively with PirateBot until they type 'exit' or 'quit'.
    """
    logger.info("Starting interactive chat with PirateBot. Type 'exit' or 'quit' to stop.")
    print("Ahoy, matey! Ye be chattin' with PirateBot. Type 'exit' or 'quit' to leave our ship!\n")

    while True:
        user_input = input("User: ")
        
        if user_input.strip().lower() in ["exit", "quit"]:
            print("PirateBot: Arrr, take care on the high seas, me hearty!")
            break
        
        try:
            response = conversation_chain.run(user_input)
            print("PirateBot:", response, "\n")
        except Exception as e:
            logger.error(f"Error generating interactive response: {e}")
            print("PirateBot: Arrr, there be a squall in me code! Try again.\n")


# ================================
# 9. Main Execution
# ================================
if __name__ == "__main__":
    # Generate a single-turn response
    generate_single_turn_response()

    #print("\n--- Starting Multi-Turn Conversation ---\n")

    # Generate multi-turn responses
    #generate_multi_turn_response()

    interactive_chat()

