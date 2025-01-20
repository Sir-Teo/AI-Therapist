# therapist_chatbot_langchain.py
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
import warnings
warnings.filterwarnings('ignore')  # Ignore all warnings

class ConversationStoppingCriteria:
    def __init__(self):
        self.response_history = []
        self.MAX_RESPONSE_LENGTH = 1000  # Maximum characters in a response
        self.SIMILARITY_THRESHOLD = 1  # Threshold for detecting similar responses
        self.MAX_REPETITIONS = 5  # Maximum number of similar responses allowed
        self.CIRCULAR_WINDOW = 3  # Window size for checking circular conversations
        
    def is_response_too_long(self, response):
        """Check if response exceeds maximum length."""
        return len(response) > self.MAX_RESPONSE_LENGTH

    def calculate_similarity(self, str1, str2):
        """Calculate similarity ratio between two strings using difflib."""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def is_response_repetitive(self, new_response):
        """Check if response is too similar to recent responses."""
        similar_count = 0
        
        for past_response in self.response_history[-self.CIRCULAR_WINDOW:]:
            similarity = self.calculate_similarity(new_response, past_response)
            if similarity > self.SIMILARITY_THRESHOLD:
                similar_count += 1
                
        return similar_count >= self.MAX_REPETITIONS

    def is_conversation_circular(self, new_response):
        """Check if conversation is going in circles."""
        if len(self.response_history) >= self.CIRCULAR_WINDOW:
            # Check if new response is similar to responses from n turns ago
            old_response = self.response_history[-self.CIRCULAR_WINDOW]
            return self.calculate_similarity(new_response, old_response) > self.SIMILARITY_THRESHOLD
        return False

    def should_stop_generation(self, new_response):
        """Main method to check if generation should stop."""
        if self.is_response_too_long(new_response):
            logger.warning("Response exceeded maximum length")
            return True
            
        if self.is_response_repetitive(new_response):
            logger.warning("Detected repetitive responses")
            return True
            
        if self.is_conversation_circular(new_response):
            logger.warning("Detected circular conversation")
            return True
            
        # If no stopping criteria met, add response to history
        self.response_history.append(new_response)
        return False


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
    # Modify the generation pipeline configuration
    generation_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        max_length=250,
        max_new_tokens=50,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.2,  # Add repetition penalty
        no_repeat_ngram_size=3,  # Prevent repetition of 3-grams
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

    # Define the prompt template with the system prompt embedded directly
    system_prompt = (
        "You are a therapist chatbot who responds with empathy, understanding, and professional support. "
        "You do not provide medical diagnoses or specific medical advice. Instead, you offer general coping strategies, "
        "emotional support, and encourage the user to seek professional help if needed."
    )

    prompt_template = PromptTemplate(
        template=f"{system_prompt}\n\n{{history}}\nUser: {{input}}\nTherapistBot:",
        input_variables=["history", "input"]
    )
    logger.info("Prompt template defined.")

    # Create the LLMChain
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=True  # Set to True for detailed logs; set to False in production
    )
    logger.info("LLMChain created successfully.")

    # Set up a ConversationChain with memory for multi-turn conversations
    # Set window_size to limit the number of past interactions stored
    memory = ConversationBufferWindowMemory(window_size=5)  # Adjust window_size as needed
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
# (System prompt is already embedded in the prompt template)
initial_user_prompt = "I'm feeling a bit anxious today."

# ================================
# 8. Generate and Print Responses
# ================================
def generate_single_turn_response():
    """
    Generates a single-turn response using LLMChain.
    """
    try:
        logger.info("Generating single-turn response...")
        # Provide 'history' and 'input' as expected by the prompt template
        response = llm_chain.invoke({
            "history": "",
            "input": initial_user_prompt
        })
        print("TherapistBot:", response["text"])
    except Exception as e:
        logger.error(f"Error generating single-turn response: {e}")

def generate_multi_turn_response():
    """
    Engages in a multi-turn conversation using ConversationChain with memory.
    """
    try:
        logger.info("Generating multi-turn responses...")

        # 1st turn
        response1 = conversation_chain.run(input=initial_user_prompt)
        print("User: " + initial_user_prompt)
        print("TherapistBot:", response1)

        # 2nd turn
        user_prompt2 = "I have trouble sleeping at night due to racing thoughts."
        response2 = conversation_chain.run(input=user_prompt2)
        print("User: " + user_prompt2)
        print("TherapistBot:", response2)

        # 3rd turn
        user_prompt3 = "Thank you. Can you suggest any relaxation techniques?"
        response3 = conversation_chain.run(input=user_prompt3)
        print("User: " + user_prompt3)
        print("TherapistBot:", response3)

    except Exception as e:
        logger.error(f"Error generating multi-turn response: {e}")

# Modify the interactive_chat function to use the stopping criteria
def interactive_chat():
    """
    Enhanced interactive chat with sophisticated stopping criteria.
    """
    logger.info("Starting interactive chat with TherapistBot. Type 'exit' or 'quit' to stop.")
    print("Hello, I'm TherapistBot. I'm here to offer support. Type 'exit' or 'quit' at any time to end our session.\n")

    STOP_PHRASES = ["thank you", "thanks", "goodbye", "bye"]
    MAX_TURNS = 10
    turn_count = 0
    stopping_criteria = ConversationStoppingCriteria()

    while turn_count < MAX_TURNS:
        user_input = input("User: ")

        if user_input.strip().lower() in ["exit", "quit"]:
            print("TherapistBot: Thank you for sharing. Remember, professional help is always there if you need it. Take care.")
            break

        if any(phrase in user_input.lower() for phrase in STOP_PHRASES):
            print("TherapistBot: You're welcome! Feel free to reach out if you need more support. Take care!")
            break

        try:
            response = conversation_chain.run(input=user_input)
            
            # Check the response quality but don't exit
            if stopping_criteria.should_stop_generation(response):
                # Instead of the generated response, provide a redirect prompt
                print("TherapistBot: I notice we might be covering similar ground. "
                      "Would you like to explore a different aspect of what's on your mind? "
                      "Feel free to share something new or ask about a specific concern.\n")
            else:
                print("TherapistBot:", response, "\n")
            turn_count += 1

            if turn_count >= MAX_TURNS:
                print("TherapistBot: It seems we've reached our conversation limit for today. "
                      "Remember, seeking professional help is always beneficial. Take care!")
                break

        except Exception as e:
            logger.error(f"Error generating interactive response: {e}")
            print("TherapistBot: I'm sorry, I'm having trouble understanding. Please try again.\n")

    if turn_count >= MAX_TURNS:
        logger.info("Maximum number of turns reached. Ending session.")


# 9. Main Execution
# ================================
if __name__ == "__main__":
    # Uncomment the following lines to test different functionalities

    # Generate a single-turn response
    # generate_single_turn_response()

    # print("\n--- Starting Multi-Turn Conversation ---\n")
    # Generate multi-turn responses
    # generate_multi_turn_response()

    # Start interactive chat with stopping criteria
    interactive_chat()
