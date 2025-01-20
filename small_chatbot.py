# small_chatbot.py

import os
import logging
import warnings
from dotenv import load_dotenv
from difflib import SequenceMatcher

import torch
import transformers 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList
)

# Updated import for HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline

# Updated imports for prompts and chains
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferWindowMemory

# If your version of LangChain supports SystemMessage:
# from langchain.schema import SystemMessage

warnings.filterwarnings('ignore')  # Ignore all warnings

# ================================
# 1. Setup Logging
# ================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# 2. Environment Variables
# ================================
load_dotenv()

HF_AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN')
if HF_AUTH_TOKEN is None:
    raise ValueError("HF_AUTH_TOKEN is not set in the environment variables.")

# ================================
# 3. Model Config
# ================================
# Replace with your actual HF model name (example only)
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LOCAL_CACHE_DIR = "/gpfs/scratch/wz1492/llama_models"  # Ensure this path exists and is writable

os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

# ================================
# Custom StoppingCriteria Class
# ================================
class StopOnString(StoppingCriteria):
    """
    Stops text generation once the model attempts to output a given string token sequence.
    NOTE: This depends on how the tokenizer splits the string.
    """
    def __init__(self, stop_string: str, tokenizer: AutoTokenizer):
        super().__init__()
        self.stop_string = stop_string
        self.tokenizer = tokenizer
        # Convert the stop_string to token IDs
        self.stop_token_ids = tokenizer(stop_string, return_tensors="pt").input_ids[0]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Check if the last tokens match our `stop_token_ids`.
        """
        if input_ids.shape[1] < len(self.stop_token_ids):
            return False
        
        # Compare the end of `input_ids` to `self.stop_token_ids`
        if torch.all(input_ids[0, -len(self.stop_token_ids):] == self.stop_token_ids):
            return True
        return False

# ================================
# 4. ConversationStoppingCriteria
# ================================
class ConversationStoppingCriteria:
    def __init__(self):
        self.response_history = []
        self.MAX_RESPONSE_LENGTH = 1000  # Max characters in a response
        # Lowered threshold to catch near-duplicates instead of perfect matches
        self.SIMILARITY_THRESHOLD = 0.9  
        self.MAX_REPETITIONS = 5  # Max number of similar responses allowed
        self.CIRCULAR_WINDOW = 3  # Window size for checking circular conv

    def is_response_too_long(self, response):
        """Check if response exceeds maximum length."""
        return len(response) > self.MAX_RESPONSE_LENGTH

    def calculate_similarity(self, str1, str2):
        """Calculate similarity ratio between two strings using difflib."""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def is_response_repetitive(self, new_response):
        """
        Check if response is too similar to recent responses
        within the last self.CIRCULAR_WINDOW turns.
        """
        similar_count = 0
        for past_response in self.response_history[-self.CIRCULAR_WINDOW:]:
            similarity = self.calculate_similarity(new_response, past_response)
            if similarity > self.SIMILARITY_THRESHOLD:
                similar_count += 1
        return similar_count >= self.MAX_REPETITIONS

    def is_conversation_circular(self, new_response):
        """Check if conversation is going in circles."""
        if len(self.response_history) >= self.CIRCULAR_WINDOW:
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
# 5. Load Tokenizer and Model
# ================================
try:
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=LOCAL_CACHE_DIR,
        use_auth_token=HF_AUTH_TOKEN
        # Remove 'truncation=True' to avoid cutting long prompts silently
    )
    logger.info("Tokenizer loaded successfully.")

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=LOCAL_CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_auth_token=HF_AUTH_TOKEN
    )
    logger.info("Model loaded successfully.")

except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    raise e

# ================================
# 6. Create HF Pipeline
# ================================
try:
    logger.info("Creating HuggingFace pipeline...")

    # 1) Build the pipeline
    generation_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        # Remove explicit max_length in favor of max_new_tokens
        max_new_tokens=150,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )

    # 2) Create a stopping criteria object for "User:"
    stop_criteria = StoppingCriteriaList([StopOnString("User:", tokenizer)])

    # 3) Wrap the pipeline with HuggingFacePipeline
    llm = HuggingFacePipeline(
        pipeline=generation_pipeline,
        pipeline_kwargs={"stopping_criteria": stop_criteria}
    )

    logger.info("Pipeline created and wrapped successfully.")

except Exception as e:
    logger.error(f"Error creating pipeline: {e}")
    raise e

# ================================
# 7. Define Prompts and Chains
# ================================
try:
    logger.info("Defining system prompt and prompt template...")

    SYSTEM_PROMPT = (
        "You are a therapist chatbot who responds with empathy, understanding, and professional support. "
        "You do not provide medical diagnoses or specific medical advice. Instead, you offer general coping "
        "strategies, emotional support, and encourage the user to seek professional help if needed."
    )

    # A simpler prompt template. 
    # If you want to incorporate the system prompt into each turn, you can do so,
    # or rely on memory to store it as a 'SystemMessage'.
    prompt_template = PromptTemplate(
        template="""
{history}
User: {input}
TherapistBot:
""".strip(),
        input_variables=["history", "input"]
    )

    logger.info("Prompt template defined.")

    # Create an LLMChain for single-turn
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=True
    )

    # Create a ConversationChain for multi-turn
    memory = ConversationBufferWindowMemory(k=5)

    # If your LangChain version supports role-based messages:
    # memory.chat_memory.add_message(SystemMessage(content=SYSTEM_PROMPT))
    #
    # Otherwise, you can prepend it as an AI or user message:
    # (Not ideal, but we do it here for demonstration)
    memory.chat_memory.add_ai_message(SYSTEM_PROMPT)

    conversation_chain = ConversationChain(
        llm=llm,
        prompt=prompt_template,
        memory=memory,
        verbose=True
    )

except Exception as e:
    logger.error(f"Error integrating with LangChain: {e}")
    raise e

# ================================
# 8. Example User Prompt
# ================================
initial_user_prompt = "I'm feeling a bit anxious today."

# ================================
# 9. Generate & Print Responses
# ================================
def generate_single_turn_response():
    """
    Generates a single-turn response using LLMChain.
    """
    try:
        logger.info("Generating single-turn response...")
        # Depending on your LangChain version, you might need to call 'run' or 'invoke'
        response = llm_chain.run({
            "history": "",
            "input": initial_user_prompt
        })
        print("TherapistBot:", response)
    except Exception as e:
        logger.error(f"Error generating single-turn response: {e}")

def generate_multi_turn_response():
    """
    Multi-turn conversation using ConversationChain.
    """
    try:
        logger.info("Generating multi-turn responses...")

        # 1st turn
        response1 = conversation_chain.run(input=initial_user_prompt)
        print("User:", initial_user_prompt)
        print("TherapistBot:", response1)

        # 2nd turn
        user_prompt2 = "I have trouble sleeping at night due to racing thoughts."
        response2 = conversation_chain.run(input=user_prompt2)
        print("User:", user_prompt2)
        print("TherapistBot:", response2)

        # 3rd turn
        user_prompt3 = "Thank you. Can you suggest any relaxation techniques?"
        response3 = conversation_chain.run(input=user_prompt3)
        print("User:", user_prompt3)
        print("TherapistBot:", response3)

    except Exception as e:
        logger.error(f"Error generating multi-turn response: {e}")

# ================================
# 10. Interactive Chat
# ================================
def interactive_chat():
    """
    Interactive chat with sophisticated stopping criteria.
    """
    logger.info("Starting interactive chat with TherapistBot. Type 'exit' or 'quit' to stop.")
    print("Hello, I'm TherapistBot. I'm here to offer support. Type 'exit' or 'quit' at any time to end our session.\n")

    STOP_PHRASES = ["thank you", "thanks", "goodbye", "bye"]
    MAX_TURNS = 10
    turn_count = 0
    stopping_criteria_instance = ConversationStoppingCriteria()

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
            
            # Check custom repetition/circular criteria
            if stopping_criteria_instance.should_stop_generation(response):
                print(
                    "TherapistBot: I notice we might be covering similar ground. "
                    "Would you like to explore a different aspect of what's on your mind? "
                    "Feel free to share something new or ask about a specific concern.\n"
                )
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

# ================================
# 11. Main
# ================================
if __name__ == "__main__":
    # Uncomment any of the following to test:
    #
    # generate_single_turn_response()
    # generate_multi_turn_response()
    interactive_chat()
