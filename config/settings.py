# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Model settings
    MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    LOCAL_CACHE_DIR = "/gpfs/scratch/wz1492/llama_models"
    HF_AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN')
    
    # Generation settings
    MAX_NEW_TOKENS = 150
    TEMPERATURE = 0.7
    TOP_P = 0.9
    REPETITION_PENALTY = 1.2
    NO_REPEAT_NGRAM_SIZE = 3
    
    # Conversation settings
    MAX_RESPONSE_LENGTH = 1000
    SIMILARITY_THRESHOLD = 0.9
    MAX_REPETITIONS = 5
    CIRCULAR_WINDOW = 3
    MAX_TURNS = 10
    
    # Validate settings
    @classmethod
    def validate(cls):
        if cls.HF_AUTH_TOKEN is None:
            raise ValueError("HF_AUTH_TOKEN is not set in environment variables")
        os.makedirs(cls.LOCAL_CACHE_DIR, exist_ok=True)