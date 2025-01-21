# core/model.py
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.logger import logger

class ModelLoader:
    def __init__(self, settings):
        self.settings = settings
        self.tokenizer = None
        self.model = None

    def load(self):
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.settings.MODEL_NAME,
                cache_dir=self.settings.LOCAL_CACHE_DIR,
                use_auth_token=self.settings.HF_AUTH_TOKEN
            )
            
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.settings.MODEL_NAME,
                cache_dir=self.settings.LOCAL_CACHE_DIR,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                use_auth_token=self.settings.HF_AUTH_TOKEN
            )
            # Explicitly set pad_token_id to eos_token_id if not set
            if self.tokenizer.pad_token_id is None:
                logger.info("Setting `pad_token_id` to `eos_token_id`.")
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.model.config.pad_token_id = self.model.config.eos_token_id
            
            return self.tokenizer, self.model
            
        except Exception as e:
            logger.error(f"Error loading model or tokenizer: {e}")
            raise e
