# utils/stopping_criteria.py
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from difflib import SequenceMatcher
from utils.logger import logger

class StopOnString(StoppingCriteria):
    def __init__(self, stop_string: str, tokenizer):
        super().__init__()
        self.stop_string = stop_string
        self.tokenizer = tokenizer
        self.stop_token_ids = tokenizer(stop_string, return_tensors="pt").input_ids[0]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[1] < len(self.stop_token_ids):
            return False
        return torch.all(input_ids[0, -len(self.stop_token_ids):] == self.stop_token_ids)

class ConversationStoppingCriteria:
    def __init__(self, settings):
        self.response_history = []
        self.settings = settings

    def calculate_similarity(self, str1, str2):
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def is_response_too_long(self, response):
        return len(response) > self.settings.MAX_RESPONSE_LENGTH

    def is_response_repetitive(self, new_response):
        similar_count = 0
        for past_response in self.response_history[-self.settings.CIRCULAR_WINDOW:]:
            similarity = self.calculate_similarity(new_response, past_response)
            if similarity > self.settings.SIMILARITY_THRESHOLD:
                similar_count += 1
        return similar_count >= self.settings.MAX_REPETITIONS

    def is_conversation_circular(self, new_response):
        if len(self.response_history) >= self.settings.CIRCULAR_WINDOW:
            old_response = self.response_history[-self.settings.CIRCULAR_WINDOW]
            return self.calculate_similarity(new_response, old_response) > self.settings.SIMILARITY_THRESHOLD
        return False

    def should_stop_generation(self, new_response):
        if self.is_response_too_long(new_response):
            logger.warning("Response exceeded maximum length")
            return True
        if self.is_response_repetitive(new_response):
            logger.warning("Detected repetitive responses")
            return True
        if self.is_conversation_circular(new_response):
            logger.warning("Detected circular conversation")
            return True
        
        self.response_history.append(new_response)
        return False
