# chat/session.py
from utils.logger import logger

class ChatSession:
    def __init__(self, conversation_chain, stopping_criteria, settings):
        self.conversation_chain = conversation_chain
        self.stopping_criteria = stopping_criteria
        self.settings = settings
        self.turn_count = 0
        self.STOP_PHRASES = ["thank you", "thanks", "goodbye", "bye"]

    def _should_end_session(self, user_input):
        if user_input.strip().lower() in ["exit", "quit"]:
            return True
        if any(phrase in user_input.lower() for phrase in self.STOP_PHRASES):
            return True
        if self.turn_count >= self.settings.MAX_TURNS:
            return True
        return False

    def process_user_input(self, user_input):
        if self._should_end_session(user_input):
            return None, True

        try:
            response = self.conversation_chain.run(input=user_input)
            
            if self.stopping_criteria.should_stop_generation(response):
                return (
                    "I notice we might be covering similar ground. "
                    "Would you like to explore a different aspect of what's on your mind? "
                    "Feel free to share something new or ask about a specific concern.\n"
                ), False
                
            self.turn_count += 1
            return response, False

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I'm having trouble understanding. Please try again.\n", False
