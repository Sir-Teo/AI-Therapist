from utils.logger import logger

class ChatSession:
    def __init__(self, conversation_chain, stopping_criteria, settings):
        self.conversation_chain = conversation_chain
        self.stopping_criteria = stopping_criteria
        self.settings = settings
        self.turn_count = 0
        self.STOP_PHRASES = ["thank you", "thanks", "goodbye", "bye"]
        self.MAX_RESPONSE_LENGTH = 150  # Confirmed max length of 150 characters

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
            # Get response from conversation chain
            response = self.conversation_chain.predict(input=user_input)
            
            # Clean and limit response
            response = self._clean_response(response)
            response = self._limit_response_length(response)
            
            if self.stopping_criteria.should_stop_generation(response):
                return (
                    "Would you like to explore a different topic? I'm here to listen.",
                    False
                )
                
            self.turn_count += 1
            return response, False

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I'm having trouble. Could you try rephrasing that?", False
            
    def _clean_response(self, response):
        """Remove unwanted prefixes and formatting."""
        # Remove common prefixes and system instructions
        response = response.replace("TherapistBot:", "")
        response = response.replace("Assistant:", "")
        response = response.replace("Human:", "")
        
        # Remove any template instructions
        if "providing emotional support" in response:
            response = response.split("providing emotional support")[1]
            
        if "Current conversation:" in response:
            response = response.split("Current conversation:")[1]
            
        # Remove any generation settings
        if "Setting `pad_token_id`" in response:
            response = response.split("Setting `pad_token_id`")[0]
            
        return response.strip()
        
    def _limit_response_length(self, response):
        """Limit response length and ensure it ends properly."""
        if len(response) <= self.MAX_RESPONSE_LENGTH:
            return response
            
        # Try to cut at the last complete sentence
        shortened = response[:self.MAX_RESPONSE_LENGTH]
        last_period = shortened.rfind('.')
        if last_period > 0:
            shortened = shortened[:last_period + 1]
        return shortened.strip()