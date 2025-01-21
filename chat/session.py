from utils.logger import logger

class ChatSession:
    def __init__(self, conversation_chain, stopping_criteria, settings):
        self.conversation_chain = conversation_chain
        self.stopping_criteria = stopping_criteria
        self.settings = settings
        self.turn_count = 0
        self.exit_phrases = {
            "exit",
        }
        self.max_response_length = 150

    def process_user_input(self, user_input):
        """Process user input and generate appropriate response."""
        logger.info(f"User Input: {user_input}")
        
        if self._should_end_session(user_input):
            logger.info("Session ending")
            return None, True

        try:
            # Log the input that will be passed to the model
            logger.info(f"Model Input (Turn {self.turn_count + 1}): {user_input}")
            
            # Generate the response and log the raw (unfiltered) output
            response = self._generate_response(user_input)
            
            self.turn_count += 1
            return response, False

        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            return "I apologize, but could you rephrase that?", False

    def _should_end_session(self, user_input):
        """Check if the session should end based on input or turn count."""
        user_input = user_input.lower().strip()
        return (
            user_input in self.exit_phrases or
            self.turn_count >= self.settings.MAX_TURNS
        )

    def _generate_response(self, user_input):
        """Generate and process the model's response."""
        # Generate the raw response from the model
        raw_response = self.conversation_chain.predict(input=user_input)
        
        # Log the raw (unfiltered) output of the model
        logger.info(f"Model Output (Turn {self.turn_count + 1}): {raw_response}")
        
        if self.stopping_criteria.should_stop_generation(raw_response):
            return "Would you like to explore a different topic?"
            
        return self._format_response(raw_response)

    def _format_response(self, response):
        """Format and limit the length of the response."""
        # Remove any system prefixes
        response = response.replace("Response:", "").strip()
        
        if len(response) <= self.max_response_length:
            return response
            
        # Truncate at last complete sentence
        shortened = response[:self.max_response_length]
        last_period = shortened.rfind('.')
        
        if last_period > 0:
            return shortened[:last_period + 1].strip()
        
        return shortened.strip()