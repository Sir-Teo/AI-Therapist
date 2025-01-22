
# chat/session.py

import re
from utils.logger import logger

class ChatSession:
    def __init__(self, conversation_chain, stopping_criteria, settings):
        self.conversation_chain = conversation_chain
        self.stopping_criteria = stopping_criteria
        self.settings = settings
        self.turn_count = 0
        self.exit_phrases = {"exit", "bye", "quit"}
        self.max_response_length = 150

    def process_user_input(self, user_input):
        logger.info(f"User Input: {user_input}")
        
        # Check if session should end
        if self._should_end_session(user_input):
            logger.info("Session ending.")
            return None, True

        try:
            # Generate the model's response
            response = self._generate_response(user_input)
            self.turn_count += 1
            return response, False

        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            return "I apologize, but could you rephrase that?", False

    def _should_end_session(self, user_input):
        user_input = user_input.lower().strip()
        return (
            user_input in self.exit_phrases
            or self.turn_count >= self.settings.MAX_TURNS
        )

    def _generate_response(self, user_input):
        raw_response = self.conversation_chain.predict(input=user_input)
        logger.info(f"Model Output (Turn {self.turn_count + 1}): {raw_response}")

        # Optional: apply your stopping criteria
        if self.stopping_criteria.should_stop_generation(raw_response):
            return "Would you like to explore a different topic?"
        
        return self._format_response(raw_response)

    def _format_response(self, response: str) -> str:
        """
        1. Remove or replace any extraneous prefixes.
        2. Ensure only one sentence is returned.
        3. Prefix with 'response:'.
        """
        # Clean up any known role labeling from the LLM
        response = response.replace("System:", "").replace("Human:", "")
        response = response.replace("Response:", "").strip()

        # Use a regex to capture the first sentence (up to the first period).
        match = re.search(r'([^.]*\.)', response)
        if match:
            # Take everything up to and including the first '.'
            single_sentence = match.group(1).strip()
        else:
            # If there's no period at all, treat the entire response as one sentence
            single_sentence = response.strip()

        # Now we can prefix the single sentence with "response:"
        final_response = f"response: {single_sentence}"

        # (Optional) If you'd like to further limit total length:
        if len(final_response) > self.max_response_length:
            final_response = final_response[:self.max_response_length].rstrip()


        return final_response