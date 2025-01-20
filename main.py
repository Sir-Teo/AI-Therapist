# main.py
import warnings
warnings.filterwarnings('ignore')

from config.settings import Settings
from core.model import ModelLoader
from core.pipeline import PipelineBuilder
from core.prompts import PromptManager
from chat.session import ChatSession
from utils.stopping_criteria import ConversationStoppingCriteria
from utils.logger import logger

def initialize_chatbot():
    # Validate settings
    Settings.validate()
    
    # Load model and tokenizer
    model_loader = ModelLoader(Settings)
    tokenizer, model = model_loader.load()
    
    # Build pipeline
    pipeline_builder = PipelineBuilder(model, tokenizer, Settings)
    llm = pipeline_builder.build()
    
    # Setup chains
    _, conversation_chain = PromptManager.setup_chains(llm)
    
    # Create stopping criteria
    stopping_criteria = ConversationStoppingCriteria(Settings)
    
    # Create chat session
    chat_session = ChatSession(conversation_chain, stopping_criteria, Settings)
    
    return chat_session

def main():
    try:
        chat_session = initialize_chatbot()
        
        print("Hello, I'm TherapistBot. I'm here to offer support. Type 'exit' or 'quit' at any time to end our session.\n")
        
        while True:
            user_input = input("User: ")
            response, should_end = chat_session.process_user_input(user_input)
            
            if should_end:
                print("TherapistBot: Thank you for sharing. Take care!")
                break
                
            if response:
                print("TherapistBot:", response)
                
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print("An error occurred. Please try again later.")

if __name__ == "__main__":
    main()