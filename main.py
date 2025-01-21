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
    Settings.validate()
    
    model_loader = ModelLoader(Settings)
    tokenizer, model = model_loader.load()
    
    pipeline_builder = PipelineBuilder(model, tokenizer, Settings)
    llm = pipeline_builder.build()
    
    _, conversation_chain = PromptManager.setup_chains(llm)
    stopping_criteria = ConversationStoppingCriteria(Settings)
    chat_session = ChatSession(conversation_chain, stopping_criteria, Settings)
    
    return chat_session

def main():
    try:
        chat_session = initialize_chatbot()
        
        print("\nHello! I'm here to listen and support you. What would you like to talk about today?")
        print("(Type 'exit' to end our conversation)\n")
        
        while True:
            user_input = input("You: ").strip()
            response, should_end = chat_session.process_user_input(user_input)
            
            if should_end:
                print("\nTake care. Remember that support is always available when you need it.")
                break
                
            if response:
                print(f"\nTherapist: {response}\n")
                
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
        print("I apologize, but I'm experiencing technical difficulties. Please try again later.")


if __name__ == "__main__":
    main()
