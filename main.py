# main.py
import warnings
import argparse  # Import argparse for command-line argument parsing

warnings.filterwarnings('ignore')

from config.settings import Settings
from core.model import ModelLoader
from core.pipeline import PipelineBuilder
from core.prompts import PromptManager
from chat.session import ChatSession
from utils.stopping_criteria import ConversationStoppingCriteria
from utils.logger import logger

def initialize_chatbot(use_memory=True):
    Settings.validate()
    
    model_loader = ModelLoader(Settings)
    tokenizer, model = model_loader.load()
    
    pipeline_builder = PipelineBuilder(model, tokenizer, Settings)
    llm = pipeline_builder.build()
    
    _, conversation_chain = PromptManager.setup_chains(llm, use_memory=use_memory)
    stopping_criteria = ConversationStoppingCriteria(Settings)
    chat_session = ChatSession(conversation_chain, stopping_criteria, Settings)
    
    return chat_session

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Initialize the Chatbot with optional memory.")
    parser.add_argument(
        '--flog',
        action='store_true',
        help='Disable the existing LangChain LLM memory when this flag is set.'
    )
    args = parser.parse_args()

    # Determine whether to use memory based on the flog flag
    use_memory = not args.flog

    try:
        chat_session = initialize_chatbot(use_memory)
        
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
