# main.py
import warnings
import argparse

warnings.filterwarnings('ignore')

from config.settings import Settings
from core.model import ModelLoader
from core.pipeline import PipelineBuilder
from core.prompts import PromptManager
from chat.session import ChatSession
from utils.stopping_criteria import ConversationStoppingCriteria
from utils.logger import logger

def initialize_chatbot(use_memory=True):
    # Validate your config or environment
    Settings.validate()
    
    # 1. Load model + tokenizer
    model_loader = ModelLoader(Settings)
    tokenizer, model = model_loader.load()
    
    # 2. Build an LLM pipeline (depends on your code in core/pipeline.py)
    pipeline_builder = PipelineBuilder(model, tokenizer, Settings)
    llm = pipeline_builder.build()
    
    # 3. Get the LLMChain from PromptManager
    conversation_chain = PromptManager.setup_chains(llm, use_memory=use_memory)
    
    # 4. Stopping criteria (optional custom logic for limiting responses)
    stopping_criteria = ConversationStoppingCriteria(Settings)
    
    # 5. Create the chat session
    chat_session = ChatSession(conversation_chain, stopping_criteria, Settings)
    
    return chat_session

def main():
    parser = argparse.ArgumentParser(description="Initialize the Chatbot with optional memory.")
    parser.add_argument(
        '--flog',
        action='store_true',
        help='Disable the existing LangChain LLM memory when this flag is set.'
    )
    args = parser.parse_args()

    # If user passes --flog, memory is disabled
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
                print(f"{response}\n")
                
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
        print("I apologize, but I'm experiencing technical difficulties. Please try again later.")


if __name__ == "__main__":
    main()
