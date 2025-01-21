# core/prompts.py
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

class PromptManager:
    @staticmethod
    def setup_chains(llm):
        # Define a cleaner prompt template that doesn't repeat the context
        prompt_template = """You are an empathetic AI therapist providing emotional support and coping strategies. Remember to:
- Listen actively and validate feelings
- Ask thoughtful questions to understand better
- Offer practical coping strategies when appropriate
- Encourage professional help if needed
- Never diagnose or give medical advice

Current conversation:
{history}
Human: {input}
Assistant: """

        prompt = PromptTemplate(
            input_variables=["history", "input"], 
            template=prompt_template
        )

        # Setup memory with a window of recent messages
        memory = ConversationBufferWindowMemory(k=5)

        # Create conversation chain
        conversation_chain = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=False  # Set to False to avoid printing debug info
        )

        return prompt, conversation_chain