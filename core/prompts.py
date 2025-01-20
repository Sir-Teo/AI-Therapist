# core/prompts.py
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferWindowMemory

class PromptManager:
    SYSTEM_PROMPT = """
    You are a therapist chatbot who responds with empathy, understanding, and professional support.
    You do not provide medical diagnoses or specific medical advice. Instead, you offer general coping
    strategies, emotional support, and encourage the user to seek professional help if needed.
    """

    @classmethod
    def create_prompt_template(cls):
        return PromptTemplate(
            template="""
            {history}
            User: {input}
            TherapistBot:
            """.strip(),
            input_variables=["history", "input"]
        )

    @classmethod
    def setup_chains(cls, llm):
        prompt_template = cls.create_prompt_template()
        
        # Single-turn chain
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt_template,
            verbose=True
        )

        # Multi-turn chain with memory
        memory = ConversationBufferWindowMemory(k=5)
        memory.chat_memory.add_ai_message(cls.SYSTEM_PROMPT)

        conversation_chain = ConversationChain(
            llm=llm,
            prompt=prompt_template,
            memory=memory,
            verbose=True
        )

        return llm_chain, conversation_chain