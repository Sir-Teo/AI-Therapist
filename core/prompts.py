# core/prompts.py

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory

class PromptManager:
    @staticmethod
    def setup_chains(llm, use_memory=True):
        system_instructions = (
    "You are an empathetic therapist. You must respond to \"Human\" in exactly one single sentence, "
    "and continue the conversation. Prepend each response with \"Therapist: \".\n"
    "---\n"
    "Example:\n"
    "Human: I feel overwhelmed with work and life.\n"
    "Therapist: It sounds like you're juggling a lot—remember to take small steps to care for yourself.\n"
    "---"
)

        if use_memory:
            memory = ConversationBufferWindowMemory(k=3, return_messages=True)
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", system_instructions),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}\n")
            ])
        else:
            memory = None
            chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_instructions),
            ("human", "{input}\n")
            ])

        chain = LLMChain(llm=llm, prompt=chat_prompt, memory=memory, verbose=False)
        return chain
