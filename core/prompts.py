# core/prompts.py
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

class PromptManager:
    @staticmethod
    def setup_chains(llm, use_memory=True):
        # Prompt template when memory is enabled
        prompt_with_memory = """As an empathetic AI therapist, provide a single thoughtful response that:
- Validates the person's feelings
- Offers relevant emotional support or coping strategies when appropriate
- Maintains professional boundaries while being warm and understanding

Previous conversation:
{history}

Person: {input}
Response:"""

        # Prompt template when memory is disabled
        prompt_without_memory = """As an empathetic AI therapist, provide a single thoughtful response that:
- Validates the person's feelings
- Offers relevant emotional support or coping strategies when appropriate
- Maintains professional boundaries while being warm and understanding

Person: {input}
Response:"""

        # Choose the appropriate prompt based on the use_memory flag
        if use_memory:
            prompt = PromptTemplate(
                input_variables=["history", "input"],
                template=prompt_with_memory
            )
            # Maintain recent context while avoiding excessive history
            memory = ConversationBufferWindowMemory(
                k=3,
                memory_key="history",
                human_prefix="Person",
                ai_prefix="Response"
            )
        else:
            prompt = PromptTemplate(
                input_variables=["input"],
                template=prompt_without_memory
            )
            # Disable memory by setting it to None
            memory = None

        conversation_chain = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=False
        )

        return prompt, conversation_chain
