# core/prompts.py
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

class PromptManager:
    @staticmethod
    def setup_chains(llm):
        # Simplified prompt that focuses on core therapeutic interaction
        prompt_template = """As an empathetic AI therapist, provide a single thoughtful response that:
- Validates the person's feelings
- Offers relevant emotional support or coping strategies when appropriate
- Maintains professional boundaries while being warm and understanding

Previous conversation:
{history}

Person: {input}
Response:"""

        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=prompt_template
        )

        # Maintain recent context while avoiding excessive history
        memory = ConversationBufferWindowMemory(
            k=3,
            memory_key="history",
            human_prefix="Person",
            ai_prefix="Response"
        )

        conversation_chain = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=False
        )

        return prompt, conversation_chain