from openai import OpenAI
from app.config import settings

class LLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
    
    def ask(self, question: str, context: str) -> str:
        """Simple LLM call with context"""
        prompt = f"""Based on the following member messages, answer the question.

Messages:

{context}

Question: {question}

Answer concisely and directly. If you can't find the information, say so."""

        response = self.client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about member data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        return response.choices[0].message.content

# Global instance
llm_client = LLMClient()

