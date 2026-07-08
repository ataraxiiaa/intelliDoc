import os
import json
import logging
from openai import OpenAI
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class Answer(BaseModel):
    answer: str
    confidence: float
    sources: List[str]
    tokens_used: int


class Synthesizer:
    def __init__(self, model_name="openrouter/free"):
        # Auto-load variables from .local.env at the project root if it exists
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_env_path = os.path.join(base_dir, "..", ".local.env")
        if os.path.exists(local_env_path):
            try:
                with open(local_env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ[k.strip()] = v.strip()
            except Exception as e:
                logger.warning(f"Could not load .local.env: {e}")

        # Fetch OpenRouter API key from env, falling back to HF_TOKEN or mock-key
        api_key = os.environ.get("OPENROUTER_KEY") or os.environ.get("HF_TOKEN") or "mock-key"
        base_url = "https://openrouter.ai/api/v1"
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/ataraxiiaa/intelliDoc",
                "X-Title": "IntelliDoc IDP Pipeline"
            }
        )
        self.model_name = model_name
        # Tiktoken only supports OpenAI models. Use Hugging Face AutoTokenizer for Qwen/Gemini estimation.
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a string using Qwen tokenizer."""
        return len(self.tokenizer.encode(text))

    def build_prompt(self, question: str, context: List[Dict], history: Optional[List[Dict]] = None) -> str:
        cites_text = ""
        for i, doc in enumerate(context, 1):
            cites_text += f"[{i}] {doc['text']}\n"

        # Format chat history
        history_text = ""
        if history:
            for msg in history:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_text += f"{role}: {msg['content']}\n"

        prompt = f"""You are an AI assistant.
Use the following context documents and chat history to structure the answer.
Cite the document numbers in square brackets immediately after using the information.

Chat History:
{history_text if history_text else "No prior history."}

Context documents:
{cites_text}

Question: {question}

Return the answer in the following format (JSON only):
{{
  "answer": "your cited answer text here...",
  "confidence": 0.95,
  "sources": ["1", "2"],
  "tokens_used": 123
}}"""
        return prompt

    def answer_question(self, question: str, context: List[Dict], history: Optional[List[Dict]] = None) -> Answer:
        prompt = self.build_prompt(question, context, history)
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]

        try:
            chat_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"}
            )
            raw_content = chat_response.choices[0].message.content or ""
            
            # Robust JSON parsing with fallback
            try:
                data = json.loads(raw_content)
            except json.JSONDecodeError:
                # Fallback: treat the raw response string as the answer text directly
                data = {
                    "answer": raw_content.strip(),
                    "confidence": 0.7,
                    "sources": []
                }
            
            # Extract tokens_used from the API usage metadata if the model didn't fill it
            tokens_used = data.get("tokens_used", 0)
            if tokens_used == 0 or tokens_used == 123:  # 123 is the placeholder from the prompt
                if chat_response.usage:
                    tokens_used = chat_response.usage.total_tokens

            return Answer(
                answer=data.get("answer", ""),
                confidence=float(data.get("confidence", 0.0) or 0.0),
                sources=data.get("sources", []),
                tokens_used=tokens_used
            )

        except Exception as e:
            logger.error(f"Failed to synthesize answer: {e}")
            # Return a graceful fallback Answer
            return Answer(
                answer=f"Error: {e}",
                confidence=0.0,
                sources=[],
                tokens_used=0
            )


if __name__ == "__main__":
    synthesizer = Synthesizer()
    print("Synthesizer initialized successfully.")
