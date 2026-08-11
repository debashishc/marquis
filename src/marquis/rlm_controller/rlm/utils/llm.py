from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class OpenAIClient:
    def __init__(self, api_key: str | None = None, model: str = "gpt-5"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def completion(
        self, messages: list[dict[str, str]] | str, max_tokens: int | None = None, **kwargs
    ) -> str:
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            response = self.client.chat.completions.create(
                model=self.model, messages=messages, max_completion_tokens=max_tokens, **kwargs
            )
            return response.choices[0].message.content or ""

        except Exception as exc:
            raise RuntimeError(f"Error generating completion: {str(exc)}") from exc
