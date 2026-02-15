"""Chat completion module using OpenAI gpt-4.1-mini with streaming."""

from __future__ import annotations

from collections.abc import Generator

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a helpful multimodal assistant. You provide clear, "
        "concise responses. When the user asks you to generate an image, "
        "acknowledge it — the system will handle image generation separately."
    ),
}


def chat_stream(history: list[dict]) -> Generator[str, None, None]:
    """Yield text chunks from gpt-4.1-mini in a streaming fashion.

    Args:
        history: Conversation history as a list of
                 {"role": "user"|"assistant", "content": "..."} dicts.

    Yields:
        Individual text chunks as they arrive from the API.
        On error, yields a single user-friendly error message.
    """
    messages = [SYSTEM_MESSAGE] + history

    try:
        stream = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content is not None:
                yield delta.content
    except Exception as e:
        yield f"\n\n[Error: {_friendly_error(e)}]"


def _friendly_error(e: Exception) -> str:
    """Convert exceptions to user-friendly messages."""
    name = type(e).__name__
    if "RateLimit" in name:
        return "Rate limit reached. Please wait a moment and try again."
    if "APIConnection" in name:
        return "Could not connect to OpenAI. Check your internet connection."
    if "Timeout" in name:
        return "Request timed out. Please try again."
    if "Authentication" in name:
        return "Invalid API key. Please check your OPENAI_API_KEY."
    return f"Something went wrong ({name}). Please try again."
