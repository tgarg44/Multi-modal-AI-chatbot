"""Text-to-speech module using OpenAI gpt-4o-mini-tts."""

from __future__ import annotations

import tempfile
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

MAX_RETRIES = 2
RETRY_DELAY = 1.0
MAX_INPUT_CHARS = 4096


def text_to_speech(text: str, voice: str = "alloy") -> str | None:
    """Convert text to speech and return path to an MP3 file.

    Args:
        text: The text to convert to speech.
        voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer).

    Returns:
        Path to a temporary MP3 file, or None on failure.
    """
    if not text or not text.strip():
        return None

    # Truncate to limit cost and latency
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS] + "..."

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=text,
                response_format="mp3",
            )
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp.write(response.content)
            tmp.close()
            return tmp.name
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
                continue
            print(f"TTS error after {MAX_RETRIES + 1} attempts: {e}")
            return None
