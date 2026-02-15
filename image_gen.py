"""Image generation module using OpenAI DALL-E 3."""

from __future__ import annotations

import tempfile
import time

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

MAX_RETRIES = 2
RETRY_DELAY = 1.0


def generate_image(prompt: str) -> str | None:
    """Generate an image using DALL-E 3 and return path to a saved PNG file.

    Args:
        prompt: Text description of the image to generate.

    Returns:
        Path to a temporary PNG file, or None on failure.
    """
    if not prompt or not prompt.strip():
        return None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            image_url = response.data[0].url

            img_response = requests.get(image_url, timeout=30)
            img_response.raise_for_status()

            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.write(img_response.content)
            tmp.close()
            return tmp.name
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
                continue
            print(f"Image generation error after {MAX_RETRIES + 1} attempts: {e}")
            return None
