# Multimodal AI Chatbot

A production-ready multimodal chatbot powered by OpenAI APIs with a Gradio web interface. Supports text chat, text-to-speech, and image generation.

## Features

- **Streaming chat** with GPT-4.1-mini
- **Text-to-speech** with gpt-4o-mini-tts (toggle on/off)
- **Image generation** with DALL-E 3 (toggle on/off)
- **Session-based memory** — conversation history persists within a session
- Deployable to **Hugging Face Spaces** (free tier)

## Project Structure

```
app.py           # Gradio UI and orchestration
chat.py          # Streaming chat completions
tts.py           # Text-to-speech generation
image_gen.py     # DALL-E 3 image generation
requirements.txt # Python dependencies
.env.example     # Environment variable template
```

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd multimodal-chatbot
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-actual-key
```

### 3. Install dependencies

**With pip:**

```bash
pip install -r requirements.txt
```

**With uv:**

```bash
uv sync
```

### 4. Run locally

```bash
python app.py
```

The app will open at `http://127.0.0.1:7860`.

## Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Gradio** as the SDK
3. Push your code to the Space repository
4. Go to **Settings → Secrets** and add `OPENAI_API_KEY` with your API key
5. The Space will build and launch automatically

> **Note:** Do not push your `.env` file. HF Spaces uses Secrets for environment variables.

## Usage

1. Type a message and press **Send** or hit Enter
2. Toggle **Enable Voice Response** to hear the assistant's reply as audio
3. Toggle **Enable Image Generation** to create an image based on your prompt
4. Click **Clear Conversation** to reset the chat
