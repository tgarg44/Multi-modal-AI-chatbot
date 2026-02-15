"""Multimodal chatbot Gradio application."""

from __future__ import annotations

import gradio as gr

from chat import chat_stream
from image_gen import generate_image
from tts import text_to_speech


def respond(
    user_message: str,
    history: list[dict],
    enable_voice: bool,
    enable_image: bool,
):
    """Handle the full response lifecycle as a generator.

    1. Stream assistant text response chunk-by-chunk.
    2. Optionally generate TTS audio after text completes.
    3. Optionally generate an image after text completes.

    Yields:
        Tuples of (history, audio_path | None, image_path | None).
    """
    if not user_message or not user_message.strip():
        yield history, None, None
        return

    # Append user message (immutable update)
    history = history + [{"role": "user", "content": user_message}]

    # Placeholder for assistant response
    history = history + [{"role": "assistant", "content": ""}]

    # Stream chat response — pass history without the empty assistant placeholder
    full_response = ""
    for chunk in chat_stream(history[:-1]):
        full_response += chunk
        history[-1] = {"role": "assistant", "content": full_response}
        yield history, None, None

    # Post-processing: TTS
    audio_path = None
    if enable_voice and full_response:
        audio_path = text_to_speech(full_response)

    # Post-processing: Image generation
    image_path = None
    if enable_image and full_response:
        image_path = generate_image(user_message)

    # Final yield with all outputs
    if audio_path or image_path:
        yield history, audio_path, image_path


def on_submit(
    user_message: str,
    history: list[dict],
    enable_voice: bool,
    enable_image: bool,
):
    """Wrapper generator that maps respond() outputs to Gradio components.

    Yields:
        (chatbot, state, audio_out, image_out, textbox)
    """
    for updated_history, audio, image in respond(
        user_message, history, enable_voice, enable_image
    ):
        yield updated_history, updated_history, audio, image, ""


def clear_all():
    """Reset all outputs."""
    return [], [], None, None


def create_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks interface."""
    with gr.Blocks(
        title="Multimodal Chatbot",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# Multimodal AI Chatbot\n"
            "Chat with GPT-4.1-mini. Optionally enable voice responses "
            "and image generation."
        )

        # Session-scoped chat history
        state = gr.State(value=[])

        with gr.Row():
            # -- Main chat area --
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    type="messages",
                    height=500,
                    show_copy_button=True,
                )
                with gr.Row():
                    txt = gr.Textbox(
                        placeholder="Type your message...",
                        show_label=False,
                        scale=4,
                        container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

            # -- Sidebar controls --
            with gr.Column(scale=1):
                voice_cb = gr.Checkbox(
                    label="Enable Voice Response", value=False
                )
                image_cb = gr.Checkbox(
                    label="Enable Image Generation", value=False
                )
                audio_out = gr.Audio(
                    label="Voice Response",
                    type="filepath",
                    interactive=False,
                )
                image_out = gr.Image(
                    label="Generated Image",
                    type="filepath",
                    interactive=False,
                )
                clear_btn = gr.Button(
                    "Clear Conversation", variant="secondary"
                )

        # -- Event wiring --
        submit_inputs = [txt, state, voice_cb, image_cb]
        submit_outputs = [chatbot, state, audio_out, image_out, txt]

        txt.submit(fn=on_submit, inputs=submit_inputs, outputs=submit_outputs)
        send_btn.click(
            fn=on_submit, inputs=submit_inputs, outputs=submit_outputs
        )
        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[chatbot, state, audio_out, image_out],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch()
