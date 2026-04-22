"""
Little Fig Studio — Main UI (v0.4)
Four tabs: Chat · Dataset Builder · Eval · Merge

Fixed for Gradio 4.x+ (messages format, not tuples).
Uses Fig Engine for inference (no GGUF dependency).
Supports any HuggingFace model ID.
"""

import gradio as gr
import os
import time
import json
from typing import Optional, Iterator


# ── Suggested models (user can type any model ID) ────────────────────────────

SUGGESTED_MODELS = [
    "GPT-2 124M (testing, fast)",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-2",
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
]

# ── Global model state ────────────────────────────────────────────────────────

_loaded_model = None
_loaded_model_id = None


def _resolve_model_id(model_name: str) -> str:
    """Resolve display name to HF model ID."""
    if model_name.startswith("GPT-2"):
        return "gpt2"
    # If it looks like a HF model ID (contains /), use as-is
    if "/" in model_name or model_name in ("gpt2",):
        return model_name
    return model_name


def _load_model(model_name: str, hw: dict):
    """Load model, caching to avoid reloads."""
    global _loaded_model, _loaded_model_id

    model_id = _resolve_model_id(model_name)

    if _loaded_model_id == model_id and _loaded_model is not None:
        return _loaded_model

    from little_fig.model import FigLanguageModel
    model = FigLanguageModel.from_pretrained(model_id, hw=hw)

    _loaded_model = model
    _loaded_model_id = model_id
    return model


def get_current_model():
    """Returns loaded model or None. Used by eval tab."""
    return _loaded_model


# ── Chat logic (Gradio 4.x messages format) ──────────────────────────────────

def respond(message: str, history: list, model_name: str,
            max_tokens: int, temperature: float, system_prompt: str, hw: dict):
    """
    Chat response generator. Compatible with Gradio 4.x ChatInterface.

    history: list of {"role": str, "content": str} dicts (messages format)
    """
    if not message.strip():
        yield ""
        return

    try:
        model = _load_model(model_name, hw)
    except Exception as e:
        yield f"❌ Model load error:\n\n{e}"
        return

    model.config.max_new_tokens = max_tokens
    model.config.temperature = temperature

    # Build prompt from history (already in messages format)
    prompt = model.apply_chat_template(message, history)

    if system_prompt.strip():
        prompt = f"System: {system_prompt}\n\n" + prompt

    start = time.time()
    full = ""
    try:
        for chunk in model.stream(prompt):
            full += chunk
            yield full
        elapsed = time.time() - start
        print(f"   ✓ ~{len(full.split())} words in {elapsed:.1f}s")
    except Exception as e:
        yield f"{full}\n\n❌ {e}"


# ── UI ────────────────────────────────────────────────────────────────────────

def run_studio(hw: Optional[dict] = None):
    if hw is None:
        from little_fig import HW
        hw = HW

    if hw.get("gpu_available"):
        hw_line = f"⚡ GPU: {hw['gpu_name']} ({hw['gpu_vram_gb']}GB)"
        hw_color = "#4ade80"
    else:
        hw_line = f"💻 CPU · {hw['cpu_cores']} cores · {hw.get('ram_available_gb', '?')}GB RAM available"
        hw_color = "#fb923c"

    theme = gr.themes.Default(
        primary_hue="green",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
    )
    css = f"""
        .fig-header {{ text-align:center; padding:1.2rem 0 0.4rem; }}
        .fig-header h1 {{ font-size:1.8rem; font-weight:700; color:#4ade80; margin:0; }}
        .fig-header p {{ color:#94a3b8; font-size:0.82rem; margin:0.2rem 0 0; }}
        .hw-badge {{
            background:#0f172a; border:1px solid #1e293b; border-radius:6px;
            padding:7px 12px; font-family:monospace; font-size:0.75rem; color:{hw_color};
        }}
        footer {{ display:none !important; }}
    """

    with gr.Blocks(title="🍐 Little Fig", theme=theme, css=css) as demo:

        gr.HTML("""
        <div class="fig-header">
            <h1>🍐 Little Fig</h1>
            <p>CPU-native LLM engine &nbsp;·&nbsp; Powered by Fig Engine &nbsp;·&nbsp; v0.4.0</p>
        </div>
        """)

        # ── Tab 1: Chat ───────────────────────────────────────────────────────
        with gr.Tab("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        show_label=False,
                        type="messages",  # Gradio 4.x messages format
                        placeholder=(
                            "Type any HuggingFace model ID in the sidebar, or pick a suggested one.\n\n"
                            "Model loads on first message. Small models (GPT-2, Qwen 0.5B) load fastest."
                        ),
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Message...", show_label=False,
                            scale=5, autofocus=True, lines=1,
                        )
                        send_btn = gr.Button("Send ↵", variant="primary", scale=1)
                        clear_btn = gr.Button("Clear", scale=1)

                with gr.Column(scale=1, min_width=280):
                    gr.Markdown("### Model")
                    model_selector = gr.Dropdown(
                        choices=SUGGESTED_MODELS,
                        value=SUGGESTED_MODELS[0],
                        label="Select or type any HF model ID",
                        allow_custom_value=True,
                        interactive=True,
                    )
                    model_hint = gr.Markdown(_model_hint(SUGGESTED_MODELS[0]))

                    gr.Markdown("### Generation")
                    max_tokens_sl = gr.Slider(64, 2048, 512, step=64, label="Max new tokens")
                    temperature_sl = gr.Slider(0.1, 2.0, 0.7, step=0.05, label="Temperature")

                    gr.Markdown("### System prompt")
                    system_prompt_box = gr.Textbox(
                        value="You are a helpful, concise AI assistant.",
                        label="", lines=3,
                    )

                    gr.Markdown("### Hardware")
                    gr.HTML(f'<div class="hw-badge">{hw_line}</div>')

            def user_submit(message, history):
                """Add user message to history (messages format)."""
                if not message.strip():
                    return "", history
                history = history + [{"role": "user", "content": message}]
                return "", history

            def bot_respond(history, model_name, max_tok, temp, sys_prompt):
                """Generate bot response (messages format, streaming)."""
                if not history or history[-1]["role"] != "user":
                    yield history
                    return

                user_msg = history[-1]["content"]
                # History for context = everything except the last user message
                context = history[:-1]

                # Add empty assistant message for streaming
                history = history + [{"role": "assistant", "content": ""}]

                for partial in respond(user_msg, context, model_name, max_tok, temp, sys_prompt, hw):
                    history[-1]["content"] = partial
                    yield history

            model_selector.change(_model_hint, model_selector, model_hint)

            sub = dict(
                fn=user_submit,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot],
                queue=False,
            )
            then = dict(
                fn=bot_respond,
                inputs=[chatbot, model_selector, max_tokens_sl, temperature_sl, system_prompt_box],
                outputs=[chatbot],
            )
            msg_input.submit(**sub).then(**then)
            send_btn.click(**sub).then(**then)
            clear_btn.click(lambda: [], outputs=[chatbot])

        # ── Tab 2: Dataset Builder ────────────────────────────────────────────
        try:
            from little_fig.studio.dataset_builder import build_dataset_tab
            build_dataset_tab()
        except Exception as e:
            with gr.Tab("📂 Dataset Builder"):
                gr.Markdown(f"⚠ Could not load dataset builder: {e}")

        # ── Tab 3: Eval ───────────────────────────────────────────────────────
        try:
            from little_fig.studio.eval_tools import build_eval_tab
            build_eval_tab(get_current_model)
        except Exception as e:
            with gr.Tab("🧪 Eval"):
                gr.Markdown(f"⚠ Could not load eval tools: {e}")

        # ── Tab 4: Merge ──────────────────────────────────────────────────────
        try:
            from little_fig.studio.merge_tools import build_merge_tab
            build_merge_tab()
        except Exception as e:
            with gr.Tab("🔀 Merge"):
                gr.Markdown(f"⚠ Could not load merge tools: {e}")

    print("🍐 Little Fig Studio → http://0.0.0.0:8888")
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=8888, show_error=True)


def _model_hint(model_name: str) -> str:
    """Show RAM hints for known models."""
    hints = {
        "gpt-2":       "<span style='color:#94a3b8;font-size:0.78rem'>~500MB · Testing only · Fast</span>",
        "gpt2":        "<span style='color:#94a3b8;font-size:0.78rem'>~500MB · Testing only · Fast</span>",
        "tinyllama":   "<span style='color:#4ade80;font-size:0.78rem'>~2.2GB download · ~600MB with Fig Engine</span>",
        "qwen2.5-0.5b":"<span style='color:#4ade80;font-size:0.78rem'>~1GB download · Very fast on CPU</span>",
        "phi-2":       "<span style='color:#facc15;font-size:0.78rem'>~5.5GB download · ~1.5GB with Fig Engine</span>",
        "gemma-3-4b":  "<span style='color:#fb923c;font-size:0.78rem'>~9GB download · ~2.5GB with Fig Engine</span>",
        "llama-3.2-1b":"<span style='color:#4ade80;font-size:0.78rem'>~2.5GB download · Good for CPU</span>",
    }
    name_lower = model_name.lower()
    for key, text in hints.items():
        if key in name_lower:
            return text
    return "<span style='color:#94a3b8;font-size:0.78rem'>HuggingFace model · Auto-detects best loading strategy</span>"
