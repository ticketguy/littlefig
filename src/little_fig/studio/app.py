"""
Little Fig Studio — Main UI
Four tabs: Chat · Dataset Builder · Eval · Merge
"""

import gradio as gr
import os
import time
import json
from typing import Optional, Iterator

# ── Model registry ────────────────────────────────────────────────────────────

HF_MODELS = {
    "Gemma 3 4B-IT (HF float32, ~16GB RAM)":   "google/gemma-3-4b-it",
    "TinyLlama 1.1B Chat (fast on CPU)":        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Phi-2 2.7B":                               "microsoft/phi-2",
    "GPT-2 124M (testing only)":                "gpt2",
}

GGUF_MODELS = {}
_models_dir = os.path.join(os.getcwd(), "models")
if os.path.isdir(_models_dir):
    for f in os.listdir(_models_dir):
        if f.endswith(".gguf"):
            label = f.replace(".gguf", "").replace("-", " ").replace("_", " ") + " [GGUF]"
            GGUF_MODELS[label] = os.path.join(_models_dir, f)

ALL_MODELS = {**GGUF_MODELS, **HF_MODELS}

# ── Global model state ────────────────────────────────────────────────────────

_loaded_model = None
_loaded_model_name = None


def _load_model(model_name: str, hw: dict):
    global _loaded_model, _loaded_model_name
    if _loaded_model_name == model_name and _loaded_model is not None:
        return _loaded_model
    model_id = ALL_MODELS[model_name]
    if model_id.endswith(".gguf"):
        from little_fig.model import FigGGUFModel
        model, _ = FigGGUFModel.from_gguf(model_id, hw=hw)
    else:
        from little_fig.model import FigLanguageModel
        model = FigLanguageModel.from_pretrained(model_id, hw=hw)
    _loaded_model = model
    _loaded_model_name = model_name
    return model


def get_current_model():
    """Returns loaded model or None. Used by eval tab."""
    return _loaded_model


# ── Chat logic ────────────────────────────────────────────────────────────────

def respond(message, history, model_name, max_tokens, temperature, system_prompt, hw):
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

    hw_json = json.dumps({k: str(v) for k, v in hw.items()})

    if hw.get("gpu_available"):
        hw_line = f"⚡ GPU: {hw['gpu_name']} ({hw['gpu_vram_gb']}GB)"
        hw_color = "#4ade80"
    else:
        hw_line = f"💻 CPU · {hw['cpu_cores']} cores · no GPU"
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

    with gr.Blocks(title="🍐 Little Fig") as demo:

        gr.HTML("""
        <div class="fig-header">
            <h1>🍐 Little Fig</h1>
            <p>Independent LLM engine &nbsp;·&nbsp; CPU-native &nbsp;·&nbsp; v0.3.0</p>
        </div>
        """)

        hw_state = gr.State(hw_json)

        # ── Tab 1: Chat ───────────────────────────────────────────────────────
        with gr.Tab("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot( show_label=False,
                        placeholder=(
                            "Model loads on first message.\n\n"
                            "GGUF models in `./models/` appear first and run fastest on CPU."
                        ),
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Message...", show_label=False,
                            scale=5, autofocus=True, lines=1,
                        )
                        send_btn = gr.Button("Send ↵", variant="primary", scale=1)
                        clear_btn = gr.Button("Clear", scale=1)

                with gr.Column(scale=1, min_width=260):
                    gr.Markdown("### Model")
                    model_selector = gr.Dropdown(
                        choices=list(ALL_MODELS.keys()),
                        value=list(ALL_MODELS.keys())[0],
                        label="", interactive=True,
                    )
                    model_hint = gr.Markdown(_model_hint(list(ALL_MODELS.keys())[0]))

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
                return "", history + [[message, None]]

            def bot_respond(history, model_name, max_tok, temp, sys_prompt):
                user_msg = history[-1][0]
                history[-1][1] = ""
                for partial in respond(
                    user_msg, history[:-1], model_name, max_tok, temp, sys_prompt, hw
                ):
                    history[-1][1] = partial
                    yield history

            model_selector.change(_model_hint, model_selector, model_hint)

            sub = dict(fn=user_submit, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot], queue=False)
            then = dict(fn=bot_respond, inputs=[chatbot, model_selector, max_tokens_sl, temperature_sl, system_prompt_box], outputs=[chatbot])
            msg_input.submit(**sub).then(**then)
            send_btn.click(**sub).then(**then)
            clear_btn.click(lambda: [], outputs=[chatbot])

        # ── Tab 2: Dataset Builder ────────────────────────────────────────────
        from little_fig.studio.dataset_builder import build_dataset_tab
        build_dataset_tab()

        # ── Tab 3: Eval ───────────────────────────────────────────────────────
        from little_fig.studio.eval_tools import build_eval_tab
        build_eval_tab(get_current_model)

        # ── Tab 4: Merge ──────────────────────────────────────────────────────
        from little_fig.studio.merge_tools import build_merge_tab
        build_merge_tab()

    print("🍐 Little Fig Studio → http://0.0.0.0:8888")
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=8888, show_error=True, theme=theme, css=css)


def _model_hint(model_name: str) -> str:
    hints = {
        "gemma-3-4b":  "<span style='color:#fb923c;font-size:0.78rem'>~9GB download · ~16GB RAM · Prefer GGUF</span>",
        "TinyLlama":   "<span style='color:#4ade80;font-size:0.78rem'>~2.2GB · ~4GB RAM · Good for CPU</span>",
        "phi-2":       "<span style='color:#facc15;font-size:0.78rem'>~5.5GB · ~8GB RAM</span>",
        "gpt2":        "<span style='color:#94a3b8;font-size:0.78rem'>~500MB · Testing only</span>",
        "GGUF":        "<span style='color:#4ade80;font-size:0.78rem'>Quantized · Fast on CPU ✓</span>",
    }
    for key, text in hints.items():
        if key.lower() in model_name.lower():
            return text
    return "<span style='color:#94a3b8;font-size:0.78rem'>HuggingFace model</span>"