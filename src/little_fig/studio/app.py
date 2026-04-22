"""
Little Fig Studio — Main UI (v0.5)
Four tabs: Chat · Dataset Builder · Eval · Merge

Modern Gradio 6.x UI with light/dark theme support.
Uses Fig Engine for inference (no GGUF dependency).
Supports any HuggingFace model ID.
"""

import gradio as gr
import os
import time
import json
from typing import Optional, Iterator, Union


# ── Suggested models (user can type any model ID) ────────────────────────────

SUGGESTED_MODELS = [
    "GPT-2 124M (testing, fast)",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-2",
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
]

# Auto-discover local GGUF files in ./models/ directory
_models_dir = os.path.join(os.getcwd(), "models")
if os.path.isdir(_models_dir):
    import glob
    for f in sorted(glob.glob(os.path.join(_models_dir, "*.gguf"))):
        label = os.path.basename(f)
        SUGGESTED_MODELS.insert(0, f)  # Local files at top

# ── Global model state ────────────────────────────────────────────────────────

_loaded_model = None
_loaded_model_id = None


def _resolve_model_id(model_name: str) -> str:
    """Resolve display name to HF model ID."""
    if model_name.startswith("GPT-2"):
        return "gpt2"
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


# ── Gradio 6.x content helpers ────────────────────────────────────────────────

def _extract_text(content) -> str:
    """Normalise Gradio 6.x content (list of parts) or plain str → str."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return " ".join(parts)
    return str(content)


def _normalise_history(history: list) -> list:
    """Return history with every content field as a plain string."""
    return [
        {"role": msg["role"], "content": _extract_text(msg.get("content", ""))}
        for msg in history
    ]


# ── Chat logic ────────────────────────────────────────────────────────────────

def respond(message: Union[str, list], history: list, model_name: str,
            max_tokens: int, temperature: float, system_prompt: str, hw: dict):
    """Chat response generator — streaming, messages format."""
    message = _extract_text(message)
    history = _normalise_history(history)

    if not message.strip():
        yield ""
        return

    try:
        model = _load_model(model_name, hw)
    except Exception as e:
        yield f"❌ **Model load error**\n\n```\n{e}\n```"
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


# ── Themes ────────────────────────────────────────────────────────────────────

def _make_light_theme():
    """Clean white background, green accent."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.emerald,
        secondary_hue=gr.themes.colors.green,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    )


def _make_dark_theme():
    """Dark background, green accent."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.emerald,
        secondary_hue=gr.themes.colors.green,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    )


_LIGHT_CSS = """
/* ── Light theme: white bg, green accent ─────────────────────── */
:root, .light {
    --fig-bg: #ffffff;
    --fig-surface: #f8faf9;
    --fig-border: #e2e8f0;
    --fig-text: #1a202c;
    --fig-text-muted: #64748b;
    --fig-accent: #059669;
    --fig-accent-light: #d1fae5;
    --fig-accent-glow: #10b981;
}
.dark {
    --fig-bg: #0f1117;
    --fig-surface: #1a1d27;
    --fig-border: #2d3348;
    --fig-text: #e2e8f0;
    --fig-text-muted: #94a3b8;
    --fig-accent: #10b981;
    --fig-accent-light: #064e3b;
    --fig-accent-glow: #34d399;
}

/* Header */
.fig-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}
.fig-header h1 {
    font-size: 2rem;
    font-weight: 800;
    color: var(--fig-accent);
    margin: 0;
    letter-spacing: -0.02em;
}
.fig-header .subtitle {
    color: var(--fig-text-muted);
    font-size: 0.85rem;
    margin-top: 0.25rem;
    font-weight: 400;
}

/* Hardware badge */
.hw-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--fig-accent-light);
    border: 1px solid var(--fig-accent);
    border-radius: 8px;
    padding: 8px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--fig-accent);
    font-weight: 500;
}
.dark .hw-badge {
    background: var(--fig-accent-light);
    color: var(--fig-accent-glow);
}

/* Sidebar section headers */
.sidebar-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--fig-text-muted);
    margin-top: 1rem;
    margin-bottom: 0.25rem;
}

/* Hide Gradio footer */
footer { display: none !important; }

/* Smoother chatbot */
.chatbot { border-radius: 12px !important; }

/* Tab styling */
button.tab-nav { font-weight: 600 !important; }

/* Primary button */
.primary {
    background: var(--fig-accent) !important;
    border: none !important;
}
"""


# ── UI ────────────────────────────────────────────────────────────────────────

def run_studio(hw: Optional[dict] = None):
    if hw is None:
        from little_fig import HW
        hw = HW

    if hw.get("gpu_available"):
        hw_line = f"⚡ GPU: {hw['gpu_name']} ({hw['gpu_vram_gb']}GB)"
    else:
        hw_line = f"💻 CPU · {hw['cpu_cores']} cores · {hw.get('ram_available_gb', '?')}GB free"

    theme = _make_light_theme()
    css = _LIGHT_CSS

    with gr.Blocks(title="🍐 Little Fig") as demo:

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="fig-header">
            <h1>🍐 Little Fig</h1>
            <p class="subtitle">CPU-native LLM engine &nbsp;·&nbsp; v0.4 &nbsp;·&nbsp; Powered by Fig Engine</p>
        </div>
        """)

        # ── Tab 1: Chat ──────────────────────────────────────────────────
        with gr.Tab("💬 Chat"):
            with gr.Row(equal_height=True):
                # ── Main chat area ────────────────────────────────────────
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        show_label=False,
                        height=520,
                        placeholder=(
                            "<center><br><br>"
                            "<strong style='font-size:1.1rem;color:var(--fig-accent)'>🍐 Ready to chat</strong><br>"
                            "<span style='color:var(--fig-text-muted);font-size:0.85rem'>"
                            "Pick a model in the sidebar, then type a message.<br>"
                            "Small models (GPT-2, Qwen 0.5B) load fastest."
                            "</span></center>"
                        ),
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Type a message…",
                            show_label=False,
                            scale=6,
                            autofocus=True,
                            lines=1,
                            container=False,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)
                        clear_btn = gr.Button("Clear", scale=1, min_width=80)

                # ── Sidebar ───────────────────────────────────────────────
                with gr.Column(scale=1, min_width=260):
                    gr.HTML('<div class="sidebar-label">Model</div>')
                    model_selector = gr.Dropdown(
                        choices=SUGGESTED_MODELS,
                        value=SUGGESTED_MODELS[0],
                        label="Select or type any HF model ID",
                        allow_custom_value=True,
                        interactive=True,
                    )
                    model_hint = gr.Markdown(_model_hint(SUGGESTED_MODELS[0]))

                    gr.HTML('<div class="sidebar-label">Generation</div>')
                    max_tokens_sl = gr.Slider(
                        64, 2048, 512, step=64,
                        label="Max tokens",
                    )
                    temperature_sl = gr.Slider(
                        0.1, 2.0, 0.7, step=0.05,
                        label="Temperature",
                    )

                    gr.HTML('<div class="sidebar-label">System Prompt</div>')
                    system_prompt_box = gr.Textbox(
                        value="You are a helpful, concise AI assistant.",
                        show_label=False,
                        lines=3,
                    )

                    gr.HTML('<div class="sidebar-label">Hardware</div>')
                    gr.HTML(f'<div class="hw-badge">{hw_line}</div>')

            # ── Chat logic wiring ─────────────────────────────────────────
            def user_submit(message, history):
                if not message.strip():
                    return "", history
                history = history + [{"role": "user", "content": message}]
                return "", history

            def bot_respond(history, model_name, max_tok, temp, sys_prompt):
                if not history or history[-1]["role"] != "user":
                    yield history
                    return
                user_msg = _extract_text(history[-1]["content"])
                context = history[:-1]
                history = history + [{"role": "assistant", "content": ""}]
                for partial in respond(user_msg, context, model_name, max_tok, temp, sys_prompt, hw):
                    history[-1]["content"] = partial
                    yield history

            model_selector.change(_model_hint, model_selector, model_hint)

            sub = dict(fn=user_submit, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot], queue=False)
            then = dict(fn=bot_respond, inputs=[chatbot, model_selector, max_tokens_sl, temperature_sl, system_prompt_box], outputs=[chatbot])
            msg_input.submit(**sub).then(**then)
            send_btn.click(**sub).then(**then)
            clear_btn.click(lambda: [], outputs=[chatbot])

        # ── Tab 2: Dataset Builder ────────────────────────────────────────
        try:
            from little_fig.studio.dataset_builder import build_dataset_tab
            build_dataset_tab()
        except Exception as e:
            with gr.Tab("📂 Dataset Builder"):
                gr.Markdown(f"⚠ Could not load dataset builder: {e}")

        # ── Tab 3: Eval ───────────────────────────────────────────────────
        try:
            from little_fig.studio.eval_tools import build_eval_tab
            build_eval_tab(get_current_model)
        except Exception as e:
            with gr.Tab("🧪 Eval"):
                gr.Markdown(f"⚠ Could not load eval tools: {e}")

        # ── Tab 4: Merge ──────────────────────────────────────────────────
        try:
            from little_fig.studio.merge_tools import build_merge_tab
            build_merge_tab()
        except Exception as e:
            with gr.Tab("🔀 Merge"):
                gr.Markdown(f"⚠ Could not load merge tools: {e}")

    print("🍐 Little Fig Studio → http://0.0.0.0:8888")
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=8888,
        show_error=True,
        theme=theme,
        css=css,
    )


def _model_hint(model_name: str) -> str:
    """Show RAM hints for known models."""
    hints = {
        "gpt-2":       ("~500 MB", "emerald", "Testing only · Fast"),
        "gpt2":        ("~500 MB", "emerald", "Testing only · Fast"),
        "tinyllama":   ("~2.2 GB", "emerald", "Good for CPU"),
        "qwen2.5-0.5b":("~1 GB", "emerald",  "Very fast on CPU"),
        "phi-2":       ("~5.5 GB", "amber",   "Moderate RAM"),
        "gemma-3-4b":  ("~9 GB", "orange",    "Large · Needs RAM"),
        "gemma-4":     ("~5 GB", "orange",     "Large · Needs RAM"),
        "llama-3.2-1b":("~2.5 GB", "emerald", "Good for CPU"),
    }
    name_lower = model_name.lower()
    for key, (size, color, note) in hints.items():
        if key in name_lower:
            return f"<span style='font-size:0.78rem;color:var(--fig-text-muted)'>{size} · {note}</span>"
    return "<span style='font-size:0.78rem;color:var(--fig-text-muted)'>HuggingFace model · Auto-detects loading strategy</span>"
