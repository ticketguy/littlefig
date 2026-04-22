"""
Little Fig Studio — v0.6 Modern UI

Gradio 6.x native:
  - gr.Sidebar for model config
  - gr.MultimodalTextbox for chat (text + images + files + voice)
  - Dark/light theme toggle with anti-flash
  - Free-text model input (HF ID or local path, no suggestions dropdown)
  - Bubble chat layout with retry/undo/copy
  - Tabs: Chat · Dataset Builder · Eval · Merge
"""

import gradio as gr
import os
import time
from typing import Optional, Union


# ── Global model state ────────────────────────────────────────────────────────

_loaded_model = None
_loaded_model_id = None
_hw = None


def _load_model(model_id: str):
    """Load model, caching to avoid reloads."""
    global _loaded_model, _loaded_model_id
    if not model_id or not model_id.strip():
        raise ValueError("No model specified")
    model_id = model_id.strip()
    if model_id == _loaded_model_id and _loaded_model is not None:
        return _loaded_model
    from little_fig.model import FigLanguageModel
    model = FigLanguageModel.from_pretrained(model_id, hw=_hw)
    _loaded_model = model
    _loaded_model_id = model_id
    return model


def _unload_model():
    global _loaded_model, _loaded_model_id
    _loaded_model = None
    _loaded_model_id = None


def get_current_model():
    return _loaded_model


# ── Gradio 6.x content helpers ───────────────────────────────────────────────

def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(p.get("text", ""))
        return " ".join(parts)
    return str(content)


def _normalise_history(history: list) -> list:
    return [
        {"role": m["role"], "content": _extract_text(m.get("content", ""))}
        for m in history
    ]


# ── JS for dark/light toggle ─────────────────────────────────────────────────

DARK_TOGGLE_JS = """
() => {
    const html = document.documentElement;
    const isDark = html.classList.contains('dark');
    if (isDark) {
        html.classList.remove('dark');
        localStorage.setItem('gradio-theme', 'light');
    } else {
        html.classList.add('dark');
        localStorage.setItem('gradio-theme', 'dark');
    }
    return isDark ? '☀️ Light' : '🌙 Dark';
}
"""

ANTI_FLASH_HEAD = """
<script>
(function(){
    let t = localStorage.getItem('gradio-theme');
    if (!t) t = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    if (t === 'dark') document.documentElement.classList.add('dark');
})();
</script>
"""

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
/* ── Custom variables ─────────────────────────────────────────── */
:root, .light {
    --fig-accent: #059669;
    --fig-accent-soft: #d1fae5;
    --fig-muted: #64748b;
}
.dark {
    --fig-accent: #10b981;
    --fig-accent-soft: #064e3b;
    --fig-muted: #94a3b8;
}

/* Logo */
.fig-logo {
    display: flex; align-items: center; gap: 10px;
    padding: 0.2rem 0 0.8rem 0;
}
.fig-logo h1 {
    font-size: 1.35rem; font-weight: 800;
    color: var(--fig-accent); margin: 0; letter-spacing: -0.02em;
}
.fig-logo span { font-size: 1.5rem; }
.fig-version {
    font-size: 0.65rem; color: var(--fig-muted);
    font-weight: 500; margin-top: 2px;
}

/* Status pill */
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.status-pill.ready {
    background: var(--fig-accent-soft);
    color: var(--fig-accent);
}
.status-pill.empty {
    background: #fef3c7; color: #92400e;
}
.dark .status-pill.empty {
    background: #422006; color: #fbbf24;
}

/* Hardware badge */
.hw-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--fig-accent-soft);
    border-radius: 8px; padding: 6px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; color: var(--fig-accent); font-weight: 500;
}

/* Section label */
.section-label {
    font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--fig-muted);
    margin: 0.8rem 0 0.15rem 0; padding: 0;
}

/* Hide Gradio footer */
footer { display: none !important; }

/* Chatbot tweaks */
.chatbot .message-wrap { border-radius: 16px !important; }
"""


# ── Theme ─────────────────────────────────────────────────────────────────────

def _make_theme():
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.emerald,
        secondary_hue=gr.themes.colors.teal,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
        radius_size="lg",
    )


# ── UI builder ────────────────────────────────────────────────────────────────

def run_studio(hw: Optional[dict] = None):
    global _hw
    if hw is None:
        from little_fig import HW
        hw = HW
    _hw = hw

    if hw.get("gpu_available"):
        hw_line = f"⚡ {hw['gpu_name']} · {hw['gpu_vram_gb']}GB"
    else:
        hw_line = f"💻 CPU · {hw['cpu_cores']} cores · {hw.get('ram_available_gb', '?')}GB free"

    with gr.Blocks(title="Little Fig", fill_height=True) as demo:

        # ═══════════════════════════════════════════════════════════════════
        # SIDEBAR
        # ═══════════════════════════════════════════════════════════════════
        with gr.Sidebar(position="left"):

            # Logo
            gr.HTML("""
            <div class="fig-logo">
                <span>🍐</span>
                <div>
                    <h1>Little Fig</h1>
                    <div class="fig-version">v0.5 · Fig Engine</div>
                </div>
            </div>
            """)

            # ── Model ─────────────────────────────────────────────────────
            gr.HTML('<p class="section-label">Model</p>')
            model_input = gr.Textbox(
                placeholder="HuggingFace ID or local path…",
                show_label=False,
                info="e.g. Qwen/Qwen2.5-0.5B-Instruct or ./models/my.gguf",
                lines=1,
            )
            with gr.Row():
                load_btn = gr.Button("Load", variant="primary", scale=2, size="sm")
                unload_btn = gr.Button("Unload", scale=1, size="sm")
            model_status = gr.HTML('<span class="status-pill empty">No model loaded</span>')

            # ── Parameters ────────────────────────────────────────────────
            with gr.Accordion("Parameters", open=True):
                temperature = gr.Slider(0.01, 2.0, value=0.7, step=0.05, label="Temperature")
                max_tokens = gr.Slider(64, 4096, value=512, step=64, label="Max tokens")
                top_p = gr.Slider(0.01, 1.0, value=0.9, step=0.05, label="Top-P")

            # ── System Prompt ─────────────────────────────────────────────
            with gr.Accordion("System Prompt", open=False):
                system_prompt = gr.Textbox(
                    value="You are a helpful, concise AI assistant.",
                    show_label=False,
                    lines=3,
                )

            # ── Hardware ──────────────────────────────────────────────────
            gr.HTML(f'<p class="section-label">Hardware</p>')
            gr.HTML(f'<div class="hw-badge">{hw_line}</div>')

            # ── Theme toggle ──────────────────────────────────────────────
            theme_btn = gr.Button("🌙 Dark", size="sm", variant="secondary")

        # ═══════════════════════════════════════════════════════════════════
        # MAIN AREA — Tabs
        # ═══════════════════════════════════════════════════════════════════

        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(
                show_label=False,
                buttons=["copy", "copy_all"],
                layout="bubble",
                resizable=True,
                height=520,
                placeholder=(
                    "<center style='padding:3rem 0'>"
                    "<p style='font-size:2.5rem;margin:0'>🍐</p>"
                    "<p style='font-size:1.1rem;font-weight:700;color:var(--fig-accent);margin:0.5rem 0 0.3rem'>Little Fig</p>"
                    "<p style='color:var(--fig-muted);font-size:0.85rem;max-width:300px;margin:0 auto'>"
                    "Type a HuggingFace model ID in the sidebar and click <b>Load</b>, then start chatting."
                    "</p></center>"
                ),
            )
            chat_input = gr.MultimodalTextbox(
                interactive=True,
                file_count="multiple",
                placeholder="Message Little Fig… or attach files",
                show_label=False,
                sources=["microphone", "upload"],
                file_types=["image", "audio", ".pdf", ".txt", ".py", ".json", ".csv"],
            )

        # ── Other tabs ────────────────────────────────────────────────────
        try:
            from little_fig.studio.dataset_builder import build_dataset_tab
            build_dataset_tab()
        except Exception as e:
            with gr.Tab("📂 Dataset Builder"):
                gr.Markdown(f"⚠ Could not load: {e}")

        try:
            from little_fig.studio.eval_tools import build_eval_tab
            build_eval_tab(get_current_model)
        except Exception as e:
            with gr.Tab("🧪 Eval"):
                gr.Markdown(f"⚠ Could not load: {e}")

        try:
            from little_fig.studio.merge_tools import build_merge_tab
            build_merge_tab()
        except Exception as e:
            with gr.Tab("🔀 Merge"):
                gr.Markdown(f"⚠ Could not load: {e}")

        # ═══════════════════════════════════════════════════════════════════
        # EVENT WIRING
        # ═══════════════════════════════════════════════════════════════════

        # ── Theme toggle ──────────────────────────────────────────────────
        theme_btn.click(fn=None, js=DARK_TOGGLE_JS, outputs=[theme_btn])

        # ── Model load/unload ─────────────────────────────────────────────
        def handle_load(model_id):
            if not model_id or not model_id.strip():
                return '<span class="status-pill empty">Enter a model ID first</span>'
            try:
                _load_model(model_id.strip())
                name = model_id.strip().split("/")[-1]
                return f'<span class="status-pill ready">✓ {name}</span>'
            except Exception as e:
                return f'<span class="status-pill empty">❌ {str(e)[:80]}</span>'

        def handle_unload():
            _unload_model()
            return '<span class="status-pill empty">No model loaded</span>'

        load_btn.click(handle_load, [model_input], [model_status])
        unload_btn.click(handle_unload, [], [model_status])

        # ── Chat: multimodal input → history → stream response ────────────
        def add_message(history, message):
            """Add user message (text + files) to history."""
            # message is {"text": "...", "files": [...]}
            text = message.get("text", "").strip() if isinstance(message, dict) else str(message).strip()
            files = message.get("files", []) if isinstance(message, dict) else []

            if not text and not files:
                return history, gr.MultimodalTextbox(value=None, interactive=True)

            # Add file messages first
            for f in files:
                # f is a filepath string
                history = history + [{"role": "user", "content": gr.File(f)}]

            # Add text message
            if text:
                history = history + [{"role": "user", "content": text}]

            return history, gr.MultimodalTextbox(value=None, interactive=False)

        def bot_respond(history, model_id, temp, max_tok, tp, sys_prompt):
            """Stream assistant response."""
            if not history:
                yield history
                return

            # Find last user text message
            user_msg = ""
            for m in reversed(history):
                if m["role"] == "user" and isinstance(m.get("content"), str):
                    user_msg = _extract_text(m["content"])
                    break

            if not user_msg:
                yield history
                return

            if _loaded_model is None:
                history = history + [{"role": "assistant", "content": "⚠️ No model loaded. Enter a model ID in the sidebar and click **Load**."}]
                yield history
                return

            _loaded_model.config.max_new_tokens = int(max_tok)
            _loaded_model.config.temperature = float(temp)

            context = _normalise_history(history[:-1]) if len(history) > 1 else []
            prompt = _loaded_model.apply_chat_template(user_msg, context)
            if sys_prompt and sys_prompt.strip():
                prompt = f"System: {sys_prompt}\n\n" + prompt

            history = history + [{"role": "assistant", "content": ""}]
            start = time.time()
            full = ""
            try:
                for chunk in _loaded_model.stream(prompt):
                    full += chunk
                    history[-1]["content"] = full
                    yield history
                elapsed = time.time() - start
                print(f"   ✓ ~{len(full.split())} words in {elapsed:.1f}s")
            except Exception as e:
                history[-1]["content"] = f"{full}\n\n❌ {e}"
                yield history

        def enable_input():
            return gr.MultimodalTextbox(interactive=True)

        chat_msg = chat_input.submit(
            add_message,
            [chatbot, chat_input],
            [chatbot, chat_input],
        )
        bot_msg = chat_msg.then(
            bot_respond,
            [chatbot, model_input, temperature, max_tokens, top_p, system_prompt],
            [chatbot],
        )
        bot_msg.then(enable_input, None, [chat_input])

        # Retry / undo
        chatbot.retry(
            bot_respond,
            [chatbot, model_input, temperature, max_tokens, top_p, system_prompt],
            [chatbot],
        )
        chatbot.undo(lambda h: (h, gr.MultimodalTextbox(interactive=True)), [chatbot], [chatbot, chat_input])

    # ── Launch ────────────────────────────────────────────────────────────
    print("🍐 Little Fig Studio → http://0.0.0.0:8888")
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=8888,
        show_error=True,
        theme=_make_theme(),
        css=CSS,
        head=ANTI_FLASH_HEAD,
    )
