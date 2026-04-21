"""
Little Fig — Model Merge
Merge LoRA adapters into base models and export for inference.

What merging does:
  Training produces a small adapter file (~10MB).
  Merging folds those weights back into the base model.
  Result: a single model file that runs without PEFT overhead.
"""

import gradio as gr
import os
import json
import torch
from datetime import datetime


CHECKPOINTS_DIR = os.path.join(os.getcwd(), "checkpoints")
MERGED_DIR = os.path.join(os.getcwd(), "merged_models")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_adapters() -> list:
    """Scan checkpoints/ for lora_adapter subdirectories."""
    adapters = []
    if not os.path.isdir(CHECKPOINTS_DIR):
        return ["(no adapters found)"]
    for run in os.listdir(CHECKPOINTS_DIR):
        adapter_path = os.path.join(CHECKPOINTS_DIR, run, "lora_adapter")
        if os.path.isdir(adapter_path):
            adapters.append(f"{run}/lora_adapter")
    return adapters if adapters else ["(no adapters found)"]


def _read_adapter_config(adapter_rel_path: str) -> dict:
    """Read adapter_config.json to get base model name."""
    full_path = os.path.join(CHECKPOINTS_DIR, adapter_rel_path.replace("checkpoints/", "").lstrip("/"))
    config_path = os.path.join(full_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


# ── Tab builder ───────────────────────────────────────────────────────────────

def build_merge_tab():
    with gr.Tab("🔀 Merge"):
        gr.Markdown("### Merge LoRA adapter into base model")
        gr.Markdown(
            "<span style='color:#94a3b8;font-size:0.85rem'>"
            "After training, merge your adapter into the base model for faster inference. "
            "The merged model can be loaded directly without PEFT."
            "</span>"
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Adapter")
                adapter_dropdown = gr.Dropdown(
                    label="Select adapter (from checkpoints/)",
                    choices=_find_adapters(),
                    value=None,
                    interactive=True,
                )
                refresh_btn = gr.Button("🔄 Refresh list")
                adapter_info = gr.Markdown("*Select an adapter to see details.*")

                gr.Markdown("#### Base model")
                base_model_input = gr.Textbox(
                    label="HuggingFace model ID",
                    placeholder="e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    info="Must match the model the adapter was trained on.",
                )
                auto_detect_note = gr.Markdown("")

                gr.Markdown("#### Output")
                output_name_input = gr.Textbox(
                    label="Merged model folder name",
                    value="merged_model",
                    info="Saved to ./merged_models/<name>",
                )

                merge_btn = gr.Button("🔀 Merge & Save", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("#### Merge log")
                merge_log = gr.Textbox(
                    label="",
                    lines=18,
                    interactive=False,
                    show_label=False,
                    placeholder="Merge output will appear here...",
                )

        merge_status = gr.Textbox(label="Status", interactive=False, lines=1)

        # ── Logic
        def refresh_adapters():
            return gr.update(choices=_find_adapters())

        def on_adapter_selected(adapter_rel):
            if not adapter_rel or "no adapters" in adapter_rel:
                return "*No adapter selected.*", "", ""
            cfg = _read_adapter_config(adapter_rel)
            base = cfg.get("base_model_name_or_path", "")
            r = cfg.get("r", "?")
            modules = cfg.get("target_modules", [])
            info = (
                f"**Rank:** {r}  \n"
                f"**Target modules:** {', '.join(modules)}  \n"
                f"**Base model:** `{base if base else 'unknown'}`"
            )
            note = f"✓ Auto-detected base model from adapter config." if base else ""
            return info, base, note

        def run_merge(adapter_rel, base_model, output_name, hw_json):
            if not adapter_rel or "no adapters" in adapter_rel:
                return "⚠ Select an adapter first.", "⚠ No adapter selected."
            if not base_model.strip():
                return "⚠ Enter the base model ID.", "⚠ Missing base model."
            if not output_name.strip():
                output_name = "merged_model"

            hw = json.loads(hw_json) if hw_json else {"gpu_available": False}
            gpu = hw.get("gpu_available", False)
            dtype = torch.float16 if gpu else torch.float32

            log_lines = []
            def log(msg):
                log_lines.append(msg)
                print(msg)

            try:
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError:
                return "pip install peft transformers", "❌ Missing dependencies."

            adapter_full = os.path.join(CHECKPOINTS_DIR, adapter_rel.split("/checkpoints/")[-1]) \
                if "checkpoints/" in adapter_rel \
                else os.path.join(CHECKPOINTS_DIR, adapter_rel)

            # Handle relative paths from dropdown
            if not os.path.isabs(adapter_full):
                adapter_full = os.path.join(CHECKPOINTS_DIR, adapter_rel)

            log(f"🍐 Starting merge")
            log(f"   Adapter  : {adapter_full}")
            log(f"   Base     : {base_model}")
            log(f"   Device   : {'GPU' if gpu else 'CPU'}")
            log(f"   dtype    : {dtype}")
            log("")

            log("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(adapter_full)

            log("Loading base model...")
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=dtype,
                device_map="auto" if gpu else "cpu",
                low_cpu_mem_usage=True,
            )

            log("Applying LoRA adapter...")
            model = PeftModel.from_pretrained(base, adapter_full)

            log("Merging weights...")
            model = model.merge_and_unload()
            model.eval()

            output_path = os.path.join(MERGED_DIR, output_name)
            os.makedirs(output_path, exist_ok=True)

            log(f"Saving merged model → {output_path}")
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)

            size_mb = sum(
                os.path.getsize(os.path.join(output_path, f))
                for f in os.listdir(output_path)
                if os.path.isfile(os.path.join(output_path, f))
            ) / (1024 ** 2)

            log("")
            log(f"✓ Merge complete")
            log(f"   Output size : {size_mb:.0f} MB")
            log(f"   Load with   :")
            log(f"   AutoModelForCausalLM.from_pretrained('{output_path}')")

            return "\n".join(log_lines), f"✓ Merged → {output_path}"

        # Hidden state for hw
        hw_state = gr.State("{}")

        refresh_btn.click(refresh_adapters, outputs=[adapter_dropdown])

        adapter_dropdown.change(
            on_adapter_selected,
            inputs=[adapter_dropdown],
            outputs=[adapter_info, base_model_input, auto_detect_note],
        )

        merge_btn.click(
            run_merge,
            inputs=[adapter_dropdown, base_model_input, output_name_input, hw_state],
            outputs=[merge_log, merge_status],
        )