"""
Little Fig — Dataset Builder
Create, edit, preview, and export JSONL instruction datasets from the UI.
"""

import gradio as gr
import json
import os
from datetime import datetime


DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ── State helpers ─────────────────────────────────────────────────────────────

def _load_dataset(path: str) -> list:
    examples = []
    if not os.path.exists(path):
        return examples
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return examples


def _save_dataset(path: str, examples: list):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def _examples_to_table(examples: list) -> list:
    """Convert examples list to Gradio Dataframe rows."""
    return [
        [i + 1, ex.get("instruction", ""), ex.get("input", ""), ex.get("output", "")]
        for i, ex in enumerate(examples)
    ]


def _list_datasets() -> list:
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".jsonl")]
    return sorted(files) if files else ["(no datasets yet)"]


# ── Tab builder ───────────────────────────────────────────────────────────────

def build_dataset_tab():
    """Returns a Gradio Blocks tab for dataset building."""

    # In-memory state for current session
    current_examples = []

    with gr.Tab("📂 Dataset Builder"):
        gr.Markdown("### Build training datasets for fine-tuning")

        with gr.Row():
            # ── Left: entry form
            with gr.Column(scale=2):
                gr.Markdown("#### Add example")
                instruction_input = gr.Textbox(
                    label="Instruction",
                    placeholder="What should the model do? e.g. 'Summarize the following text.'",
                    lines=2,
                )
                input_input = gr.Textbox(
                    label="Input (optional)",
                    placeholder="Context or input for the instruction. Leave blank if none.",
                    lines=2,
                )
                output_input = gr.Textbox(
                    label="Expected output",
                    placeholder="The ideal response the model should learn to produce.",
                    lines=4,
                )
                with gr.Row():
                    add_btn = gr.Button("Add Example", variant="primary")
                    clear_form_btn = gr.Button("Clear Form")

                gr.Markdown("---")
                gr.Markdown("#### Save / Load")
                with gr.Row():
                    filename_input = gr.Textbox(
                        label="Dataset filename",
                        value="my_dataset.jsonl",
                        scale=3,
                    )
                    save_btn = gr.Button("💾 Save", variant="primary", scale=1)

                existing_datasets = gr.Dropdown(
                    label="Load existing dataset",
                    choices=_list_datasets(),
                    value=None,
                    interactive=True,
                )
                load_btn = gr.Button("📂 Load Selected")

            # ── Right: preview table
            with gr.Column(scale=3):
                gr.Markdown("#### Current dataset")
                stats_display = gr.Markdown("*No examples yet.*")
                preview_table = gr.Dataframe(
                    headers=["#", "Instruction", "Input", "Output"],
                    datatype=["number", "str", "str", "str"],
                    column_count=(4, "fixed"),
                    interactive=False,
                    wrap=True,
                )
                with gr.Row():
                    delete_idx = gr.Number(
                        label="Delete example # (1-based)",
                        precision=0,
                        minimum=1,
                    )
                    delete_btn = gr.Button("🗑 Delete", variant="stop")

        status_box = gr.Textbox(label="Status", interactive=False, lines=1)

        # ── Logic
        def refresh_display(examples):
            count = len(examples)
            stats = f"**{count} example{'s' if count != 1 else ''}** in current dataset."
            table = _examples_to_table(examples)
            return stats, table

        def add_example(instruction, inp, output, state_json):
            examples = json.loads(state_json) if state_json else []
            if not instruction.strip():
                return *refresh_display(examples), json.dumps(examples), "⚠ Instruction cannot be empty.", "", "", ""
            if not output.strip():
                return *refresh_display(examples), json.dumps(examples), "⚠ Output cannot be empty.", instruction, inp, output

            examples.append({
                "instruction": instruction.strip(),
                "input": inp.strip(),
                "output": output.strip(),
            })
            stats, table = refresh_display(examples)
            return stats, table, json.dumps(examples), f"✓ Added example #{len(examples)}", "", "", ""

        def delete_example(idx, state_json):
            examples = json.loads(state_json) if state_json else []
            idx = int(idx) - 1
            if idx < 0 or idx >= len(examples):
                return *refresh_display(examples), json.dumps(examples), f"⚠ No example at that index."
            removed = examples.pop(idx)
            stats, table = refresh_display(examples)
            return stats, table, json.dumps(examples), f"✓ Deleted example (was: '{removed['instruction'][:40]}...')"

        def save_dataset(filename, state_json):
            examples = json.loads(state_json) if state_json else []
            if not examples:
                return "⚠ Nothing to save.", gr.update()
            if not filename.endswith(".jsonl"):
                filename += ".jsonl"
            path = os.path.join(DATA_DIR, filename)
            _save_dataset(path, examples)
            return f"✓ Saved {len(examples)} examples → {path}", gr.update(choices=_list_datasets())

        def load_dataset_fn(filename, state_json):
            if not filename or filename == "(no datasets yet)":
                return *refresh_display([]), json.dumps([]), "⚠ No file selected."
            path = os.path.join(DATA_DIR, filename)
            examples = _load_dataset(path)
            stats, table = refresh_display(examples)
            return stats, table, json.dumps(examples), f"✓ Loaded {len(examples)} examples from {filename}"

        def clear_form():
            return "", "", ""

        # Hidden state
        state = gr.State("[]")

        add_btn.click(
            add_example,
            inputs=[instruction_input, input_input, output_input, state],
            outputs=[stats_display, preview_table, state, status_box,
                     instruction_input, input_input, output_input],
        )
        delete_btn.click(
            delete_example,
            inputs=[delete_idx, state],
            outputs=[stats_display, preview_table, state, status_box],
        )
        save_btn.click(
            save_dataset,
            inputs=[filename_input, state],
            outputs=[status_box, existing_datasets],
        )
        load_btn.click(
            load_dataset_fn,
            inputs=[existing_datasets, state],
            outputs=[stats_display, preview_table, state, status_box],
        )
        clear_form_btn.click(clear_form, outputs=[instruction_input, input_input, output_input])