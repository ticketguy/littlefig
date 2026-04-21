"""
Little Fig — Eval Tooling
Run test prompts against a loaded model and score the outputs.

Scoring methods:
  - Exact match     : output == expected (normalized)
  - Contains check  : expected substring in output
  - Length check    : output is within expected token range
  - Manual          : you score it yourself in the UI
"""

import gradio as gr
import json
import os
import time
import re
from datetime import datetime


EVAL_DIR = os.path.join(os.getcwd(), "evals")
os.makedirs(EVAL_DIR, exist_ok=True)


# ── Scoring ───────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def score_output(output: str, expected: str, method: str) -> tuple:
    """Returns (score: float 0-1, label: str)"""
    out_n = _normalize(output)
    exp_n = _normalize(expected)

    if method == "Exact match":
        match = out_n == exp_n
        return (1.0 if match else 0.0), ("✓ Match" if match else "✗ No match")

    if method == "Contains":
        match = exp_n in out_n
        return (1.0 if match else 0.0), ("✓ Contains" if match else "✗ Missing")

    if method == "Length (words)":
        exp_words = len(exp_n.split())
        out_words = len(out_n.split())
        ratio = min(out_words, exp_words) / max(out_words, exp_words, 1)
        return round(ratio, 2), f"{out_words} vs {exp_words} words ({ratio:.0%} match)"

    # Manual — no auto score
    return None, "— manual"


# ── Tab builder ───────────────────────────────────────────────────────────────

def build_eval_tab(get_model_fn):
    """
    get_model_fn: callable that returns the currently loaded model,
                  or None if no model is loaded.
    """

    with gr.Tab("🧪 Eval"):
        gr.Markdown("### Test your model against expected outputs")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("#### Test suite")

                eval_prompts = gr.Dataframe(
                    headers=["Prompt", "Expected output", "Score method"],
                    datatype=["str", "str", "str"],
                    column_count=(3, "fixed"),
                    row_count=(5, "dynamic"),
                    interactive=True,
                    wrap=True,
                    value=[
                        ["What is 2 + 2?", "4", "Contains"],
                        ["Say hello in one word.", "Hello", "Contains"],
                        ["", "", "Exact match"],
                        ["", "", "Exact match"],
                        ["", "", "Exact match"],
                    ],
                )

                score_method_note = gr.Markdown(
                    "<span style='color:#94a3b8;font-size:0.8rem'>"
                    "Score methods: **Exact match** · **Contains** · **Length (words)** · **Manual**"
                    "</span>"
                )

                with gr.Row():
                    run_eval_btn = gr.Button("▶ Run Eval", variant="primary")
                    save_eval_btn = gr.Button("💾 Save Results")

            with gr.Column(scale=3):
                gr.Markdown("#### Results")
                results_table = gr.Dataframe(
                    headers=["Prompt", "Expected", "Model output", "Score", "Time (s)"],
                    datatype=["str", "str", "str", "str", "number"],
                    column_count=(5, "fixed"),
                    interactive=False,
                    wrap=True,
                )
                summary_display = gr.Markdown("*Run eval to see results.*")

        eval_status = gr.Textbox(label="Status", interactive=False, lines=1)

        # ── Logic
        def run_eval(prompts_data):
            model = get_model_fn()
            if model is None:
                return [], "*No model loaded. Send a message in the Chat tab first.*", \
                       "⚠ Load a model first (use Chat tab to trigger load)."

            rows = []
            scores = []

            for row in prompts_data:
                prompt, expected, method = row[0], row[1], row[2]
                if not str(prompt).strip():
                    continue

                method = method if method in ["Exact match", "Contains", "Length (words)", "Manual"] \
                         else "Contains"

                t0 = time.time()
                try:
                    # Use non-streaming generate for eval
                    chat_prompt = model.apply_chat_template(str(prompt), [])
                    output = model.generate(chat_prompt)
                except Exception as e:
                    output = f"ERROR: {e}"
                    rows.append([prompt, expected, output, "✗ Error", 0])
                    continue

                elapsed = round(time.time() - t0, 1)
                score_val, score_label = score_output(output, str(expected), method)

                if score_val is not None:
                    scores.append(score_val)

                rows.append([
                    str(prompt)[:80],
                    str(expected)[:60],
                    output[:120],
                    score_label,
                    elapsed,
                ])

            if scores:
                avg = sum(scores) / len(scores)
                summary = f"**{len(rows)} tests run** · Average score: **{avg:.0%}** · " \
                          f"Pass (≥0.5): **{sum(1 for s in scores if s >= 0.5)}/{len(scores)}**"
            else:
                summary = f"**{len(rows)} tests run** · Manual scoring required."

            return rows, summary, f"✓ Eval complete — {len(rows)} prompts tested."

        def save_results(results_data):
            if not results_data:
                return "⚠ No results to save."
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(EVAL_DIR, f"eval_{timestamp}.json")
            with open(path, "w") as f:
                json.dump({
                    "timestamp": timestamp,
                    "results": results_data,
                }, f, indent=2)
            return f"✓ Saved → {path}"

        run_eval_btn.click(
            run_eval,
            inputs=[eval_prompts],
            outputs=[results_table, summary_display, eval_status],
        )
        save_eval_btn.click(
            save_results,
            inputs=[results_table],
            outputs=[eval_status],
        )