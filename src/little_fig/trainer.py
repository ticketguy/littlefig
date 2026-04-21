"""
Little Fig — Training Engine
LoRA fine-tuning with automatic GPU/CPU config.
Gemma-aware: uses correct target modules for Gemma architecture.

On CPU:   TinyLlama 1.1B practical. Gemma 4B: very slow but possible.
On GPU:   Gemma 4B fine-tuning feasible with 8GB+ VRAM.

Dataset format (JSONL):
    {"instruction": "...", "input": "...", "output": "..."}

Usage:
    trainer = FigTrainer.from_model("google/gemma-3-4b-it")
    trainer.load_dataset("./data/my_data.jsonl")
    trainer.train(output_dir="./checkpoints/run_01")
"""

import os
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, List
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer,
)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class FigTrainingConfig:
    # LoRA hyperparameters
    lora_r: int = 16              # Rank. 8–32 typical. Higher = more capacity.
    lora_alpha: int = 32          # Scaling. Usually 2x rank.
    lora_dropout: float = 0.05
    # target_modules set per-model in FigTrainer — don't override unless you know

    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 1           # Keep at 1 on CPU. 2-4 on GPU.
    gradient_accumulation: int = 8
    max_seq_length: int = 1024    # Gemma supports 8192 but keep low on CPU
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.01

    # Checkpointing
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3

    # GPU options (auto-set at runtime from HW detection)
    use_fp16: bool = False        # Set True automatically on GPU
    use_bf16: bool = False        # Set True on Ampere+ GPU


# ── Dataset ───────────────────────────────────────────────────────────────────

class FigInstructDataset(Dataset):
    """
    Instruction-tuning dataset.
    Expects JSONL: {"instruction": "...", "input": "...", "output": "..."}
    input field is optional.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        print(f"   Loading dataset: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    self.examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"   ⚠ Line {i+1} skipped: {e}")

        print(f"   ✓ {len(self.examples)} training examples loaded")

    def __len__(self):
        return len(self.examples)

    def _build_prompt(self, item: dict) -> tuple:
        """Returns (full_prompt, output_only) so we can mask input in labels."""
        instruction = item.get("instruction", "")
        inp = item.get("input", "").strip()
        output = item.get("output", "")

        if inp:
            prompt_part = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{inp}\n\n"
                f"### Response:\n"
            )
        else:
            prompt_part = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n"
            )

        return prompt_part, output

    def __getitem__(self, idx):
        item = self.examples[idx]
        prompt_part, output = self._build_prompt(item)
        full_text = prompt_part + output

        # Tokenize full text
        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize prompt only (to find where output starts)
        prompt_enc = self.tokenizer(
            prompt_part,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = full_enc["input_ids"].squeeze()
        labels = input_ids.clone()
        prompt_len = prompt_enc["input_ids"].shape[1]

        # Mask prompt tokens — only train on output
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": full_enc["attention_mask"].squeeze(),
            "labels": labels,
        }


# ── Trainer ───────────────────────────────────────────────────────────────────

class FigTrainer:
    """
    LoRA fine-tuner for Little Fig.
    Auto-detects GPU vs CPU and adjusts config accordingly.
    Gemma-aware: uses full attention target modules for Gemma architecture.
    """

    # Per-architecture LoRA target modules
    TARGET_MODULES = {
        "gemma":     ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "llama":     ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mistral":   ["q_proj", "k_proj", "v_proj", "o_proj"],
        "tinyllama": ["q_proj", "v_proj"],
        "phi":       ["q_proj", "v_proj", "fc1", "fc2"],
        "default":   ["q_proj", "v_proj"],
    }

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str = "",
        config: Optional[FigTrainingConfig] = None,
        hw: Optional[dict] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.config = config or FigTrainingConfig()
        self.dataset = None
        self._peft_model = None

        if hw is None:
            try:
                from little_fig import HW
                hw = HW
            except ImportError:
                hw = {"gpu_available": False}
        self.hw = hw

        # Auto-set fp16/bf16 based on GPU
        if hw.get("gpu_available"):
            # bf16 preferred on newer GPUs (Ampere+), fp16 otherwise
            props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
            if props and props.major >= 8:
                self.config.use_bf16 = True
            else:
                self.config.use_fp16 = True

    def _get_target_modules(self) -> List[str]:
        name = self.model_name.lower()
        for key in self.TARGET_MODULES:
            if key in name:
                return self.TARGET_MODULES[key]
        return self.TARGET_MODULES["default"]

    @staticmethod
    def from_model(
        model_name: str,
        config: Optional[FigTrainingConfig] = None,
        hw: Optional[dict] = None,
    ) -> "FigTrainer":
        if hw is None:
            try:
                from little_fig import HW
                hw = HW
            except ImportError:
                hw = {"gpu_available": False}

        gpu = hw.get("gpu_available", False)
        dtype = hw.get("recommended_dtype", torch.float32)

        print(f"🍐 Trainer loading: {model_name}")
        print(f"   Device: {'GPU' if gpu else 'CPU'} | dtype: {dtype}")

        if "gemma" in model_name.lower() and "4b" in model_name.lower() and not gpu:
            print("   ⚠ Gemma 4B on CPU: training will be very slow.")
            print("   ⚠ Reducing max_seq_length to 512 for feasibility.")
            if config is None:
                config = FigTrainingConfig()
            config.max_seq_length = 512
            config.batch_size = 1

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        load_kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "device_map": "auto" if gpu else "cpu",
        }

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        model.enable_input_require_grads()

        return FigTrainer(model, tokenizer, model_name, config, hw)

    def load_dataset(self, data_path: str):
        self.dataset = FigInstructDataset(
            data_path, self.tokenizer, self.config.max_seq_length
        )

    def _apply_lora(self):
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("pip install peft  ← required for LoRA training")

        target_modules = self._get_target_modules()
        print(f"   LoRA target modules: {target_modules}")

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self._peft_model = get_peft_model(self.model, lora_config)
        trainable, total = self._peft_model.get_nb_trainable_parameters()
        pct = 100 * trainable / total
        print(f"   ✓ LoRA applied: {trainable:,} / {total:,} params trainable ({pct:.2f}%)")

        return self._peft_model

    def train(self, output_dir: str = "./checkpoints/run_01"):
        if self.dataset is None:
            raise ValueError(
                "No dataset. Call trainer.load_dataset('./data/my_data.jsonl') first."
            )

        os.makedirs(output_dir, exist_ok=True)

        print(f"\n🍐 Training run → {output_dir}")
        print(f"   Model      : {self.model_name}")
        print(f"   Examples   : {len(self.dataset)}")
        print(f"   Epochs     : {self.config.num_epochs}")
        print(f"   LoRA rank  : {self.config.lora_r}")
        print(f"   Max seq    : {self.config.max_seq_length}")
        print(f"   Eff. batch : {self.config.batch_size * self.config.gradient_accumulation}")
        print(f"   Device     : {'GPU' if self.hw.get('gpu_available') else 'CPU'}")

        peft_model = self._apply_lora()

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            no_cuda=not self.hw.get("gpu_available", False),
            fp16=self.config.use_fp16,
            bf16=self.config.use_bf16,
            dataloader_pin_memory=self.hw.get("gpu_available", False),
            report_to="none",
            remove_unused_columns=False,
            gradient_checkpointing=True,   # Saves memory at small speed cost
        )

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=DataCollatorForSeq2Seq(
                self.tokenizer,
                model=peft_model,
                label_pad_token_id=-100,
                pad_to_multiple_of=8,
            ),
        )

        trainer.train()

        # Save adapter only (~10MB, not the full model)
        adapter_path = os.path.join(output_dir, "lora_adapter")
        peft_model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)

        print(f"\n🍐 ✓ Training complete.")
        print(f"   Adapter saved: {adapter_path}")
        print(f"   Load with: FigTrainer.load_adapter('{self.model_name}', '{adapter_path}')")

    @staticmethod
    def load_adapter(
        base_model_name: str,
        adapter_path: str,
        hw: Optional[dict] = None,
    ) -> "FigTrainer":
        """Load a trained LoRA adapter merged into the base model for inference."""
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("pip install peft")

        if hw is None:
            try:
                from little_fig import HW
                hw = HW
            except ImportError:
                hw = {"gpu_available": False}

        gpu = hw.get("gpu_available", False)
        dtype = hw.get("recommended_dtype", torch.float32)

        print(f"🍐 Loading base: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map="auto" if gpu else "cpu",
            low_cpu_mem_usage=True,
        )

        print(f"   Applying adapter: {adapter_path}")
        model = PeftModel.from_pretrained(base, adapter_path)
        model = model.merge_and_unload()
        model.eval()

        print("   ✓ Fine-tuned model ready")
        return FigTrainer(model, tokenizer, base_model_name, hw=hw)


# ── CLI entry point ───────────────────────────────────────────────────────────

def cli_train():
    """little-fig-train CLI — basic usage."""
    import argparse
    parser = argparse.ArgumentParser(description="Little Fig — LoRA fine-tuning")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--data", required=True, help="Path to JSONL dataset")
    parser.add_argument("--output", default="./checkpoints/run_01", help="Output dir")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-seq", type=int, default=1024)
    args = parser.parse_args()

    config = FigTrainingConfig(
        num_epochs=args.epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        max_seq_length=args.max_seq,
    )

    trainer = FigTrainer.from_model(args.model, config)
    trainer.load_dataset(args.data)
    trainer.train(args.output)