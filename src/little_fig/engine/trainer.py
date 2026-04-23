"""
Fig Engine — Unified Trainer (v3: FigPipeline + Ember)

Trains a FigModel using the selected tier:
    Tier 1: Streaming LoRA — standard backprop, LoRA params only
    Tier 2: LISA — rotating layer unfreezing with AdamW
    Tier 3: MeZO — zeroth-order optimization (no backward)
    Tier 4: LOMO — fused backward + update, O(1) gradient memory

Supports:
    - HuggingFace datasets (Hub or local JSONL)
    - ChatML / messages format
    - Instruction format (instruction/input/output)
    - Ember memory operation training data
    - Sequence packing
    - FigPipeline (GPU compute + CPU optimizer states)
    - torch.compile acceleration
    - Automatic tier selection
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import json
import time
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union

from .model import FigModel
from .tier import TrainingTier, select_tier, estimate_memory
from .lisa import LISAScheduler, LISAConfig
from .mezo import MeZOOptimizer, MeZOConfig
from .lomo import LOMOOptimizer, LOMOConfig
from .packing import PackedDataset, collate_packed


@dataclass
class FigTrainingConfig:
    """Configuration for Fig Engine training."""
    # Training tier
    tier: Optional[str] = None   # "streaming_lora", "lisa", "mezo", "lomo", or None (auto)
    
    # LoRA (Tier 1)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    
    # LISA (Tier 2)
    lisa_active_layers: int = 2
    lisa_switch_interval: int = 5
    
    # MeZO (Tier 3)
    mezo_epsilon: float = 1e-3
    
    # LOMO (Tier 4)
    lomo_clip_grad_norm: float = 1.0
    
    # General training
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 512
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Sequence packing
    use_packing: bool = True
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3
    
    # torch.compile
    compile_model: bool = False
    
    # FigPipeline (GPU compute + CPU optimizer states)
    use_pipeline: bool = True  # Auto-use FigPipeline when GPU available
    
    # Fig Memory Optimizations
    activation_checkpointing: bool = True   # Recompute activations in backward (~70% activation savings)
    memory_mode: str = "fast"               # "fast" (FP32 cache), "figcache" (75% less), "lowram" (min memory)
    figsweep_window: int = 0                # FigSweep rolling window (0=disabled, >0=window size)
    
    # Output
    output_dir: str = "./checkpoints/fig_run"
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps
    
    @property
    def training_tier(self) -> Optional[TrainingTier]:
        if self.tier is None:
            return None
        return TrainingTier(self.tier)


class FigTrainer:
    """
    Unified trainer for Fig Engine.
    
    Usage:
        model = FigModel.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        config = FigTrainingConfig(num_epochs=3)
        trainer = FigTrainer(model, config)
        trainer.load_dataset("tatsu-lab/alpaca")  # or path to JSONL
        trainer.train()
    """
    
    def __init__(self, model: FigModel, config: FigTrainingConfig):
        self.model = model
        self.config = config
        self.dataset = None
        self.dataloader = None
        
        # Resolve training tier
        if config.training_tier is not None:
            self.tier = config.training_tier
        else:
            total_params = sum(p.numel() for p in model.parameters())
            self.tier = select_tier(total_params)
        
        print(f"🍐 Training tier: {self.tier.value}")
        
        # Apply activation checkpointing
        if config.activation_checkpointing:
            self._apply_activation_checkpointing()
        
        # Apply Fig memory mode
        if config.figsweep_window > 0:
            model.enable_figsweep(window_size=config.figsweep_window)
        elif config.memory_mode != "fast":
            model.set_memory_mode(config.memory_mode)
    
    def _apply_activation_checkpointing(self):
        """Apply gradient checkpointing to transformer blocks.
        Recomputes activations in backward instead of storing them.
        ~70-80% activation memory savings at ~33% extra compute cost.
        """
        hf_model = self.model.model  # underlying HF model
        
        # Try HF's built-in gradient checkpointing first
        if hasattr(hf_model, "gradient_checkpointing_enable"):
            try:
                hf_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                print(f"🍐 Activation checkpointing enabled (HF native)")
                return
            except Exception:
                pass
        
        # Fallback: torch.utils.checkpoint on transformer blocks
        try:
            import torch.utils.checkpoint as ckpt
            count = 0
            for name, module in hf_model.named_modules():
                cls_name = module.__class__.__name__
                if any(x in cls_name for x in ["DecoderLayer", "Block", "TransformerBlock"]):
                    original_forward = module.forward
                    def make_ckpt_forward(orig):
                        def ckpt_forward(*args, **kwargs):
                            return ckpt.checkpoint(orig, *args, use_reentrant=False, **kwargs)
                        return ckpt_forward
                    module.forward = make_ckpt_forward(original_forward)
                    count += 1
            if count > 0:
                print(f"🍐 Activation checkpointing: {count} transformer blocks wrapped")
        except Exception as e:
            print(f"   ⚠ Activation checkpointing failed: {e}")
    
    def load_dataset(
        self,
        data_source: str,
        split: str = "train",
        text_column: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Load a dataset from HuggingFace Hub or local file.
        
        Supports:
            - HuggingFace dataset ID: "tatsu-lab/alpaca"
            - Local JSONL file: "./data/my_data.jsonl"
            - Local JSON file: "./data/my_data.json"
        
        Auto-detects format:
            - messages: [{"role": "user", "content": "..."}, ...]
            - instruction: {"instruction": "...", "input": "...", "output": "..."}
            - text: {"text": "..."}
        """
        tokenizer = self.model.tokenizer
        max_len = self.config.max_seq_length
        
        # Load raw data
        if os.path.isfile(data_source):
            examples = self._load_local(data_source)
        else:
            examples = self._load_hub(data_source, split, max_samples)
        
        if max_samples and len(examples) > max_samples:
            examples = examples[:max_samples]
        
        print(f"🍐 Dataset: {len(examples)} examples")
        
        # Detect format and tokenize
        tokenized = self._tokenize_examples(examples, tokenizer, max_len)
        
        # Sequence packing
        if self.config.use_packing:
            self.dataset = PackedDataset(
                tokenized, max_length=max_len,
                pad_token_id=tokenizer.pad_token_id or 0,
                eos_token_id=tokenizer.eos_token_id or 2,
            )
        else:
            self.dataset = SimpleDataset(tokenized, max_len, tokenizer.pad_token_id or 0)
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_packed,
            drop_last=True,
        )
    
    def load_ember_dataset(
        self,
        n_examples: int = 1000,
        max_samples: Optional[int] = None,
    ):
        """
        Generate and load Ember memory operation training data.
        
        Uses EmberTrainingDataGenerator to create synthetic examples
        that teach the model to perform memory operations (store, recall,
        consolidate, forget, conflict detection, episode segmentation,
        reflection).
        
        Requires the model to be loaded with ember_mode=True.
        """
        from .ember_integration import EmberTrainingDataGenerator
        
        gen = EmberTrainingDataGenerator()
        examples = gen.generate_dataset(n_examples=n_examples)
        
        if max_samples and len(examples) > max_samples:
            examples = examples[:max_samples]
        
        print(f"🔥 Ember dataset: {len(examples)} memory training examples")
        
        tokenizer = self.model.tokenizer
        max_len = self.config.max_seq_length
        
        tokenized = self._tokenize_instruction(examples, tokenizer, max_len)
        
        if self.config.use_packing:
            self.dataset = PackedDataset(
                tokenized, max_length=max_len,
                pad_token_id=tokenizer.pad_token_id or 0,
                eos_token_id=tokenizer.eos_token_id or 2,
            )
        else:
            self.dataset = SimpleDataset(tokenized, max_len, tokenizer.pad_token_id or 0)
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_packed,
            drop_last=True,
        )

    def _load_local(self, path: str) -> List[dict]:
        """Load from local JSONL or JSON file."""
        examples = []
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            examples.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    examples = data
                else:
                    examples = [data]
        return examples
    
    def _load_hub(self, dataset_id: str, split: str, max_samples: Optional[int]) -> List[dict]:
        """Load from HuggingFace Hub."""
        from datasets import load_dataset
        
        ds = load_dataset(dataset_id, split=split)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        
        return [dict(row) for row in ds]
    
    def _tokenize_examples(
        self,
        examples: List[dict],
        tokenizer,
        max_length: int,
    ) -> List[Dict[str, List[int]]]:
        """Detect format and tokenize."""
        if not examples:
            return []
        
        sample = examples[0]
        
        # Detect format
        if "messages" in sample:
            return self._tokenize_messages(examples, tokenizer, max_length)
        elif "instruction" in sample:
            return self._tokenize_instruction(examples, tokenizer, max_length)
        elif "text" in sample:
            return self._tokenize_text(examples, tokenizer, max_length)
        elif "prompt" in sample and "completion" in sample:
            return self._tokenize_prompt_completion(examples, tokenizer, max_length)
        else:
            # Try to find a text-like column
            for key in sample:
                if isinstance(sample[key], str) and len(sample[key]) > 20:
                    print(f"   Auto-detected text column: '{key}'")
                    return self._tokenize_text(
                        [{"text": ex[key]} for ex in examples],
                        tokenizer, max_length,
                    )
            raise ValueError(
                f"Cannot detect dataset format. Keys: {list(sample.keys())}. "
                f"Expected: messages, instruction/output, text, or prompt/completion"
            )
    
    def _tokenize_messages(self, examples, tokenizer, max_length):
        """Tokenize ChatML / messages format."""
        tokenized = []
        for ex in examples:
            messages = ex["messages"]
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                # Fallback: concatenate messages
                text = ""
                for msg in messages:
                    text += f"{msg['role']}: {msg['content']}\n"
            
            enc = tokenizer(text, truncation=True, max_length=max_length)
            tokenized.append({
                "input_ids": enc["input_ids"],
                "labels": enc["input_ids"].copy(),
            })
        return tokenized
    
    def _tokenize_instruction(self, examples, tokenizer, max_length):
        """Tokenize instruction/input/output format."""
        tokenized = []
        for ex in examples:
            instruction = ex.get("instruction", "")
            inp = ex.get("input", "").strip()
            output = ex.get("output", "")
            
            if inp:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            
            full_text = prompt + output
            
            # Tokenize full and prompt separately for label masking
            full_enc = tokenizer(full_text, truncation=True, max_length=max_length)
            prompt_enc = tokenizer(prompt, truncation=True, max_length=max_length)
            
            labels = full_enc["input_ids"].copy()
            prompt_len = len(prompt_enc["input_ids"])
            labels[:prompt_len] = [-100] * prompt_len
            
            tokenized.append({
                "input_ids": full_enc["input_ids"],
                "labels": labels,
            })
        return tokenized
    
    def _tokenize_text(self, examples, tokenizer, max_length):
        """Tokenize plain text format."""
        tokenized = []
        for ex in examples:
            text = ex["text"]
            enc = tokenizer(text, truncation=True, max_length=max_length)
            tokenized.append({
                "input_ids": enc["input_ids"],
                "labels": enc["input_ids"].copy(),
            })
        return tokenized
    
    def _tokenize_prompt_completion(self, examples, tokenizer, max_length):
        """Tokenize prompt/completion format."""
        tokenized = []
        for ex in examples:
            prompt = ex["prompt"]
            completion = ex["completion"]
            full_text = prompt + completion
            
            full_enc = tokenizer(full_text, truncation=True, max_length=max_length)
            prompt_enc = tokenizer(prompt, truncation=True, max_length=max_length)
            
            labels = full_enc["input_ids"].copy()
            prompt_len = len(prompt_enc["input_ids"])
            labels[:prompt_len] = [-100] * prompt_len
            
            tokenized.append({
                "input_ids": full_enc["input_ids"],
                "labels": labels,
            })
        return tokenized
    
    def train(self):
        """Run training loop."""
        if self.dataloader is None:
            raise ValueError("No dataset loaded. Call trainer.load_dataset() first.")
        
        config = self.config
        model = self.model.model  # Underlying HF model
        
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Setup based on tier
        if self.tier == TrainingTier.MEZO:
            return self._train_mezo()
        elif self.tier == TrainingTier.LOMO:
            return self._train_lomo()
        elif self.tier == TrainingTier.LISA:
            return self._train_lisa()
        else:
            return self._train_lora()
    
    def _train_lora(self):
        """Tier 1: Standard LoRA training with backprop.
        Uses FigPipeline for CPU-resident optimizer states when GPU is available.
        """
        config = self.config
        model = self.model

        # Check if GPU available for FigPipeline
        use_pipeline = torch.cuda.is_available() and config.use_pipeline
        if use_pipeline:
            return self._train_lora_pipeline()

        # CPU path: AdamW
        trainable_params = model.get_trainable_parameters()
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler
        total_steps = len(self.dataloader) * config.num_epochs // config.gradient_accumulation_steps
        warmup_steps = int(total_steps * config.warmup_ratio)
        scheduler = self._get_scheduler(optimizer, warmup_steps, total_steps)
        
        self._training_loop(model, optimizer, scheduler, "Streaming LoRA")

    def _train_lora_pipeline(self):
        """Tier 1 with FigPipeline: GPU compute + CPU optimizer states."""
        from .figpipeline import FigPipeline, PipelineConfig

        config = self.config
        model = self.model

        pipe_config = PipelineConfig(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm,
        )
        pipeline = FigPipeline(model.model, pipe_config)

        print(f"\n🍐 Training: Streaming LoRA + FigPipeline (GPU compute, CPU optimizer)")
        print(f"   Epochs: {config.num_epochs}")
        print(f"   Steps per epoch: {len(self.dataloader)}")

        global_step = 0
        for epoch in range(config.num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            t_epoch = time.time()

            for batch_idx, batch in enumerate(self.dataloader):
                loss = pipeline.train_step(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch.get("attention_mask"),
                )
                epoch_loss += loss
                epoch_steps += 1
                global_step += 1

                if global_step % config.logging_steps == 0:
                    avg = epoch_loss / epoch_steps
                    elapsed = time.time() - t_epoch
                    steps_per_sec = epoch_steps / elapsed
                    gpu_mb = pipeline.gpu_memory_mb
                    print(f"   step={global_step:5d}  loss={avg:.4f}  "
                          f"speed={steps_per_sec:.2f} steps/s  GPU={gpu_mb:.0f}MB")

            avg_loss = epoch_loss / max(epoch_steps, 1)
            print(f"   Epoch {epoch+1}/{config.num_epochs}: avg_loss={avg_loss:.4f}")

        pipeline.cleanup()
        self._save_checkpoint(model, global_step, "final")
    
    def _train_lisa(self):
        """Tier 2: LISA training."""
        config = self.config
        model = self.model
        
        # Setup LISA scheduler
        lisa_config = LISAConfig(
            active_layers=config.lisa_active_layers,
            switch_interval=config.lisa_switch_interval,
        )
        lisa_scheduler = LISAScheduler(model.model, lisa_config)
        
        # Optimizer on trainable params (updated when LISA switches layers)
        trainable_params = lisa_scheduler.get_trainable_params()
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        total_steps = len(self.dataloader) * config.num_epochs // config.gradient_accumulation_steps
        warmup_steps = int(total_steps * config.warmup_ratio)
        scheduler = self._get_scheduler(optimizer, warmup_steps, total_steps)
        
        self._training_loop(
            model, optimizer, scheduler, "LISA",
            lisa_scheduler=lisa_scheduler,
        )
    
    def _train_mezo(self):
        """Tier 3: MeZO training (no backward pass)."""
        config = self.config
        model = self.model
        model.model.eval()  # MeZO uses eval mode (no dropout)
        
        mezo_config = MeZOConfig(
            learning_rate=config.learning_rate,
            epsilon=config.mezo_epsilon,
            weight_decay=config.weight_decay,
        )
        mezo = MeZOOptimizer(model.model, mezo_config)
        
        print(f"\n🍐 Training: MeZO (zeroth-order, no backward pass)")
        print(f"   Epochs: {config.num_epochs}")
        print(f"   Steps per epoch: {len(self.dataloader)}")
        
        global_step = 0
        for epoch in range(config.num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            t_epoch = time.time()
            
            for batch_idx, batch in enumerate(self.dataloader):
                def forward_fn():
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch["labels"],
                    )
                    return outputs.loss
                
                loss = mezo.step(forward_fn)
                epoch_loss += loss
                epoch_steps += 1
                global_step += 1
                
                if global_step % config.logging_steps == 0:
                    avg = epoch_loss / epoch_steps
                    elapsed = time.time() - t_epoch
                    steps_per_sec = epoch_steps / elapsed
                    print(f"   step={global_step:5d}  loss={avg:.4f}  "
                          f"speed={steps_per_sec:.2f} steps/s")
            
            avg_loss = epoch_loss / max(epoch_steps, 1)
            print(f"   Epoch {epoch+1}/{config.num_epochs}: avg_loss={avg_loss:.4f}")
        
        self._save_checkpoint(model, global_step, "final")
    
    def _train_lomo(self):
        """Tier 4: LOMO training (fused backward + update)."""
        config = self.config
        model = self.model
        
        lomo_config = LOMOConfig(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            clip_grad_norm=config.lomo_clip_grad_norm,
        )
        lomo = LOMOOptimizer(model.model, lomo_config)
        
        print(f"\n🍐 Training: LOMO (fused backward, O(1) gradient memory)")
        print(f"   Epochs: {config.num_epochs}")
        
        global_step = 0
        for epoch in range(config.num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            t_epoch = time.time()
            
            for batch_idx, batch in enumerate(self.dataloader):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
                loss = outputs.loss
                epoch_loss += loss.item()
                epoch_steps += 1
                global_step += 1
                
                # Fused backward + update (no separate optimizer.step)
                lomo.fused_backward(loss)
                
                if global_step % config.logging_steps == 0:
                    avg = epoch_loss / epoch_steps
                    elapsed = time.time() - t_epoch
                    steps_per_sec = epoch_steps / elapsed
                    print(f"   step={global_step:5d}  loss={avg:.4f}  "
                          f"speed={steps_per_sec:.2f} steps/s")
            
            avg_loss = epoch_loss / max(epoch_steps, 1)
            print(f"   Epoch {epoch+1}/{config.num_epochs}: avg_loss={avg_loss:.4f}")
        
        lomo.remove_hooks()
        self._save_checkpoint(model, global_step, "final")
    
    def _training_loop(
        self,
        model: FigModel,
        optimizer,
        scheduler,
        tier_name: str,
        lisa_scheduler=None,
    ):
        """Standard training loop for LoRA and LISA tiers.
        """
        config = self.config
        
        total_steps = len(self.dataloader) * config.num_epochs
        accum_steps = config.gradient_accumulation_steps
        
        print(f"\n🍐 Training: {tier_name}")
        print(f"   Epochs: {config.num_epochs}")
        print(f"   Steps per epoch: {len(self.dataloader)}")
        print(f"   Effective batch: {config.effective_batch_size}")
        print(f"   Total optim steps: {total_steps // accum_steps}")
        
        model.model.train()
        global_step = 0
        accum_loss = 0.0
        
        for epoch in range(config.num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            t_epoch = time.time()
            
            for batch_idx, batch in enumerate(self.dataloader):
                # LISA: may switch active layers
                if lisa_scheduler is not None:
                    switched = lisa_scheduler.step(global_step)
                    if switched:
                        trainable_params = lisa_scheduler.get_trainable_params()
                        optimizer = torch.optim.AdamW(
                            trainable_params,
                            lr=config.learning_rate,
                            weight_decay=config.weight_decay,
                        )
                
                # Forward
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
                loss = outputs.loss / accum_steps
                
                # Backward
                loss.backward()
                
                accum_loss += loss.item()
                global_step += 1
                
                # Optimizer step every accum_steps
                if global_step % accum_steps == 0:
                    # Gradient clipping
                    if config.max_grad_norm > 0:
                        trainable = model.get_trainable_parameters()
                        torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    epoch_loss += accum_loss
                    epoch_steps += 1
                    
                    if epoch_steps % config.logging_steps == 0:
                        avg = epoch_loss / epoch_steps
                        lr = scheduler.get_last_lr()[0]
                        elapsed = time.time() - t_epoch
                        steps_per_sec = epoch_steps / elapsed
                        print(f"   step={global_step:5d}  loss={avg:.4f}  "
                              f"lr={lr:.2e}  speed={steps_per_sec:.2f} steps/s")
                    
                    accum_loss = 0.0
                
                # Save checkpoint
                if config.save_steps > 0 and global_step % (config.save_steps * accum_steps) == 0:
                    self._save_checkpoint(model, global_step, f"step_{global_step}")
            
            avg_loss = epoch_loss / max(epoch_steps, 1)
            print(f"   Epoch {epoch+1}/{config.num_epochs}: avg_loss={avg_loss:.4f}")
        
        # Final save
        self._save_checkpoint(model, global_step, "final")
    
    def _get_scheduler(self, optimizer, warmup_steps, total_steps):
        """Create a cosine learning rate scheduler with warmup."""
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(float(warmup_steps), 1)
            progress = float(step - warmup_steps) / max(float(total_steps - warmup_steps), 1)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return LambdaLR(optimizer, lr_lambda)
    
    def _save_checkpoint(self, model: FigModel, step: int, tag: str):
        """Save adapter checkpoint."""
        save_dir = os.path.join(self.config.output_dir, f"checkpoint-{tag}")
        model.save_adapter(save_dir)
        
        # Save training state
        state = {
            "step": step,
            "tier": self.tier.value,
            "config": {k: v for k, v in vars(self.config).items() if not k.startswith("_")},
        }
        with open(os.path.join(save_dir, "training_state.json"), "w") as f:
            json.dump(state, f, indent=2)


class SimpleDataset(Dataset):
    """Simple non-packed dataset with padding."""
    
    def __init__(self, examples: List[dict], max_length: int, pad_token_id: int):
        self.examples = examples
        self.max_length = max_length
        self.pad_token_id = pad_token_id
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        ids = ex["input_ids"][:self.max_length]
        labels = ex["labels"][:self.max_length]
        
        pad_len = self.max_length - len(ids)
        
        return {
            "input_ids": torch.tensor(ids + [self.pad_token_id] * pad_len, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(ids) + [0] * pad_len, dtype=torch.long),
            "labels": torch.tensor(labels + [-100] * pad_len, dtype=torch.long),
        }
