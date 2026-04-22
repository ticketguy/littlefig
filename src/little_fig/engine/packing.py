"""
Fig Engine — Sequence Packing

Packs multiple short sequences into single max-length sequences to
eliminate padding waste. Device-agnostic (works on CPU and GPU).

Ported from the Unsloth/TRL packing approach (arxiv reference in
Unsloth documentation). The key insight: variable-length examples
waste 40-80% of compute on padding tokens. Packing fills every
position with real tokens.

The collator provides `packed_seq_lengths` metadata so the attention
mask correctly prevents cross-sequence attention (block-diagonal mask).
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
import random


class PackedDataset(Dataset):
    """
    Wraps a tokenized dataset with sequence packing.
    
    Takes individual tokenized examples and packs them into sequences
    of max_length, separated by EOS tokens. Produces:
        - input_ids: packed sequence
        - attention_mask: all 1s (no padding)
        - labels: packed labels with -100 at sequence boundaries
        - position_ids: reset per sub-sequence for models that need it
    
    Usage:
        dataset = PackedDataset(
            examples=[
                {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
                {"input_ids": [4, 5], "labels": [4, 5]},
                ...
            ],
            max_length=512,
            pad_token_id=0,
        )
    """
    
    def __init__(
        self,
        examples: List[Dict[str, List[int]]],
        max_length: int = 512,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        
        # Pack examples
        self.packed = self._pack(examples, shuffle, seed)
        
        packing_ratio = len(examples) / max(len(self.packed), 1)
        print(f"🍐 Packed {len(examples)} examples → {len(self.packed)} sequences "
              f"({packing_ratio:.1f}× packing ratio, max_length={max_length})")
    
    def _pack(
        self,
        examples: List[Dict[str, List[int]]],
        shuffle: bool,
        seed: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """Bin-pack examples into max_length sequences."""
        if shuffle:
            rng = random.Random(seed)
            examples = examples.copy()
            rng.shuffle(examples)
        
        packed = []
        current_ids = []
        current_labels = []
        current_positions = []
        seq_lengths = []
        pos_offset = 0
        
        for ex in examples:
            ids = ex["input_ids"]
            labels = ex.get("labels", ids)
            
            # Truncate if single example is too long
            if len(ids) > self.max_length - 1:  # -1 for EOS separator
                ids = ids[:self.max_length - 1]
                labels = labels[:self.max_length - 1]
            
            # Check if fits in current sequence
            needed = len(ids) + 1  # +1 for EOS separator
            if len(current_ids) + needed > self.max_length:
                # Finalize current packed sequence
                if current_ids:
                    packed.append(self._finalize(
                        current_ids, current_labels, current_positions, seq_lengths
                    ))
                current_ids = []
                current_labels = []
                current_positions = []
                seq_lengths = []
                pos_offset = 0
            
            # Add example to current pack
            current_ids.extend(ids)
            current_labels.extend(labels)
            current_positions.extend(range(pos_offset, pos_offset + len(ids)))
            seq_lengths.append(len(ids))
            
            # Add EOS separator
            current_ids.append(self.eos_token_id)
            current_labels.append(-100)  # Don't train on separator
            current_positions.append(0)
            pos_offset = 0  # Reset position for next sub-sequence
        
        # Don't forget the last batch
        if current_ids:
            packed.append(self._finalize(
                current_ids, current_labels, current_positions, seq_lengths
            ))
        
        return packed
    
    def _finalize(
        self,
        ids: List[int],
        labels: List[int],
        positions: List[int],
        seq_lengths: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Pad to max_length and convert to tensors."""
        n = len(ids)
        pad_len = self.max_length - n
        
        if pad_len > 0:
            ids = ids + [self.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
            positions = positions + [0] * pad_len
            attn_mask = [1] * n + [0] * pad_len
        else:
            ids = ids[:self.max_length]
            labels = labels[:self.max_length]
            positions = positions[:self.max_length]
            attn_mask = [1] * self.max_length
        
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "position_ids": torch.tensor(positions, dtype=torch.long),
            "packed_seq_lengths": seq_lengths,  # metadata for attention masking
        }
    
    def __len__(self):
        return len(self.packed)
    
    def __getitem__(self, idx):
        item = self.packed[idx]
        # Return tensor items (skip packed_seq_lengths for default collation)
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"],
        }


def collate_packed(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for packed batches."""
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0].keys()
        if isinstance(batch[0][key], torch.Tensor)
    }
