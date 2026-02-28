"""
NER Dataset & DataLoader
========================
PyTorch Dataset class for BIO-tagged CVE NER data.

Handles:
- Loading from CoNLL-style BIO files or pre-tokenized JSON
- Token-label alignment with HuggingFace tokenizers
- Train / val / test splitting
- Collation into padded batches
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import PreTrainedTokenizerFast

from .preprocessor import IGNORE_LABEL_ID, LABEL2ID, CVEPreprocessor

logger = logging.getLogger(__name__)


class NERDataset(Dataset):
    """PyTorch Dataset for NER token classification.

    Each item is a dict with:
        input_ids       (max_seq_length,)  — token IDs
        attention_mask  (max_seq_length,)  — 1 for real, 0 for pad
        labels          (max_seq_length,)  — label IDs, -100 for subwords/pads
    """

    def __init__(
        self,
        sequences: List[List[Tuple[str, str]]],
        tokenizer: PreTrainedTokenizerFast,
        max_seq_length: int = 128,
    ):
        """
        Args:
            sequences: List of (word, BIO-label) lists.
            tokenizer: HuggingFace fast tokenizer.
            max_seq_length: Maximum token length (pad/truncate).
        """
        self.preprocessor = CVEPreprocessor(tokenizer, max_seq_length)
        self.items: List[Dict] = []

        for seq in sequences:
            if not seq:
                continue
            words = [w for w, _ in seq]
            labels = [l for _, l in seq]
            encoded = self.preprocessor.tokenize_and_align(words, labels)
            self.items.append(encoded)

        logger.info("NERDataset: %d samples loaded", len(self.items))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.items[idx]

    @classmethod
    def from_bio_file(
        cls,
        path: str,
        tokenizer: PreTrainedTokenizerFast,
        max_seq_length: int = 128,
    ) -> "NERDataset":
        """Load a dataset from CoNLL BIO file."""
        from .annotator import BIOAnnotator
        annotator = BIOAnnotator()
        sequences = annotator.load_bio_file(path)
        return cls(sequences, tokenizer, max_seq_length)

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        tokenizer: PreTrainedTokenizerFast,
        max_seq_length: int = 128,
    ) -> "NERDataset":
        """Load a dataset from JSONL file with 'words' and 'labels' keys."""
        sequences = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                seq = list(zip(obj["words"], obj["labels"]))
                sequences.append(seq)
        return cls(sequences, tokenizer, max_seq_length)

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple["NERDataset", "NERDataset", "NERDataset"]:
        """Split dataset into train / val / test subsets.

        Returns three NERDataset objects.
        """
        n = len(self)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds, test_ds = random_split(
            self, [n_train, n_val, n_test], generator=generator
        )

        # Wrap back into NERDataset containers (no re-tokenization)
        def subset_to_dataset(subset) -> "NERDataset":
            ds = object.__new__(type(self))
            ds.preprocessor = self.preprocessor
            ds.items = [self.items[i] for i in subset.indices]
            return ds

        return subset_to_dataset(train_ds), subset_to_dataset(val_ds), subset_to_dataset(test_ds)


class NERDataLoader:
    """Convenience wrapper that creates train/val/test DataLoaders."""

    def __init__(
        self,
        dataset: NERDataset,
        batch_size: int = 16,
        eval_batch_size: int = 32,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        num_workers: int = 0,
    ):
        self.train_ds, self.val_ds, self.test_ds = dataset.split(
            train_ratio, val_ratio, seed
        )
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        import torch
        self._pin_memory = torch.cuda.is_available()
        self._persistent = num_workers > 0

    @property
    def train(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ner_collate_fn,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent,
        )

    @property
    def val(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ner_collate_fn,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent,
        )

    @property
    def test(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ner_collate_fn,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent,
        )

    def __repr__(self) -> str:
        return (
            f"NERDataLoader(train={len(self.train_ds)}, "
            f"val={len(self.val_ds)}, test={len(self.test_ds)})"
        )


def ner_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate a list of NER samples into a padded batch dict."""
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0].keys()
        if isinstance(batch[0][key], torch.Tensor)
    }
