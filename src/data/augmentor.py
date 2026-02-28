"""
Data Augmentor
==============
Uses DistilRoBERTa (masked language model) to augment BIO-tagged CVE sequences.

Strategy (paper-aligned):
- Randomly mask non-entity tokens in the sentence
- Let DistilRoBERTa predict top-k alternatives
- Replace masked tokens while preserving entity spans
- Generates `num_augmented_per_sample` new sequences per original

This entity-aware augmentation ensures entities (vendor, product, version, etc.)
remain unchanged — only surrounding context words are perturbed.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple

import torch
from transformers import pipeline

logger = logging.getLogger(__name__)


class DataAugmentor:
    """Entity-aware data augmentation using DistilRoBERTa MLM."""

    MASK_TOKEN = "<mask>"

    def __init__(
        self,
        model_name: str = "distilroberta-base",
        top_k: int = 10,
        augment_ratio: float = 0.15,
        num_augmented_per_sample: int = 2,
        device: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Args:
            model_name: HuggingFace MLM model (default: distilroberta-base).
            top_k: Number of candidate replacements per masked token.
            augment_ratio: Fraction of non-entity tokens to mask per sample.
            num_augmented_per_sample: New sequences to generate per original.
            device: GPU device index. None = auto-detect.
            seed: Random seed for reproducibility.
        """
        self.top_k = top_k
        self.augment_ratio = augment_ratio
        self.num_augmented_per_sample = num_augmented_per_sample
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)

        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        logger.info("Loading MLM pipeline: %s (device=%s)", model_name, device)
        self.fill_mask = pipeline(
            "fill-mask",
            model=model_name,
            top_k=top_k,
            device=device,
        )
        self.mask_token = self.fill_mask.tokenizer.mask_token

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def augment_sequence(
        self,
        word_label_pairs: List[Tuple[str, str]],
    ) -> List[List[Tuple[str, str]]]:
        """Generate augmented versions of a BIO-tagged sequence.

        Args:
            word_label_pairs: List of (word, BIO-label) tuples.

        Returns:
            List of `num_augmented_per_sample` augmented sequences,
            each as a list of (word, BIO-label) tuples.
        """
        words = [w for w, _ in word_label_pairs]
        labels = [l for _, l in word_label_pairs]

        # Identify non-entity token positions (safe to mask)
        maskable_positions = [
            i for i, label in enumerate(labels) if label == "O"
        ]

        if not maskable_positions:
            return []  # Nothing to augment

        augmented = []
        for _ in range(self.num_augmented_per_sample):
            aug = self._augment_once(words, labels, maskable_positions)
            if aug:
                augmented.append(aug)

        return augmented

    def augment_dataset(
        self,
        sequences: List[List[Tuple[str, str]]],
        show_progress: bool = True,
    ) -> List[List[Tuple[str, str]]]:
        """Augment a full list of BIO sequences.

        Returns the original sequences PLUS all augmented ones.
        """
        from tqdm import tqdm

        all_sequences = list(sequences)
        iterator = tqdm(sequences, desc="Augmenting") if show_progress else sequences

        for seq in iterator:
            augmented = self.augment_sequence(seq)
            all_sequences.extend(augmented)

        logger.info(
            "Augmented %d → %d sequences (%.1fx)",
            len(sequences),
            len(all_sequences),
            len(all_sequences) / max(len(sequences), 1),
        )
        return all_sequences

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _augment_once(
        self,
        words: List[str],
        labels: List[str],
        maskable_positions: List[int],
    ) -> Optional[List[Tuple[str, str]]]:
        """Single augmentation pass: mask some tokens and replace."""
        num_to_mask = max(1, int(len(maskable_positions) * self.augment_ratio))
        positions_to_mask = random.sample(
            maskable_positions, min(num_to_mask, len(maskable_positions))
        )

        new_words = list(words)

        for pos in positions_to_mask:
            # Build masked sentence
            temp = list(new_words)
            temp[pos] = self.mask_token
            masked_text = " ".join(temp)

            try:
                predictions = self.fill_mask(masked_text)
                # Pick a random replacement from top-k (not always top-1)
                if predictions:
                    chosen = random.choice(predictions)
                    replacement = chosen["token_str"].strip()
                    if replacement:
                        new_words[pos] = replacement
            except Exception as e:
                logger.debug("Augmentation error at pos %d: %s", pos, e)
                continue

        if new_words == words:
            return None  # No changes made

        return list(zip(new_words, labels))

    @staticmethod
    def simple_synonym_augment(
        word_label_pairs: List[Tuple[str, str]],
        synonyms: Dict[str, List[str]],
        augment_ratio: float = 0.15,
    ) -> List[Tuple[str, str]]:
        """Lightweight synonym-based augmentation (no model needed).

        Useful for quick tests or when DistilRoBERTa is unavailable.

        Args:
            word_label_pairs: (word, label) pairs.
            synonyms: Dict mapping word → list of synonyms.
            augment_ratio: Fraction of non-entity tokens to replace.

        Returns:
            Single augmented sequence.
        """
        words = [w for w, _ in word_label_pairs]
        labels = [l for _, l in word_label_pairs]

        maskable = [i for i, l in enumerate(labels) if l == "O"]
        num_replace = max(1, int(len(maskable) * augment_ratio))
        to_replace = random.sample(maskable, min(num_replace, len(maskable)))

        new_words = list(words)
        for pos in to_replace:
            word = words[pos].lower()
            if word in synonyms:
                new_words[pos] = random.choice(synonyms[word])

        return list(zip(new_words, labels))
