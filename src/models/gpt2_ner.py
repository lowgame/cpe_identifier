"""
GPT-2 NER Model
===============
GPT-2 adapted for token classification (NER).

GPT-2 is a causal (left-to-right) language model. For NER, we adapt it
by adding a token classification head on top of its hidden states.

Key difference from BERT/XLNet: GPT-2 uses causal attention, so each
token only attends to previous tokens. This is suboptimal for NER
(BERT's bidirectional attention is better) but the paper includes it
as a comparison point.

Architecture:
    GPT-2 backbone (gpt2, 12 layers, 768 hidden)
    → All hidden states (not just last token)
    → Dropout
    → Linear classification head (768 → num_labels)
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2ForTokenClassification

from ..data.preprocessor import LABEL_LIST
from .base_model import BaseNERModel, NUM_LABELS

logger = logging.getLogger(__name__)


class GPT2ForNER(BaseNERModel):
    """GPT-2-based NER model for CPE entity extraction."""

    def __init__(
        self,
        pretrained_model: str = "gpt2",
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1,
    ):
        super().__init__(num_labels=num_labels, dropout=dropout)
        self._model_name = f"GPT2-NER ({pretrained_model})"
        self.pretrained_model = pretrained_model

        self.gpt2 = GPT2ForTokenClassification.from_pretrained(
            pretrained_model,
            num_labels=num_labels,
            id2label={i: l for i, l in enumerate(LABEL_LIST)},
            label2id={l: i for i, l in enumerate(LABEL_LIST)},
            ignore_mismatched_sizes=True,
        )

        # GPT-2 tokenizer doesn't have a default pad token; we handle this
        # by setting the pad_token_id to eos_token_id during training
        self.gpt2.config.pad_token_id = self.gpt2.config.eos_token_id

        # Additional dropout layer on top of GPT-2's built-in dropout
        self.dropout = nn.Dropout(dropout)

    @property
    def model_name(self) -> str:
        return self._model_name

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through GPT-2 + classification head.

        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels:         (batch, seq_len) or None
            token_type_ids: ignored (GPT-2 doesn't use segment IDs)

        Returns:
            Dict with 'loss' and 'logits'.
        """
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        result: Dict[str, torch.Tensor] = {"logits": outputs.logits}
        if outputs.loss is not None:
            result["loss"] = outputs.loss

        return result

    @classmethod
    def from_pretrained_checkpoint(
        cls,
        checkpoint_dir: str,
        pretrained_model: str = "gpt2",
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1,
    ) -> "GPT2ForNER":
        model = cls(pretrained_model, num_labels, dropout)
        model.load(checkpoint_dir)
        return model
