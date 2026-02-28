"""
XLNet NER Model
===============
XLNet-based token classification for CPE entity extraction.

XLNet uses permutation language modeling and has a different attention
mechanism (relative positional encoding, segment recurrence). Adapted
here for standard NER token classification.

Architecture:
    XLNet encoder (xlnet-base-cased, 12 layers, 768 hidden)
    → Sequence summary / last token pooling
    → Dropout
    → Linear classification head (768 → num_labels)
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import XLNetConfig, XLNetForTokenClassification

from ..data.preprocessor import LABEL_LIST
from .base_model import BaseNERModel, NUM_LABELS

logger = logging.getLogger(__name__)


class XLNetForNER(BaseNERModel):
    """XLNet-based NER model for CPE entity extraction."""

    def __init__(
        self,
        pretrained_model: str = "xlnet-base-cased",
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1,
    ):
        super().__init__(num_labels=num_labels, dropout=dropout)
        self._model_name = f"XLNet-NER ({pretrained_model})"
        self.pretrained_model = pretrained_model

        self.xlnet = XLNetForTokenClassification.from_pretrained(
            pretrained_model,
            num_labels=num_labels,
            dropout=dropout,
            id2label={i: l for i, l in enumerate(LABEL_LIST)},
            label2id={l: i for i, l in enumerate(LABEL_LIST)},
            ignore_mismatched_sizes=True,
        )

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
        """Forward pass through XLNet + classification head.

        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels:         (batch, seq_len) or None
            token_type_ids: (batch, seq_len) or None

        Returns:
            Dict with 'loss' and 'logits'.
        """
        outputs = self.xlnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
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
        pretrained_model: str = "xlnet-base-cased",
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1,
    ) -> "XLNetForNER":
        model = cls(pretrained_model, num_labels, dropout)
        model.load(checkpoint_dir)
        return model
