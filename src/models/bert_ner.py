"""
BERT NER Model
==============
Best-performing model from the paper: F1=95.48%, Accuracy=99.13%.

Architecture:
    BERT encoder (bert-base-uncased, 12 layers, 768 hidden)
    → Dropout layer
    → Linear classification head (768 → num_labels)

Training uses standard token-level cross-entropy loss with
IGNORE_LABEL_ID=-100 masking subword continuation tokens and special tokens.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import BertConfig, BertForTokenClassification, BertModel

from ..data.preprocessor import LABEL_LIST
from .base_model import BaseNERModel, NUM_LABELS

logger = logging.getLogger(__name__)


class BERTForNER(BaseNERModel):
    """BERT-based NER model for CPE entity extraction."""

    def __init__(
        self,
        pretrained_model: str = "bert-base-uncased",
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1,
    ):
        super().__init__(num_labels=num_labels, dropout=dropout)
        self._model_name = f"BERT-NER ({pretrained_model})"
        self.pretrained_model = pretrained_model

        # HuggingFace BertForTokenClassification encapsulates the full pipeline:
        # BERT encoder + dropout + token classification head
        self.bert = BertForTokenClassification.from_pretrained(
            pretrained_model,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
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
        """Forward pass through BERT + classification head.

        Args:
            input_ids:      (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) 1/0 mask
            labels:         (batch, seq_len) label IDs or None
            token_type_ids: (batch, seq_len) segment IDs (optional)

        Returns:
            Dict with 'loss' (if labels given) and 'logits'.
        """
        outputs = self.bert(
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
        pretrained_model: str = "bert-base-uncased",
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1,
    ) -> "BERTForNER":
        """Load a fine-tuned model from a saved checkpoint directory."""
        model = cls(pretrained_model, num_labels, dropout)
        model.load(checkpoint_dir)
        return model
