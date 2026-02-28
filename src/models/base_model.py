"""
Base NER Model
==============
Abstract base class for all NER models.

Defines the interface and shared utilities (save/load, token alignment,
entity extraction from logits) used by BERT, XLNet, and GPT-2 variants.
"""

import abc
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

from ..data.preprocessor import ID2LABEL, IGNORE_LABEL_ID, LABEL_LIST

logger = logging.getLogger(__name__)

NUM_LABELS = len(LABEL_LIST)


class BaseNERModel(nn.Module, abc.ABC):
    """Abstract base class for CPE NER models."""

    def __init__(self, num_labels: int = NUM_LABELS, dropout: float = 0.1):
        super().__init__()
        self.num_labels = num_labels
        self.dropout_rate = dropout

    # ------------------------------------------------------------------ #
    #  Abstract interface                                                  #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run forward pass.

        Returns dict with:
            loss    — scalar cross-entropy loss (if labels provided)
            logits  — (batch, seq_len, num_labels) unnormalized scores
        """
        ...

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...

    # ------------------------------------------------------------------ #
    #  Shared utilities                                                    #
    # ------------------------------------------------------------------ #

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[List[str]]:
        """Run inference and decode label IDs to BIO label strings.

        Returns list of label lists (one per batch item).
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
        logits = outputs["logits"]
        pred_ids = torch.argmax(logits, dim=-1)  # (batch, seq_len)

        batch_labels = []
        for i in range(pred_ids.shape[0]):
            mask = attention_mask[i].bool()
            labels = [
                ID2LABEL.get(id_.item(), "O")
                for id_, m in zip(pred_ids[i], mask)
                if m
            ]
            batch_labels.append(labels)
        return batch_labels

    def save(self, save_dir: str) -> None:
        """Save model weights and config to directory."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_dir / "model.pt")
        logger.info("Model saved to %s", save_dir)

    def load(self, save_dir: str, device: Optional[torch.device] = None) -> None:
        """Load model weights from directory."""
        save_dir = Path(save_dir)
        map_location = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(save_dir / "model.pt", map_location=map_location)
        self.load_state_dict(state)
        logger.info("Model loaded from %s", save_dir)

    @staticmethod
    def compute_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        num_labels: int,
    ) -> torch.Tensor:
        """Cross-entropy loss ignoring IGNORE_LABEL_ID positions."""
        loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL_ID)
        return loss_fn(logits.view(-1, num_labels), labels.view(-1))

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        params = self.count_parameters()
        return f"{self.model_name} ({params:,} trainable params)"
