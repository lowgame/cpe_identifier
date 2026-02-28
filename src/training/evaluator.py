"""
NER Evaluator
=============
Computes precision, recall, F1, and accuracy for NER predictions.

Uses seqeval for entity-level (span-level) metrics — the standard
evaluation protocol for NER tasks. Also computes token-level accuracy.

Paper targets (BERT):
    F1=95.48%, Accuracy=99.13%, Precision=94.83%, Recall=96.14%
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from seqeval.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from ..data.preprocessor import ID2LABEL, IGNORE_LABEL_ID, LABEL_LIST

logger = logging.getLogger(__name__)


@dataclass
class NERMetrics:
    """Container for evaluation metrics."""
    precision: float
    recall: float
    f1: float
    accuracy: float
    per_entity: Dict[str, Dict[str, float]]
    loss: float = 0.0

    def __str__(self) -> str:
        lines = [
            f"F1:        {self.f1:.4f} ({self.f1*100:.2f}%)",
            f"Accuracy:  {self.accuracy:.4f} ({self.accuracy*100:.2f}%)",
            f"Precision: {self.precision:.4f} ({self.precision*100:.2f}%)",
            f"Recall:    {self.recall:.4f} ({self.recall*100:.2f}%)",
        ]
        if self.per_entity:
            lines.append("\nPer-entity breakdown:")
            for entity, scores in self.per_entity.items():
                lines.append(
                    f"  {entity:10s} | P={scores.get('precision',0):.3f} "
                    f"R={scores.get('recall',0):.3f} "
                    f"F1={scores.get('f1-score',0):.3f} "
                    f"N={int(scores.get('number',0))}"
                )
        return "\n".join(lines)


class NERMetricComputer:
    """Decodes model logits and computes NER metrics."""

    def decode_batch(
        self,
        logits: torch.Tensor,
        label_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Decode logits and label IDs to BIO label string lists.

        Args:
            logits:         (batch, seq_len, num_labels)
            label_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            (predictions, references) — both as list-of-label-lists,
            with IGNORE_LABEL_ID positions removed.
        """
        pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
        label_ids = label_ids.cpu().numpy()
        mask = attention_mask.cpu().numpy()

        all_preds: List[List[str]] = []
        all_refs: List[List[str]] = []

        for i in range(len(pred_ids)):
            preds_i: List[str] = []
            refs_i: List[str] = []
            for j in range(len(pred_ids[i])):
                if label_ids[i][j] == IGNORE_LABEL_ID:
                    continue
                if mask[i][j] == 0:
                    continue
                preds_i.append(ID2LABEL.get(int(pred_ids[i][j]), "O"))
                refs_i.append(ID2LABEL.get(int(label_ids[i][j]), "O"))

            if preds_i:
                all_preds.append(preds_i)
                all_refs.append(refs_i)

        return all_preds, all_refs

    def compute(
        self,
        predictions: List[List[str]],
        references: List[List[str]],
    ) -> Dict[str, float]:
        """Compute entity-level metrics using seqeval.

        Returns dict with: precision, recall, f1, accuracy
        """
        if not predictions or not references:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

        try:
            p = precision_score(references, predictions, zero_division=0)
            r = recall_score(references, predictions, zero_division=0)
            f1 = f1_score(references, predictions, zero_division=0)
            acc = accuracy_score(references, predictions)
        except Exception as e:
            logger.warning("seqeval error: %s", e)
            p, r, f1, acc = 0.0, 0.0, 0.0, 0.0

        return {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "accuracy": float(acc),
        }

    def compute_detailed(
        self,
        predictions: List[List[str]],
        references: List[List[str]],
    ) -> NERMetrics:
        """Full metrics including per-entity breakdown."""
        from seqeval.metrics import classification_report
        import re

        basic = self.compute(predictions, references)

        # Per-entity metrics from seqeval report
        per_entity: Dict[str, Dict[str, float]] = {}
        try:
            report = classification_report(
                references, predictions, output_dict=True, zero_division=0
            )
            for key, val in report.items():
                if key not in ("micro avg", "macro avg", "weighted avg", "accuracy"):
                    per_entity[key] = val
        except Exception:
            pass

        return NERMetrics(
            precision=basic["precision"],
            recall=basic["recall"],
            f1=basic["f1"],
            accuracy=basic["accuracy"],
            per_entity=per_entity,
        )

    def token_accuracy(
        self,
        predictions: List[List[str]],
        references: List[List[str]],
    ) -> float:
        """Token-level accuracy (not entity-level). Includes 'O' tokens."""
        correct = total = 0
        for pred_seq, ref_seq in zip(predictions, references):
            for p, r in zip(pred_seq, ref_seq):
                total += 1
                correct += int(p == r)
        return correct / max(total, 1)
