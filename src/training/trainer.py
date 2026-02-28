"""
NER Trainer
===========
Training loop for BERT / XLNet / GPT-2 NER models.

Features (paper-aligned):
- 20 epochs with AdamW optimizer
- Linear warmup + cosine decay LR schedule
- Gradient clipping (max_norm=1.0)
- Best-model checkpoint (by validation F1)
- TensorBoard logging
- Early stopping (patience configurable)
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, ModuleNotFoundError):
    # TensorBoard not installed — use a no-op writer
    class SummaryWriter:  # type: ignore[no-redef]
        def __init__(self, **kwargs): pass
        def add_scalar(self, *a, **kw): pass
        def close(self): pass
from transformers import get_linear_schedule_with_warmup

from ..models.base_model import BaseNERModel
from .evaluator import NERMetricComputer

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters and paths for a training run."""

    # Optimization
    epochs: int = 20
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0

    # Checkpoint & logging
    save_dir: str = "models"
    model_name: str = "bert_ner"
    log_dir: str = "logs"
    log_interval: int = 50       # log every N steps

    # Early stopping
    patience: int = 5             # epochs without F1 improvement
    min_delta: float = 0.001      # minimum F1 improvement to count

    # Device
    device: str = "auto"          # 'auto', 'cpu', 'cuda', 'cuda:0', ...

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


class NERTrainer:
    """Handles the full training lifecycle for a NER model."""

    def __init__(
        self,
        model: BaseNERModel,
        config: TrainingConfig,
    ):
        self.model = model
        self.config = config
        self.device = config.resolve_device()
        self.model.to(self.device)

        # Paths
        self.save_dir = Path(config.save_dir) / config.model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config.log_dir) / config.model_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.metric_computer = NERMetricComputer()

        # State
        self.best_f1 = 0.0
        self.epochs_without_improvement = 0
        self.global_step = 0

        logger.info("Trainer initialized | device=%s | model=%s", self.device, model.model_name)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        """Run full training loop.

        Returns dict with training history: train_loss, val_f1, val_loss per epoch.
        """
        optimizer = self._build_optimizer()
        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "val_f1": [],
            "val_precision": [], "val_recall": [], "val_accuracy": [],
        }

        logger.info(
            "Starting training: %d epochs, %d steps/epoch, %d warmup steps",
            self.config.epochs, len(train_loader), warmup_steps,
        )

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch, train_loader, optimizer, scheduler)
            val_metrics = self._evaluate(val_loader)
            elapsed = time.time() - t0

            # Log to TensorBoard
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            self.writer.add_scalar("F1/val", val_metrics["f1"], epoch)
            self.writer.add_scalar("Precision/val", val_metrics["precision"], epoch)
            self.writer.add_scalar("Recall/val", val_metrics["recall"], epoch)
            self.writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)

            # History
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_f1"].append(val_metrics["f1"])
            history["val_precision"].append(val_metrics["precision"])
            history["val_recall"].append(val_metrics["recall"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            logger.info(
                "Epoch %02d/%02d | train_loss=%.4f | val_loss=%.4f | "
                "F1=%.4f | Prec=%.4f | Rec=%.4f | Acc=%.4f | %.1fs",
                epoch, self.config.epochs,
                train_loss, val_metrics["loss"],
                val_metrics["f1"], val_metrics["precision"],
                val_metrics["recall"], val_metrics["accuracy"],
                elapsed,
            )

            # Checkpoint best model
            if val_metrics["f1"] > self.best_f1 + self.config.min_delta:
                self.best_f1 = val_metrics["f1"]
                self.epochs_without_improvement = 0
                self.model.save(str(self.save_dir / "best"))
                logger.info("  ↑ New best F1=%.4f — checkpoint saved", self.best_f1)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.config.patience:
                    logger.info(
                        "Early stopping triggered (no improvement for %d epochs)",
                        self.config.patience,
                    )
                    break

        self.writer.close()
        logger.info("Training complete. Best val F1=%.4f", self.best_f1)
        return history

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set and return metrics dict."""
        metrics = self._evaluate(test_loader)
        logger.info(
            "Test results | F1=%.4f | Precision=%.4f | Recall=%.4f | Accuracy=%.4f",
            metrics["f1"], metrics["precision"], metrics["recall"], metrics["accuracy"],
        )
        return metrics

    # ------------------------------------------------------------------ #
    #  Internal methods                                                    #
    # ------------------------------------------------------------------ #

    def _train_epoch(
        self,
        epoch: int,
        loader: DataLoader,
        optimizer: AdamW,
        scheduler,
    ) -> float:
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = self.model(**batch)
            loss = outputs["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            self.global_step += 1

            if step % self.config.log_interval == 0:
                avg = total_loss / step
                lr = scheduler.get_last_lr()[0]
                logger.debug(
                    "  Epoch %d | step %d/%d | loss=%.4f | lr=%.2e",
                    epoch, step, len(loader), avg, lr,
                )
                self.writer.add_scalar("Loss/train_step", loss.item(), self.global_step)

        return total_loss / len(loader)

    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds: List[List[str]] = []
        all_labels: List[List[str]] = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = self.model(**batch)

                if "loss" in outputs:
                    total_loss += outputs["loss"].item()

                preds, labels = self.metric_computer.decode_batch(
                    outputs["logits"], batch["labels"], batch["attention_mask"]
                )
                all_preds.extend(preds)
                all_labels.extend(labels)

        metrics = self.metric_computer.compute(all_preds, all_labels)
        metrics["loss"] = total_loss / max(len(loader), 1)
        return metrics

    def _build_optimizer(self) -> AdamW:
        """AdamW with layer-wise weight decay (no decay for bias/LayerNorm)."""
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        params = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(params, lr=self.config.learning_rate)
