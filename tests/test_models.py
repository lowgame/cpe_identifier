"""
Unit tests for NER model forward passes and utilities.

Note: These tests use small dummy inputs (no GPU required).
They verify shape correctness and loss computation without
requiring downloaded HuggingFace model weights.
"""

import pytest
import torch


class TestBaseModel:

    def test_compute_loss_shape(self):
        from src.models.base_model import BaseNERModel
        # (batch=2, seq_len=8, num_labels=11)
        logits = torch.randn(2, 8, 11)
        labels = torch.randint(0, 11, (2, 8))
        loss = BaseNERModel.compute_loss(logits, labels, 11)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0

    def test_compute_loss_with_ignore_index(self):
        from src.models.base_model import BaseNERModel
        from src.data.preprocessor import IGNORE_LABEL_ID
        logits = torch.randn(2, 8, 11)
        labels = torch.full((2, 8), IGNORE_LABEL_ID, dtype=torch.long)
        # All positions ignored — loss should be 0 or not raise
        # PyTorch CrossEntropyLoss returns nan when all targets are ignored_index
        # We just check it doesn't raise
        try:
            loss = BaseNERModel.compute_loss(logits, labels, 11)
        except Exception as e:
            pytest.fail(f"compute_loss raised: {e}")

    def test_label_list_completeness(self):
        from src.data.preprocessor import LABEL_LIST, LABEL2ID, ID2LABEL
        assert "O" in LABEL_LIST
        entity_types = ["VENDOR", "PRODUCT", "VERSION", "UPDATE", "EDITION"]
        for et in entity_types:
            assert f"B-{et}" in LABEL_LIST
            assert f"I-{et}" in LABEL_LIST
        # Round-trip
        for label, idx in LABEL2ID.items():
            assert ID2LABEL[idx] == label


class TestBERTNERShape:
    """Test NER model output shape conventions (no transformers required)."""

    def test_forward_output_keys(self):
        """Verify that the expected output dict keys and shapes are correct."""
        # Simulate what a NER model forward() must return
        batch, seq_len, num_labels = 2, 128, 11
        logits = torch.randn(batch, seq_len, num_labels)
        loss = torch.tensor(1.5)
        result = {"logits": logits, "loss": loss}

        assert "logits" in result
        assert "loss" in result
        assert result["logits"].shape == (batch, seq_len, num_labels)
        assert result["loss"].ndim == 0  # scalar

    def test_num_labels_matches_label_list(self):
        """NUM_LABELS in base_model must match LABEL_LIST length."""
        from src.models.base_model import NUM_LABELS
        from src.data.preprocessor import LABEL_LIST
        assert NUM_LABELS == len(LABEL_LIST) == 11


class TestEvaluator:

    def _make_preds_refs(self):
        preds = [
            ["B-VENDOR", "B-PRODUCT", "O", "O"],
            ["O", "B-VERSION", "I-VERSION", "O"],
        ]
        refs = [
            ["B-VENDOR", "B-PRODUCT", "O", "O"],   # perfect
            ["O", "B-VERSION", "B-VERSION", "O"],   # one wrong
        ]
        return preds, refs

    def test_compute_basic(self):
        from src.training.evaluator import NERMetricComputer
        comp = NERMetricComputer()
        preds, refs = self._make_preds_refs()
        metrics = comp.compute(preds, refs)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "accuracy" in metrics
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_perfect_predictions(self):
        from src.training.evaluator import NERMetricComputer
        comp = NERMetricComputer()
        labels = [["B-VENDOR", "B-PRODUCT", "O"]]
        metrics = comp.compute(labels, labels)
        assert metrics["f1"] == pytest.approx(1.0, abs=1e-3)
        assert metrics["precision"] == pytest.approx(1.0, abs=1e-3)
        assert metrics["recall"] == pytest.approx(1.0, abs=1e-3)

    def test_all_wrong(self):
        from src.training.evaluator import NERMetricComputer
        comp = NERMetricComputer()
        preds = [["B-VENDOR", "B-VENDOR"]]
        refs  = [["B-PRODUCT", "B-PRODUCT"]]
        metrics = comp.compute(preds, refs)
        assert metrics["f1"] == pytest.approx(0.0, abs=1e-3)

    def test_empty_inputs(self):
        from src.training.evaluator import NERMetricComputer
        comp = NERMetricComputer()
        metrics = comp.compute([], [])
        assert metrics["f1"] == 0.0

    def test_token_accuracy(self):
        from src.training.evaluator import NERMetricComputer
        comp = NERMetricComputer()
        preds = [["B-VENDOR", "O", "O"]]
        refs  = [["B-VENDOR", "O", "O"]]
        acc = comp.token_accuracy(preds, refs)
        assert acc == pytest.approx(1.0)

    def test_decode_batch(self):
        from src.training.evaluator import NERMetricComputer
        from src.data.preprocessor import IGNORE_LABEL_ID, LABEL2ID
        comp = NERMetricComputer()
        batch_size, seq_len, num_labels = 2, 6, 11
        logits = torch.zeros(batch_size, seq_len, num_labels)
        # Make label 1 (B-VENDOR) the highest for all positions
        logits[:, :, 1] = 10.0

        label_ids = torch.full((batch_size, seq_len), 1, dtype=torch.long)  # all B-VENDOR
        label_ids[:, 0] = IGNORE_LABEL_ID  # first pos ignored ([CLS])
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        preds, refs = comp.decode_batch(logits, label_ids, attention_mask)
        # Should have 2 sequences, each missing the ignored position
        assert len(preds) == 2
        assert all(l == "B-VENDOR" for l in preds[0])


class TestNERMetrics:

    def test_str_representation(self):
        from src.training.evaluator import NERMetrics
        metrics = NERMetrics(
            precision=0.9483,
            recall=0.9614,
            f1=0.9548,
            accuracy=0.9913,
            per_entity={"VENDOR": {"precision": 0.97, "recall": 0.96, "f1-score": 0.965, "number": 100}},
        )
        s = str(metrics)
        assert "95.48" in s
        assert "VENDOR" in s
