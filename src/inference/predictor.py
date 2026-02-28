"""
CPE Predictor
=============
End-to-end inference pipeline: raw CVE text → extracted CPE entities.

Usage:
    predictor = CPEPredictor.from_checkpoint("models/bert_ner/best", model_type="bert")
    result = predictor.predict("Apache Log4j 2.14.1 allows RCE")
    print(result.entities)     # {'VENDOR': ['Apache'], 'PRODUCT': ['Log4j'], 'VERSION': ['2.14.1']}
    print(result.cpe_string)   # cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer

from ..data.annotator import extract_entities
from ..data.preprocessor import (
    CVEPreprocessor,
    entities_to_cpe,
    LABEL_LIST,
)
from ..models.base_model import BaseNERModel

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Structured result from a single CVE prediction."""
    text: str
    tokens: List[str]
    bio_labels: List[str]
    entities: Dict[str, List[str]]
    cpe_string: str
    confidence: Optional[float] = None

    def summary(self) -> str:
        lines = [
            f"Text: {self.text}",
            f"Entities:",
        ]
        for etype, vals in self.entities.items():
            lines.append(f"  {etype}: {', '.join(vals)}")
        lines.append(f"CPE: {self.cpe_string}")
        return "\n".join(lines)


class CPEPredictor:
    """Runs inference with a trained NER model to extract CPE entities."""

    def __init__(
        self,
        model: BaseNERModel,
        tokenizer_name: str,
        max_seq_length: int = 128,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # GPT-2 needs a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.preprocessor = CVEPreprocessor(self.tokenizer, max_seq_length)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def predict(self, text: str) -> PredictionResult:
        """Extract CPE entities from a single CVE description string."""
        text = self.preprocessor.clean_text(text)
        words = text.split()
        if not words:
            return PredictionResult(
                text=text, tokens=[], bio_labels=[],
                entities={}, cpe_string="cpe:2.3:a:*:*:*:*:*:*:*:*:*:*",
            )

        encoded = self.preprocessor.tokenize_and_align(words)
        input_ids = encoded["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = encoded["attention_mask"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

        # Decode: map subword predictions back to words
        bio_labels = self._decode_word_labels(encoded, logits.squeeze(0))

        # Extract entity spans
        entities = extract_entities(words, bio_labels)

        # Build CPE string from first occurrence of each entity type
        flat_entities = {
            k: v[0] if v else "" for k, v in entities.items()
        }
        cpe_str = entities_to_cpe(flat_entities)

        # Confidence: mean max-softmax probability over non-pad tokens
        probs = torch.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values.squeeze(0)
        mask = attention_mask.squeeze(0).bool()
        confidence = float(max_probs[mask].mean().item())

        return PredictionResult(
            text=text,
            tokens=words,
            bio_labels=bio_labels,
            entities=entities,
            cpe_string=cpe_str,
            confidence=confidence,
        )

    def predict_batch(self, texts: List[str]) -> List[PredictionResult]:
        """Run inference on a list of CVE texts."""
        return [self.predict(t) for t in texts]

    def predict_cve(self, cve: Dict) -> PredictionResult:
        """Predict from a CVE dict (with 'description' key)."""
        return self.predict(cve.get("description", ""))

    # ------------------------------------------------------------------ #
    #  Factory methods                                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        model_type: str = "bert",
        device: Optional[str] = None,
        max_seq_length: int = 128,
    ) -> "CPEPredictor":
        """Load a trained predictor from a model checkpoint directory.

        Args:
            checkpoint_dir: Path to directory with model.pt.
            model_type: 'bert', 'xlnet', or 'gpt2'.
            device: 'cpu', 'cuda', or None (auto-detect).
            max_seq_length: Token sequence length.
        """
        model_type = model_type.lower()
        pretrained_map = {
            "bert":  ("bert-base-uncased", "BERTForNER"),
            "xlnet": ("xlnet-base-cased",  "XLNetForNER"),
            "gpt2":  ("gpt2",              "GPT2ForNER"),
        }

        if model_type not in pretrained_map:
            raise ValueError(f"Unknown model_type: {model_type}. Use bert/xlnet/gpt2.")

        pretrained, class_name = pretrained_map[model_type]

        # Import the right class
        if model_type == "bert":
            from ..models.bert_ner import BERTForNER as ModelClass
        elif model_type == "xlnet":
            from ..models.xlnet_ner import XLNetForNER as ModelClass
        else:
            from ..models.gpt2_ner import GPT2ForNER as ModelClass

        model = ModelClass(pretrained_model=pretrained)
        model.load(checkpoint_dir, device=torch.device(device or "cpu"))

        return cls(
            model=model,
            tokenizer_name=pretrained,
            max_seq_length=max_seq_length,
            device=device,
        )

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _decode_word_labels(
        self,
        encoded: Dict,
        logits: torch.Tensor,
    ) -> List[str]:
        """Map subword logits back to word-level BIO labels.

        For each word, use the prediction of its first subword token.
        """
        from ..data.preprocessor import ID2LABEL, LABEL_LIST

        # Re-encode to get word_ids (without tensors)
        word_ids = encoded.get("word_ids", None)
        pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()

        if word_ids is None:
            # Fallback: just return per-token predictions
            return [ID2LABEL.get(p, "O") for p in pred_ids]

        word_to_label: Dict[int, str] = {}
        for tok_idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id not in word_to_label:
                word_to_label[word_id] = ID2LABEL.get(pred_ids[tok_idx], "O")

        max_word = max(word_to_label.keys()) if word_to_label else -1
        return [word_to_label.get(i, "O") for i in range(max_word + 1)]
