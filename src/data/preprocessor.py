"""
CVE Text Preprocessor
=====================
Cleans CVE text, tokenizes sentences, pads/truncates to max_seq_length,
and aligns subword tokens with word-level BIO tags.

Key design decisions from the paper:
- max_seq_length=128 (covers 93.96% of CVE sentences)
- BIO tags aligned to first subword token of each word
- Special tokens ([CLS], [SEP]) assigned 'O' label (ignored in loss)
"""

import html
import logging
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import nltk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

# Download punkt tokenizer if needed (quiet)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


LABEL_LIST = [
    "O",
    "B-VENDOR", "I-VENDOR",
    "B-PRODUCT", "I-PRODUCT",
    "B-VERSION", "I-VERSION",
    "B-UPDATE", "I-UPDATE",
    "B-EDITION", "I-EDITION",
]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# Special label ID used for subword continuation tokens (masked in loss)
IGNORE_LABEL_ID = -100


class CVEPreprocessor:
    """Handles text cleaning, tokenization, and subword-to-BIO alignment."""

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerFast",
        max_seq_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def clean_text(self, text: str) -> str:
        """Normalize and clean raw CVE description text."""
        # Decode HTML entities (e.g. &lt; → <)
        text = html.unescape(text)
        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Remove zero-width / control chars
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text

    def split_sentences(self, text: str) -> List[str]:
        """Split CVE description into sentences using NLTK punkt."""
        text = self.clean_text(text)
        return nltk.sent_tokenize(text)

    def tokenize_and_align(
        self,
        words: List[str],
        labels: Optional[List[str]] = None,
    ) -> Dict:
        """Tokenize word list and align BIO labels to subword tokens.

        Args:
            words: List of whitespace-split tokens.
            labels: Optional BIO label per word. If None, fills with 'O'.

        Returns:
            Dict with keys: input_ids, attention_mask, labels
            (all padded/truncated to max_seq_length)
        """
        if labels is None:
            labels = ["O"] * len(words)

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        aligned_labels = self._align_labels(encoding, labels)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": aligned_labels,
        }

    def preprocess_cve(
        self,
        cve: Dict,
        word_labels: Optional[List[Tuple[str, str]]] = None,
    ) -> List[Dict]:
        """Preprocess a single CVE dict into model-ready sentence batches.

        Args:
            cve: Dict with at least 'description' key.
            word_labels: If provided, list of (word, label) for the full text.

        Returns:
            List of tokenized sentence dicts.
        """
        text = self.clean_text(cve.get("description", ""))
        sentences = self.split_sentences(text)
        results = []

        if word_labels:
            # Align provided word-level labels to sentences
            label_map = {w: l for w, l in word_labels}

        for sent in sentences:
            words = sent.split()
            if not words:
                continue
            if word_labels:
                labels = [label_map.get(w, "O") for w in words]
            else:
                labels = None
            encoded = self.tokenize_and_align(words, labels)
            encoded["cve_id"] = cve.get("id", "")
            encoded["original_text"] = sent
            results.append(encoded)

        return results

    def words_from_text(self, text: str) -> List[str]:
        """Whitespace tokenize cleaned CVE text into words."""
        return self.clean_text(text).split()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _align_labels(self, encoding, word_labels: List[str]):
        """Map word-level BIO labels to subword token positions.

        Rules (paper-consistent):
        - First subword of a word → word's label
        - Continuation subwords  → IGNORE_LABEL_ID (masked in loss)
        - [CLS], [SEP], [PAD]    → IGNORE_LABEL_ID
        """
        import torch

        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(IGNORE_LABEL_ID)
            elif word_id != prev_word_id:
                label = word_labels[word_id] if word_id < len(word_labels) else "O"
                label_ids.append(LABEL2ID.get(label, 0))
            else:
                label_ids.append(IGNORE_LABEL_ID)
            prev_word_id = word_id

        return torch.tensor(label_ids, dtype=torch.long)


# ------------------------------------------------------------------ #
#  CPE string parser                                                   #
# ------------------------------------------------------------------ #

def parse_cpe_string(cpe: str) -> Dict[str, str]:
    """Parse a CPE 2.3 formatted string into component dict.

    CPE 2.3 format: cpe:2.3:part:vendor:product:version:update:edition:...

    Example:
        cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*
        →  {'part': 'a', 'vendor': 'apache', 'product': 'log4j',
             'version': '2.14.1', 'update': '*', 'edition': '*'}
    """
    fields = ["part", "vendor", "product", "version", "update", "edition",
              "sw_edition", "target_sw", "target_hw", "other"]
    parts = cpe.split(":")
    result: Dict[str, str] = {}

    # cpe:2.3:<part>:<vendor>:<product>:<version>:...
    if len(parts) >= 3 and parts[0] == "cpe" and parts[1] == "2.3":
        for i, field in enumerate(fields):
            idx = i + 2  # offset by 'cpe' and '2.3'
            if idx < len(parts):
                val = parts[idx].replace("_", " ") if parts[idx] != "*" else ""
                result[field] = val

    return result


def entities_to_cpe(entities: Dict[str, str]) -> str:
    """Convert extracted entity dict to CPE 2.3 string.

    Args:
        entities: Dict with any of: vendor/VENDOR, product/PRODUCT,
                  version/VERSION, update/UPDATE, edition/EDITION.
                  Keys are case-insensitive.

    Returns:
        CPE 2.3 formatted string.
    """
    # Normalize keys to lowercase for lookup
    norm = {k.lower(): v for k, v in entities.items()}

    def sanitize(val: str) -> str:
        return val.lower().replace(" ", "_") if val else "*"

    return ":".join([
        "cpe", "2.3", "a",
        sanitize(norm.get("vendor", "")),
        sanitize(norm.get("product", "")),
        sanitize(norm.get("version", "")),
        sanitize(norm.get("update", "")),
        sanitize(norm.get("edition", "")),
        "*", "*", "*", "*",
    ])
