"""
BIO Annotator
=============
Converts CPE metadata from NVD into BIO-tagged training data.

Strategy (from paper):
1. Parse CPE strings from NVD CVE configurations
2. Match CPE components (vendor, product, version, update, edition) to CVE text
3. Apply BIO tagging: B- for first token, I- for continuation
4. Auto-annotate using a trained BERT model for unlabeled CVEs
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

from .preprocessor import LABEL2ID, LABEL_LIST, parse_cpe_string

logger = logging.getLogger(__name__)

# Minimum entity length to annotate (avoids single-char false positives)
MIN_ENTITY_LEN = 2


class BIOAnnotator:
    """Annotates CVE text with BIO tags using CPE metadata from NVD."""

    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def annotate_cve(self, cve: Dict) -> List[Tuple[str, str]]:
        """Annotate a CVE description with BIO tags from its CPE matches.

        Args:
            cve: Dict with 'description' and 'cpe_matches' keys.

        Returns:
            List of (word, bio_label) tuples for the full description.
        """
        text = cve.get("description", "")
        cpe_matches = cve.get("cpe_matches", [])

        if not text:
            return []

        # Collect all entity strings per type from CPE metadata
        entity_spans = self._extract_entity_strings(cpe_matches)

        # Tokenize text
        words = text.split()
        labels = ["O"] * len(words)

        # Match entities to word positions
        for entity_type, entity_values in entity_spans.items():
            for entity_value in sorted(entity_values, key=len, reverse=True):
                if len(entity_value) < MIN_ENTITY_LEN:
                    continue
                self._tag_entity(words, labels, entity_value, entity_type)

        return list(zip(words, labels))

    def annotate_batch(self, cves: List[Dict]) -> List[List[Tuple[str, str]]]:
        """Annotate a batch of CVEs. Returns list of word-label lists."""
        results = []
        for cve in cves:
            results.append(self.annotate_cve(cve))
        return results

    def save_annotations(
        self,
        annotations: List[List[Tuple[str, str]]],
        output_path: str,
        format: str = "bio",
    ) -> None:
        """Save annotations to file.

        Args:
            annotations: List of (word, label) lists.
            output_path: Output file path.
            format: 'bio' (CoNLL-style) or 'json'.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "bio":
            self._save_bio_format(annotations, output_path)
        elif format == "json":
            self._save_json_format(annotations, output_path)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'bio' or 'json'.")

        logger.info("Saved %d annotated sequences to %s", len(annotations), output_path)

    def load_bio_file(self, path: str) -> List[List[Tuple[str, str]]]:
        """Load CoNLL-style BIO annotated file."""
        sequences = []
        current: List[Tuple[str, str]] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line == "":
                    if current:
                        sequences.append(current)
                        current = []
                elif line.startswith("#"):
                    continue
                else:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        current.append((parts[0], parts[1]))

        if current:
            sequences.append(current)

        logger.info("Loaded %d sequences from %s", len(sequences), path)
        return sequences

    def get_label_statistics(
        self, annotations: List[List[Tuple[str, str]]]
    ) -> Dict[str, int]:
        """Count label occurrences across all annotations."""
        counts: Dict[str, int] = {label: 0 for label in LABEL_LIST}
        for seq in annotations:
            for _, label in seq:
                counts[label] = counts.get(label, 0) + 1
        return counts

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _extract_entity_strings(
        self, cpe_matches: List[str]
    ) -> Dict[str, List[str]]:
        """Parse CPE strings into entity type → value lists."""
        entity_type_to_values: Dict[str, List[str]] = {
            "VENDOR": [],
            "PRODUCT": [],
            "VERSION": [],
            "UPDATE": [],
            "EDITION": [],
        }

        for cpe in cpe_matches:
            parsed = parse_cpe_string(cpe)
            for field in ("vendor", "product", "version", "update", "edition"):
                val = parsed.get(field, "")
                if val and val != "*":
                    # Normalize underscores → spaces (CPE convention)
                    val = val.replace("_", " ").strip()
                    entity_type_to_values[field.upper()].append(val)

        # Deduplicate
        return {k: list(set(v)) for k, v in entity_type_to_values.items()}

    def _tag_entity(
        self,
        words: List[str],
        labels: List[str],
        entity_value: str,
        entity_type: str,
    ) -> None:
        """Find entity_value in words list and apply BIO tags in-place."""
        entity_words = entity_value.split()
        n = len(words)
        m = len(entity_words)

        if m == 0:
            return

        for i in range(n - m + 1):
            window = words[i : i + m]

            if self._words_match(window, entity_words):
                # Only tag if not already tagged (avoid overwriting)
                if all(labels[i + j] == "O" for j in range(m)):
                    labels[i] = f"B-{entity_type}"
                    for j in range(1, m):
                        labels[i + j] = f"I-{entity_type}"

    def _words_match(self, a: List[str], b: List[str]) -> bool:
        """Check if word lists match, optionally case-insensitive."""
        if len(a) != len(b):
            return False
        if self.case_sensitive:
            return a == b
        return [w.lower() for w in a] == [w.lower() for w in b]

    def _save_bio_format(
        self,
        annotations: List[List[Tuple[str, str]]],
        path: Path,
    ) -> None:
        """Save in CoNLL-style: word TAB label, blank line between sentences."""
        with open(path, "w", encoding="utf-8") as f:
            for seq in annotations:
                for word, label in seq:
                    f.write(f"{word}\t{label}\n")
                f.write("\n")

    def _save_json_format(
        self,
        annotations: List[List[Tuple[str, str]]],
        path: Path,
    ) -> None:
        """Save as JSONL: one JSON object per line with words and labels."""
        with open(path, "w", encoding="utf-8") as f:
            for seq in annotations:
                obj = {
                    "words": [w for w, _ in seq],
                    "labels": [l for _, l in seq],
                }
                f.write(json.dumps(obj) + "\n")


# ------------------------------------------------------------------ #
#  Utility: extract entity spans from BIO label sequence               #
# ------------------------------------------------------------------ #

def extract_entities(
    words: List[str], labels: List[str]
) -> Dict[str, List[str]]:
    """Extract named entities from a BIO-labeled word sequence.

    Args:
        words: List of tokens.
        labels: Corresponding BIO labels.

    Returns:
        Dict mapping entity type → list of entity strings.
    """
    entities: Dict[str, List[str]] = {}
    current_type: Optional[str] = None
    current_tokens: List[str] = []

    def flush():
        if current_type and current_tokens:
            entity_str = " ".join(current_tokens)
            entities.setdefault(current_type, []).append(entity_str)

    for word, label in zip(words, labels):
        if label.startswith("B-"):
            flush()
            current_type = label[2:]
            current_tokens = [word]
        elif label.startswith("I-") and current_type == label[2:]:
            current_tokens.append(word)
        else:
            flush()
            current_type = None
            current_tokens = []

    flush()
    return entities
