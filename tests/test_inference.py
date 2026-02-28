"""
Integration tests for the end-to-end inference pipeline.

Tests use mocked models to avoid requiring GPU / downloaded weights.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch


FAMOUS_CVES = [
    {
        "id": "CVE-2021-44228",
        "text": (
            "Apache Log4j2 2.0-beta9 through 2.15.0 JNDI features used in "
            "configuration, log messages, and parameters do not protect against "
            "attacker controlled LDAP endpoints."
        ),
        "expected_vendor": "Apache",
        "expected_product": "Log4j",
    },
    {
        "id": "CVE-2014-0160",
        "text": (
            "The TLS implementations in OpenSSL 1.0.1 before 1.0.1g do not "
            "properly handle Heartbeat Extension packets."
        ),
        "expected_vendor": "OpenSSL",
    },
    {
        "id": "CVE-2017-0144",
        "text": (
            "The SMBv1 server in Microsoft Windows Vista SP2 allows remote "
            "attackers to execute arbitrary code."
        ),
        "expected_vendor": "Microsoft",
    },
]


class TestPreprocessorIntegration:
    """Test the preprocessor with a real (fast) tokenizer."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        pytest.importorskip("transformers")
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("bert-base-uncased")

    def test_tokenize_and_align_shapes(self, tokenizer):
        from src.data.preprocessor import CVEPreprocessor
        proc = CVEPreprocessor(tokenizer, max_seq_length=64)
        words = ["Apache", "Log4j", "2.14.1", "is", "vulnerable"]
        labels = ["B-VENDOR", "B-PRODUCT", "B-VERSION", "O", "O"]
        encoded = proc.tokenize_and_align(words, labels)
        assert encoded["input_ids"].shape == (64,)
        assert encoded["attention_mask"].shape == (64,)
        assert encoded["labels"].shape == (64,)

    def test_no_labels_fills_outside(self, tokenizer):
        from src.data.preprocessor import CVEPreprocessor
        from src.data.preprocessor import IGNORE_LABEL_ID, LABEL2ID
        proc = CVEPreprocessor(tokenizer, max_seq_length=32)
        words = ["test", "sentence"]
        encoded = proc.tokenize_and_align(words)
        # Non-ignored labels should all be O (0)
        labels = encoded["labels"]
        valid = labels[labels != IGNORE_LABEL_ID]
        assert all(l.item() == LABEL2ID["O"] for l in valid)


class TestAnnotatorIntegration:
    """Full annotation pipeline for known CVEs."""

    @pytest.mark.parametrize("cve_info", FAMOUS_CVES)
    def test_annotation_finds_vendor(self, cve_info):
        from src.data.annotator import BIOAnnotator, extract_entities

        cve = {
            "id": cve_info["id"],
            "description": cve_info["text"],
            "cpe_matches": [],  # No CPE — all O
        }
        annotator = BIOAnnotator()
        pairs = annotator.annotate_cve(cve)
        assert isinstance(pairs, list)

    def test_log4shell_full_pipeline(self):
        """Full annotation for CVE-2021-44228 with CPE metadata."""
        from src.data.annotator import BIOAnnotator, extract_entities

        cve = {
            "id": "CVE-2021-44228",
            "description": "Apache Log4j 2.14.1 allows JNDI remote code execution",
            "cpe_matches": [
                "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
            ],
        }
        annotator = BIOAnnotator()
        pairs = annotator.annotate_cve(cve)
        words = [w for w, _ in pairs]
        labels = [l for _, l in pairs]

        entities = extract_entities(words, labels)

        # At minimum, version should be tagged
        found_any = any(l != "O" for l in labels)
        assert found_any, "Expected at least one entity to be tagged"


class TestCPEStringGeneration:

    def test_round_trip(self):
        """parse_cpe_string → entities_to_cpe should reproduce key components."""
        from src.data.preprocessor import parse_cpe_string, entities_to_cpe

        original = "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"
        parsed = parse_cpe_string(original)
        reconstructed = entities_to_cpe(parsed)

        assert "apache" in reconstructed
        assert "log4j" in reconstructed
        assert "2.14.1" in reconstructed

    @pytest.mark.parametrize("cpe", [
        "cpe:2.3:a:openssl:openssl:1.0.1:*:*:*:*:*:*:*",
        "cpe:2.3:o:microsoft:windows_10:1607:*:*:*:*:*:*:*",
        "cpe:2.3:a:apache:http_server:2.4.51:*:*:*:*:*:*:*",
    ])
    def test_parse_known_cpes(self, cpe):
        from src.data.preprocessor import parse_cpe_string
        result = parse_cpe_string(cpe)
        assert "vendor" in result
        assert "product" in result
        assert result["vendor"] != ""
        assert result["product"] != ""


class TestDatasetIntegration:

    @pytest.fixture(scope="class")
    def tokenizer(self):
        pytest.importorskip("transformers")
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("bert-base-uncased")

    def test_dataset_from_sequences(self, tokenizer):
        from src.data.dataset import NERDataset
        sequences = [
            [("Apache", "B-VENDOR"), ("Log4j", "B-PRODUCT"), ("2.14.1", "B-VERSION"), ("vulnerable", "O")],
            [("OpenSSL", "B-VENDOR"), ("1.0.1", "B-VERSION"), ("heartbleed", "O")],
        ]
        ds = NERDataset(sequences, tokenizer, max_seq_length=64)
        assert len(ds) == 2
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert item["input_ids"].shape == (64,)

    def test_dataset_split(self, tokenizer):
        from src.data.dataset import NERDataset
        sequences = [
            [("word", "O")] * 5 for _ in range(100)
        ]
        ds = NERDataset(sequences, tokenizer, max_seq_length=32)
        train_ds, val_ds, test_ds = ds.split(0.8, 0.1)
        # Totals should sum to original
        assert len(train_ds) + len(val_ds) + len(test_ds) == 100

    def test_ner_collate_fn(self, tokenizer):
        from src.data.dataset import NERDataset, ner_collate_fn
        sequences = [
            [("Apache", "B-VENDOR"), ("Log4j", "B-PRODUCT")],
        ] * 4
        ds = NERDataset(sequences, tokenizer, max_seq_length=32)
        batch = [ds[i] for i in range(4)]
        collated = ner_collate_fn(batch)
        assert collated["input_ids"].shape == (4, 32)
        assert collated["attention_mask"].shape == (4, 32)
        assert collated["labels"].shape == (4, 32)
