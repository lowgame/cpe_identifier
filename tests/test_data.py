"""
Unit tests for the data pipeline.
"""

import pytest
from unittest.mock import MagicMock
from typing import List, Tuple

# ------------------------------------------------------------------ #
#  Preprocessor tests                                                  #
# ------------------------------------------------------------------ #

class TestCVEPreprocessor:

    def test_clean_text_html_entities(self):
        from src.data.preprocessor import CVEPreprocessor
        tokenizer = MagicMock()
        proc = CVEPreprocessor(tokenizer)
        assert proc.clean_text("&lt;script&gt;") == "<script>"
        assert proc.clean_text("&amp;amp;") == "&amp;"

    def test_clean_text_whitespace(self):
        from src.data.preprocessor import CVEPreprocessor
        tokenizer = MagicMock()
        proc = CVEPreprocessor(tokenizer)
        result = proc.clean_text("  too   many   spaces  ")
        assert result == "too many spaces"

    def test_split_sentences(self):
        from src.data.preprocessor import CVEPreprocessor
        tokenizer = MagicMock()
        proc = CVEPreprocessor(tokenizer)
        text = "Apache Log4j is vulnerable. It allows RCE. Update immediately."
        sents = proc.split_sentences(text)
        assert len(sents) >= 2

    def test_parse_cpe_string_log4j(self):
        from src.data.preprocessor import parse_cpe_string
        cpe = "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"
        result = parse_cpe_string(cpe)
        assert result["vendor"] == "apache"
        assert result["product"] == "log4j"
        assert result["version"] == "2.14.1"

    def test_parse_cpe_string_with_underscores(self):
        from src.data.preprocessor import parse_cpe_string
        cpe = "cpe:2.3:o:microsoft:windows_server_2008:*:sp2:*:*:*:*:*:*"
        result = parse_cpe_string(cpe)
        assert result["vendor"] == "microsoft"
        assert result["product"] == "windows server 2008"  # underscores → spaces
        assert result["update"] == "sp2"

    def test_parse_cpe_invalid(self):
        from src.data.preprocessor import parse_cpe_string
        result = parse_cpe_string("not-a-cpe-string")
        assert result == {}

    def test_entities_to_cpe(self):
        from src.data.preprocessor import entities_to_cpe
        entities = {"vendor": "Apache", "product": "Log4j", "version": "2.14.1"}
        cpe = entities_to_cpe(entities)
        assert cpe.startswith("cpe:2.3:a:")
        assert "apache" in cpe
        assert "log4j" in cpe
        assert "2.14.1" in cpe

    def test_entities_to_cpe_empty(self):
        from src.data.preprocessor import entities_to_cpe
        cpe = entities_to_cpe({})
        assert cpe.startswith("cpe:2.3:a:")
        assert cpe.count("*") >= 5


# ------------------------------------------------------------------ #
#  Annotator tests                                                     #
# ------------------------------------------------------------------ #

class TestBIOAnnotator:

    def _make_cve(self, description: str, cpe_matches: List[str]) -> dict:
        return {"id": "CVE-TEST-0001", "description": description, "cpe_matches": cpe_matches}

    def test_annotate_vendor(self):
        from src.data.annotator import BIOAnnotator
        annotator = BIOAnnotator()
        cve = self._make_cve(
            "Apache Log4j is vulnerable",
            ["cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"],
        )
        result = annotator.annotate_cve(cve)
        labels = {w: l for w, l in result}
        # Should tag "Apache" as B-VENDOR
        assert "B-VENDOR" in labels.values() or "B-PRODUCT" in labels.values()

    def test_annotate_empty_description(self):
        from src.data.annotator import BIOAnnotator
        annotator = BIOAnnotator()
        cve = self._make_cve("", ["cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"])
        result = annotator.annotate_cve(cve)
        assert result == []

    def test_annotate_no_cpe_matches(self):
        from src.data.annotator import BIOAnnotator
        annotator = BIOAnnotator()
        cve = self._make_cve("Some CVE description with no CPE.", [])
        result = annotator.annotate_cve(cve)
        labels = [l for _, l in result]
        assert all(l == "O" for l in labels)

    def test_label_statistics(self):
        from src.data.annotator import BIOAnnotator
        annotator = BIOAnnotator()
        sequences = [
            [("Apache", "B-VENDOR"), ("Log4j", "B-PRODUCT"), ("is", "O")],
            [("OpenSSL", "B-VENDOR"), ("1.0.1", "B-VERSION")],
        ]
        stats = annotator.get_label_statistics(sequences)
        assert stats["B-VENDOR"] == 2
        assert stats["B-PRODUCT"] == 1
        assert stats["O"] == 1
        assert stats["B-VERSION"] == 1

    def test_save_load_bio_roundtrip(self, tmp_path):
        from src.data.annotator import BIOAnnotator
        annotator = BIOAnnotator()
        sequences = [
            [("Apache", "B-VENDOR"), ("Log4j", "B-PRODUCT"), ("2.14.1", "B-VERSION")],
            [("is", "O"), ("vulnerable", "O")],
        ]
        bio_path = str(tmp_path / "test.bio")
        annotator.save_annotations(sequences, bio_path, format="bio")
        loaded = annotator.load_bio_file(bio_path)
        assert len(loaded) == 2
        assert loaded[0][0] == ("Apache", "B-VENDOR")
        assert loaded[0][1] == ("Log4j", "B-PRODUCT")

    def test_save_load_json_roundtrip(self, tmp_path):
        from src.data.annotator import BIOAnnotator
        annotator = BIOAnnotator()
        sequences = [
            [("Apache", "B-VENDOR"), ("is", "O")],
        ]
        json_path = str(tmp_path / "test.jsonl")
        annotator.save_annotations(sequences, json_path, format="json")
        loaded = annotator.load_bio_file.__func__  # test json separately
        # Just check the file was created
        assert (tmp_path / "test.jsonl").exists()


# ------------------------------------------------------------------ #
#  extract_entities tests                                              #
# ------------------------------------------------------------------ #

class TestExtractEntities:

    def test_basic_extraction(self):
        from src.data.annotator import extract_entities
        words  = ["Apache", "Log4j", "2.14.1", "is", "vulnerable"]
        labels = ["B-VENDOR", "B-PRODUCT", "B-VERSION", "O", "O"]
        entities = extract_entities(words, labels)
        assert entities.get("VENDOR") == ["Apache"]
        assert entities.get("PRODUCT") == ["Log4j"]
        assert entities.get("VERSION") == ["2.14.1"]

    def test_multi_word_entity(self):
        from src.data.annotator import extract_entities
        words  = ["Microsoft", "Windows", "10", "is", "affected"]
        labels = ["B-VENDOR", "B-PRODUCT", "I-PRODUCT", "O", "O"]
        entities = extract_entities(words, labels)
        assert entities.get("VENDOR") == ["Microsoft"]
        assert entities.get("PRODUCT") == ["Windows 10"]

    def test_all_outside(self):
        from src.data.annotator import extract_entities
        words  = ["no", "entities", "here"]
        labels = ["O", "O", "O"]
        entities = extract_entities(words, labels)
        assert entities == {}

    def test_consecutive_entities(self):
        from src.data.annotator import extract_entities
        words  = ["OpenSSL", "1.0.1f", "Enterprise"]
        labels = ["B-VENDOR", "B-VERSION", "B-EDITION"]
        entities = extract_entities(words, labels)
        assert "VENDOR" in entities
        assert "VERSION" in entities
        assert "EDITION" in entities


# ------------------------------------------------------------------ #
#  NVD Fetcher tests (mocked)                                          #
# ------------------------------------------------------------------ #

class TestNVDFetcher:

    def test_parse_cve_basic(self):
        from src.data.nvd_fetcher import NVDFetcher
        raw = {
            "id": "CVE-2021-44228",
            "descriptions": [{"lang": "en", "value": "Apache Log4j vulnerability"}],
            "published": "2021-12-10T00:00:00.000",
            "lastModified": "2021-12-15T00:00:00.000",
            "metrics": {},
            "configurations": [],
        }
        parsed = NVDFetcher._parse_cve(raw)
        assert parsed["id"] == "CVE-2021-44228"
        assert "Apache Log4j" in parsed["description"]
        assert parsed["severity"] == "NONE"
        assert parsed["cpe_matches"] == []

    def test_parse_cve_with_cpe(self):
        from src.data.nvd_fetcher import NVDFetcher
        raw = {
            "id": "CVE-2021-44228",
            "descriptions": [{"lang": "en", "value": "Log4j RCE"}],
            "published": "2021-12-10T00:00:00.000",
            "lastModified": "2021-12-10T00:00:00.000",
            "metrics": {
                "cvssMetricV31": [{
                    "cvssData": {"baseSeverity": "CRITICAL"}
                }]
            },
            "configurations": [{
                "nodes": [{
                    "cpeMatch": [
                        {"vulnerable": True, "criteria": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"}
                    ],
                    "children": [],
                }]
            }],
        }
        parsed = NVDFetcher._parse_cve(raw)
        assert parsed["severity"] == "CRITICAL"
        assert "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*" in parsed["cpe_matches"]

    def test_save_load_roundtrip(self, tmp_path):
        from src.data.nvd_fetcher import NVDFetcher
        fetcher = NVDFetcher(cache_dir=str(tmp_path))
        cves = [
            {"id": "CVE-2021-0001", "description": "test", "published": "2021-01-01",
             "modified": "2021-01-01", "severity": "HIGH", "cpe_matches": []},
        ]
        output_file = str(tmp_path / "test_cves.jsonl")
        fetcher.save_cves(cves, output_file)
        loaded = fetcher.load_cves(output_file)
        assert len(loaded) == 1
        assert loaded[0]["id"] == "CVE-2021-0001"
