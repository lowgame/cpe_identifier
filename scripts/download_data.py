#!/usr/bin/env python3
"""
Download CVE data from NVD API and prepare annotated training data.

Usage:
    python scripts/download_data.py --start-year 2020 --end-year 2022
    python scripts/download_data.py --start-year 2020 --end-year 2022 --api-key YOUR_KEY
    python scripts/download_data.py --cve-id CVE-2021-44228   # single CVE
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.nvd_fetcher import NVDFetcher
from src.data.annotator import BIOAnnotator
from src.data.preprocessor import parse_cpe_string

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Download and annotate CVE data")
    parser.add_argument("--start-year", type=int, default=2020, help="Start year (inclusive)")
    parser.add_argument("--end-year",   type=int, default=2022, help="End year (inclusive)")
    parser.add_argument("--api-key",    type=str, default=None,  help="NVD API key")
    parser.add_argument("--cache-dir",  type=str, default="data/raw/nvd_cache")
    parser.add_argument("--output-dir", type=str, default="data/annotated")
    parser.add_argument("--cve-id",     type=str, default=None,  help="Fetch a single CVE by ID")
    parser.add_argument("--no-cache",   action="store_true",     help="Skip cache, re-fetch")
    parser.add_argument("--format",     choices=["bio", "json"], default="bio",
                        help="Output annotation format")
    return parser.parse_args()


def main():
    args = parse_args()
    fetcher = NVDFetcher(api_key=args.api_key, cache_dir=args.cache_dir)
    annotator = BIOAnnotator()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cve_id:
        # Single CVE mode
        logger.info("Fetching %s ...", args.cve_id)
        cve = fetcher.fetch_cve_by_id(args.cve_id)
        if not cve:
            logger.error("CVE not found: %s", args.cve_id)
            sys.exit(1)
        logger.info("CVE ID:      %s", cve["id"])
        logger.info("Published:   %s", cve["published"])
        logger.info("Severity:    %s", cve["severity"])
        logger.info("Description: %s", cve["description"])
        logger.info("CPE matches (%d):", len(cve["cpe_matches"]))
        for cpe in cve["cpe_matches"][:10]:
            parts = parse_cpe_string(cpe)
            logger.info("  %s", parts)
        ann = annotator.annotate_cve(cve)
        if ann:
            logger.info("Annotation:")
            for word, label in ann:
                if label != "O":
                    logger.info("  %-20s %s", word, label)
        return

    # Multi-year batch mode
    logger.info(
        "Downloading CVEs: %d–%d (use_cache=%s)",
        args.start_year, args.end_year, not args.no_cache,
    )
    cves = fetcher.fetch_cves(
        start_year=args.start_year,
        end_year=args.end_year,
        use_cache=not args.no_cache,
    )

    # Filter to CVEs that have CPE matches (needed for annotation)
    cves_with_cpe = [c for c in cves if c.get("cpe_matches")]
    logger.info(
        "Total CVEs: %d | With CPE matches: %d (%.1f%%)",
        len(cves), len(cves_with_cpe),
        100 * len(cves_with_cpe) / max(len(cves), 1),
    )

    # Annotate
    logger.info("Annotating %d CVEs ...", len(cves_with_cpe))
    annotations = annotator.annotate_batch(cves_with_cpe)

    # Filter empty annotations
    annotations = [a for a in annotations if a]
    logger.info("Non-empty annotations: %d", len(annotations))

    # Statistics
    stats = annotator.get_label_statistics(annotations)
    logger.info("Label distribution:")
    for label, count in sorted(stats.items(), key=lambda x: -x[1]):
        if count > 0:
            logger.info("  %-15s %8d", label, count)

    # Save
    ext = "bio" if args.format == "bio" else "jsonl"
    output_file = output_dir / f"cves_{args.start_year}_{args.end_year}.{ext}"
    annotator.save_annotations(annotations, str(output_file), format=args.format)
    logger.info("Done. Saved to %s", output_file)

    # Also save raw CVEs (for unlabeled auto-annotation later)
    raw_output = Path("data/raw") / f"cves_{args.start_year}_{args.end_year}.jsonl"
    fetcher.save_cves(cves, str(raw_output))
    logger.info("Raw CVEs saved to %s", raw_output)


if __name__ == "__main__":
    main()
