"""
NVD CVE Fetcher
===============
Fetches CVE data from the NVD API 2.0.
Handles rate limiting, pagination, caching, and CPE metadata extraction.

NVD API docs: https://nvd.nist.gov/developers/vulnerabilities
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class NVDFetcher:
    """Fetches CVE data from NVD API 2.0 with rate limiting and caching."""

    BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    RESULTS_PER_PAGE = 2000

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "data/raw/nvd_cache",
        delay: float = 0.6,
    ):
        """
        Args:
            api_key: NVD API key (env var NVD_API_KEY). Without key: ~5 req/30s.
                     With key: ~50 req/30s.
            cache_dir: Directory to cache raw API responses.
            delay: Seconds between requests (default: 0.6 for unauthenticated).
        """
        self.api_key = api_key or os.getenv("NVD_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = 0.3 if self.api_key else delay

        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"apiKey": self.api_key})
        self.session.headers.update({"User-Agent": "CPE-Identifier/1.0"})

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def fetch_cves(
        self,
        start_year: int = 2020,
        end_year: int = 2023,
        keyword: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[Dict]:
        """Fetch CVEs for a year range, returning list of parsed CVE dicts.

        Each dict contains:
            id           — CVE-ID string
            description  — English description text
            published    — ISO timestamp string
            severity     — CVSS severity (CRITICAL/HIGH/MEDIUM/LOW/NONE)
            cpe_matches  — list of CPE 2.3 strings from configurations
        """
        all_cves: List[Dict] = []
        for year in range(start_year, end_year + 1):
            cves = self._fetch_year(year, keyword=keyword, use_cache=use_cache)
            logger.info("Year %d: fetched %d CVEs", year, len(cves))
            all_cves.extend(cves)
        return all_cves

    def fetch_cve_by_id(self, cve_id: str) -> Optional[Dict]:
        """Fetch a single CVE by its ID (e.g. 'CVE-2021-44228')."""
        url = f"{self.BASE_URL}?cveId={cve_id}"
        data = self._request(url)
        if data and data.get("vulnerabilities"):
            return self._parse_cve(data["vulnerabilities"][0]["cve"])
        return None

    def stream_cves(
        self,
        start_date: str,
        end_date: str,
    ) -> Generator[Dict, None, None]:
        """Stream CVEs between ISO date strings (YYYY-MM-DDThh:mm:ss.000).

        Yields parsed CVE dicts. Useful for large ranges without RAM pressure.
        """
        start_index = 0
        total_results = None

        while True:
            params = {
                "pubStartDate": start_date,
                "pubEndDate": end_date,
                "resultsPerPage": self.RESULTS_PER_PAGE,
                "startIndex": start_index,
            }
            data = self._request(self.BASE_URL, params=params)
            if not data:
                break

            if total_results is None:
                total_results = data.get("totalResults", 0)
                logger.info("Total CVEs in range: %d", total_results)

            for item in data.get("vulnerabilities", []):
                yield self._parse_cve(item["cve"])

            start_index += len(data.get("vulnerabilities", []))
            if start_index >= total_results:
                break

            time.sleep(self.delay)

    def save_cves(self, cves: List[Dict], output_path: str) -> None:
        """Save parsed CVEs as JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for cve in cves:
                f.write(json.dumps(cve) + "\n")
        logger.info("Saved %d CVEs to %s", len(cves), output_path)

    def load_cves(self, path: str) -> List[Dict]:
        """Load CVEs from a JSONL file."""
        cves = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cves.append(json.loads(line))
        return cves

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _fetch_year(
        self,
        year: int,
        keyword: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[Dict]:
        cache_file = self.cache_dir / f"cves_{year}.jsonl"
        if use_cache and cache_file.exists():
            logger.info("Loading cached CVEs for %d from %s", year, cache_file)
            return self.load_cves(str(cache_file))

        start_date = f"{year}-01-01T00:00:00.000"
        end_date = f"{year}-12-31T23:59:59.999"
        cves = list(
            tqdm(
                self.stream_cves(start_date, end_date),
                desc=f"Fetching {year} CVEs",
                unit="CVE",
            )
        )
        self.save_cves(cves, str(cache_file))
        return cves

    def _request(
        self,
        url: str,
        params: Optional[Dict] = None,
        retries: int = 3,
    ) -> Optional[Dict]:
        for attempt in range(retries):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    time.sleep(self.delay)
                    return resp.json()
                elif resp.status_code == 429:
                    wait = 30 * (attempt + 1)
                    logger.warning("Rate limited — waiting %ds", wait)
                    time.sleep(wait)
                elif resp.status_code == 403:
                    logger.error("403 Forbidden — check your API key")
                    return None
                else:
                    logger.warning(
                        "HTTP %d for %s (attempt %d)",
                        resp.status_code, url, attempt + 1,
                    )
                    time.sleep(2 ** attempt)
            except requests.RequestException as e:
                logger.warning("Request error (attempt %d): %s", attempt + 1, e)
                time.sleep(2 ** attempt)
        return None

    @staticmethod
    def _parse_cve(cve_data: Dict) -> Dict:
        """Parse raw NVD CVE JSON into a clean dict."""
        cve_id = cve_data.get("id", "")

        # English description
        description = ""
        for desc in cve_data.get("descriptions", []):
            if desc.get("lang") == "en":
                description = desc.get("value", "").strip()
                break

        # Published / modified dates
        published = cve_data.get("published", "")
        modified = cve_data.get("lastModified", "")

        # CVSS severity
        severity = "NONE"
        metrics = cve_data.get("metrics", {})
        for metric_key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            metric_list = metrics.get(metric_key, [])
            if metric_list:
                severity = (
                    metric_list[0]
                    .get("cvssData", {})
                    .get("baseSeverity", "NONE")
                )
                break

        # CPE matches from configurations
        cpe_matches: List[str] = []
        for node in _iter_config_nodes(cve_data.get("configurations", [])):
            for match in node.get("cpeMatch", []):
                if match.get("vulnerable") and match.get("criteria"):
                    cpe_matches.append(match["criteria"])

        return {
            "id": cve_id,
            "description": description,
            "published": published,
            "modified": modified,
            "severity": severity,
            "cpe_matches": list(set(cpe_matches)),
        }


def _iter_config_nodes(configurations: List[Dict]) -> Generator[Dict, None, None]:
    """Recursively yield all CPE-match nodes from NVD configuration trees."""
    for config in configurations:
        for node in config.get("nodes", []):
            yield from _iter_nodes(node)


def _iter_nodes(node: Dict) -> Generator[Dict, None, None]:
    yield node
    for child in node.get("children", []):
        yield from _iter_nodes(child)
