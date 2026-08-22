"""
CBN Regulatory Corpus Scraper
==============================
Collects CBN circulars, guidelines, and regulatory publications
relevant to AML/KYC compliance in Nigeria.

Author: Team Ogun — ICC Product
Date: August 2026
"""

import requests
from bs4 import BeautifulSoup
import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CBNCircularScraper:
    """
    Scraper for collecting CBN regulatory circulars and guidelines
    from the official CBN website.
    """

    def __init__(
        self,
        base_url: str = "https://www.cbn.gov.ng/",
        output_dir: str = "./data/raw_corpus",
        request_delay: float = 1.5
    ):
        """
        Initialize the scraper.

        Args:
            base_url: Base URL of the CBN website.
            output_dir: Directory to save scraped data.
            request_delay: Delay between requests in seconds (politeness).
        """
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.request_delay = request_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ),
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make a polite HTTP request with error handling."""
        try:
            time.sleep(self.request_delay)
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.warning(f"Request failed for {url}: {e}")
            return None

    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse date strings from the CBN website."""
        date_formats = [
            '%d %B %Y',
            '%B %d, %Y',
            '%d-%m-%Y',
            '%Y-%m-%d',
        ]
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.strip(), fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        return None

    def _extract_pdf_links(self, soup: BeautifulSoup, page_url: str) -> list[dict]:
        """Extract PDF document links from a parsed HTML page."""
        documents = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Handle relative URLs
            if href.startswith('/'):
                href = f"{self.base_url}{href}"

            if href.endswith('.pdf') or href.endswith('.PDF'):
                title = link.get_text(strip=True)
                date_text = ""
                # Try to find a date near the link
                parent = link.parent
                if parent:
                    date_match = re.search(
                        r'(\d{1,2}[\s\-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
                        r'[a-z]*[\s\-]\d{4})',
                        parent.get_text(),
                        re.IGNORECASE
                    )
                    if date_match:
                        date_text = self._parse_date(date_match.group(1))

                documents.append({
                    'title': title,
                    'url': href,
                    'date': date_text,
                    'source_page': page_url,
                    'collected_at': datetime.utcnow().isoformat(),
                })

        return documents

    def scrape_circulars_page(self, page_url: str) -> list[dict]:
        """Scrape a single CBN publications page for circular links."""
        logger.info(f"Scraping: {page_url}")
        response = self._make_request(page_url)
        if not response:
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        return self._extract_pdf_links(soup, page_url)

    def scrape_regulatory_corpus(
        self,
        start_page: int = 1,
        end_page: int = 5
    ) -> list[dict]:
        """
        Scrape multiple pages of CBN circulars.

        Args:
            start_page: First page to scrape.
            end_page: Last page to scrape.

        Returns:
            List of scraped document metadata dictionaries.
        """
        all_documents = []
        circulars_url = f"{self.base_url}out/2026/ccd/"

        for page_num in range(start_page, end_page + 1):
            url = f"{circulars_url}?page={page_num}"
            documents = self.scrape_circulars_page(url)
            all_documents.extend(documents)
            logger.info(f"Page {page_num}: found {len(documents)} documents")

        logger.info(f"Total documents collected: {len(all_documents)}")
        return all_documents

    def save_corpus(self, documents: list[dict], filename: str = "cbn_regulatory_corpus.json") -> None:
        """Save the scraped corpus to a JSON file."""
        output_file = self.output_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'source': 'CBN Official Website',
                    'collected_at': datetime.utcnow().isoformat(),
                    'total_documents': len(documents),
                    'collection_scope': 'AML/KYC Regulatory Circulars',
                },
                'documents': documents,
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Corpus saved to {output_file}")

    def download_pdfs(self, documents: list[dict], max_downloads: int = 10) -> None:
        """Download PDF files for text extraction."""
        pdf_dir = self.output_dir / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        for i, doc in enumerate(documents[:max_downloads]):
            response = self._make_request(doc['url'])
            if response and response.headers.get('content-type', '').startswith('application/pdf'):
                filename = f"{i+1}_{re.sub(r'[^a-zA-Z0-9]', '_', doc['title'])[:50]}.pdf"
                filepath = pdf_dir / filename
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Downloaded: {filename}")


class NFIUScraper:
    """Scraper for NFIU (Nigerian Financial Intelligence Unit) advisories."""

    def __init__(self, base_url: str = "https://nfiu.gov.ng/"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
        })

    def fetch_advisories(self) -> list[dict]:
        """Fetch NFIU advisories and alerts."""
        # Placeholder: would scrape actual NFIU advisory pages
        return []


class CBNDataValidator:
    """Validates scraped regulatory data for completeness and correctness."""

    VALID_TIER_LABELS = {'Tier 1', 'Tier 2', 'Tier 3', 'KYC Tier 1', 'KYC Tier 2', 'KYC Tier 3'}
    VALID_RISK_LABELS = {'Low', 'Medium', 'High', 'Critical', 'Suspicious', 'Normal'}
    BVN_PATTERN = re.compile(r'^\d{11}$')
    NIN_PATTERN = re.compile(r'^\d{11}$')

    @staticmethod
    def validate_document_structure(doc: dict) -> bool:
        """Validate that a document entry has required fields."""
        required_fields = ['title', 'text', 'tier_label', 'obligation_category', 'risk_flag']
        for field in required_fields:
            if field not in doc:
                logger.warning(f"Missing field: {field}")
                return False
        return True

    @staticmethod
    def validate_bvn(bvn: str) -> bool:
        """Validate BVN format (11 digits)."""
        return bool(CBNDataValidator.BVN_PATTERN.match(bvn))

    @staticmethod
    def validate_nin(nin: str) -> bool:
        """Validate NIN format (11 digits)."""
        return bool(CBNDataValidator.NIN_PATTERN.match(nin))

    @staticmethod
    def validate_tier_label(label: str) -> bool:
        """Validate that a tier label is one of the recognized values."""
        return label.strip() in CBNDataValidator.VALID_TIER_LABELS


if __name__ == "__main__":
    scraper = CBNCircularScraper(output_dir="./data/raw_corpus")
    corpus = scraper.scrape_regulatory_corpus(start_page=1, end_page=3)
    scraper.save_corpus(corpus)

    # Validate collected data
    validator = CBNDataValidator()
    valid_bvn = validator.validate_bvn("12345678901")
    logger.info(f"BVN validation test: {valid_bvn}")
