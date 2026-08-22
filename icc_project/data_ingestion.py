"""
Data Ingestion Pipeline for ICC System
=======================================
Handles ingestion of various data sources (PDFs, JSON, CSV, API responses)
into a unified format for downstream processing.

Author: Team Ogun — ICC Product
Date: August 2026
"""

import json
import csv
import logging
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ComplianceRecord:
    """Unified record format for all ingested compliance data."""
    record_id: str
    source_type: str  # 'circular', 'transaction', 'onboarding', 'chat', 'suspicious_report'
    raw_text: str
    processed_text: str = ""
    timestamp: str = ""
    source_file: str = ""
    tier_label: Optional[str] = None
    obligation_category: Optional[str] = None
    risk_flag: Optional[str] = None
    customer_id: Optional[str] = None
    bvn: Optional[str] = None
    nin: Optional[str] = None
    metadata: Optional[dict] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class DataIngestionPipeline:
    """
    Multi-source data ingestion pipeline that normalizes data from
    various formats into a unified ComplianceRecord format.
    """

    def __init__(self, output_dir: str = "./data/ingested"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[ComplianceRecord] = []

    def ingest_json(self, filepath: str, source_type: str = "transaction") -> list[ComplianceRecord]:
        """Ingest JSON-formatted data files."""
        logger.info(f"Ingesting JSON: {filepath}")
        path = Path(filepath)
        if not path.exists():
            logger.error(f"File not found: {filepath}")
            return []

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        records = []
        # Handle both list and dict-of-lists formats
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = next(iter(data.values()), [])
        else:
            logger.warning(f"Unexpected data format in {filepath}")
            return []

        for i, item in enumerate(items):
            record = ComplianceRecord(
                record_id=f"{source_type}_{i}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                source_type=source_type,
                raw_text=item.get('text', item.get('description', str(item))),
                tier_label=item.get('tier_label'),
                obligation_category=item.get('obligation_category'),
                risk_flag=item.get('risk_flag'),
                customer_id=item.get('customer_id'),
                bvn=item.get('bvn'),
                nin=item.get('nin'),
                metadata=item.get('metadata', {}),
                source_file=filepath,
            )
            records.append(record)

        self.records.extend(records)
        logger.info(f"Ingested {len(records)} records from {filepath}")
        return records

    def ingest_csv(self, filepath: str, source_type: str = "transaction") -> list[ComplianceRecord]:
        """Ingest CSV-formatted data files."""
        logger.info(f"Ingesting CSV: {filepath}")
        df = pd.read_csv(filepath, encoding='utf-8')
        records = []

        for i, row in df.iterrows():
            text_field = None
            for col in ['text', 'description', 'narration', 'memo', 'content']:
                if col in df.columns:
                    text_field = str(row.get(col, ''))
                    break

            if text_field is None:
                text_field = row.to_dict().__str__()

            record = ComplianceRecord(
                record_id=f"{source_type}_{i}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                source_type=source_type,
                raw_text=text_field,
                tier_label=row.get('tier_label', None),
                obligation_category=row.get('obligation_category', None),
                risk_flag=row.get('risk_flag', None),
                customer_id=row.get('customer_id', None),
                bvn=row.get('bvn', None),
                nin=row.get('nin', None),
                metadata={k: str(v) for k, v in row.items() if k not in
                          ['text', 'description', 'narration', 'memo', 'content',
                           'tier_label', 'obligation_category', 'risk_flag']},
                source_file=filepath,
            )
            records.append(record)

        self.records.extend(records)
        logger.info(f"Ingested {len(records)} records from {filepath}")
        return records

    def ingest_pdf_text(self, filepath: str, source_type: str = "circular") -> list[ComplianceRecord]:
        """Ingest text extracted from PDF documents."""
        logger.info(f"Ingesting PDF text: {filepath}")
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(filepath)
            full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

            # Split into chunks for processing
            chunks = self._chunk_text(full_text, chunk_size=512, overlap=64)
            records = []

            for i, chunk in enumerate(chunks):
                record = ComplianceRecord(
                    record_id=f"pdf_{i}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    source_type=source_type,
                    raw_text=chunk,
                    source_file=filepath,
                )
                records.append(record)

            self.records.extend(records)
            logger.info(f"Ingested {len(records)} chunks from PDF: {filepath}")
            return records

        except ImportError:
            logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")
            return []
        except Exception as e:
            logger.error(f"Error reading PDF {filepath}: {e}")
            return []

    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
        """Split text into overlapping chunks for processing."""
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap

        if not chunks:
            chunks = [text]

        return chunks

    def save_ingested_data(self, filename: str = "ingested_records.json") -> str:
        """Save all ingested records to a JSON file."""
        output_file = self.output_dir / filename
        records_data = [asdict(r) for r in self.records]

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'total_records': len(records_data),
                    'ingested_at': datetime.utcnow().isoformat(),
                    'source_types': list(set(r.source_type for r in self.records)),
                },
                'records': records_data,
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(records_data)} records to {output_file}")
        return str(output_file)

    def get_statistics(self) -> dict:
        """Return statistics about ingested data."""
        source_counts = {}
        for record in self.records:
            source_counts[record.source_type] = source_counts.get(record.source_type, 0) + 1

        return {
            'total_records': len(self.records),
            'source_type_distribution': source_counts,
            'records_with_tier_labels': sum(1 for r in self.records if r.tier_label),
            'records_with_risk_flags': sum(1 for r in self.records if r.risk_flag),
            'records_with_bvn': sum(1 for r in self.records if r.bvn),
            'records_with_nin': sum(1 for r in self.records if r.nin),
        }


if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    # Example: ingest a JSON file
    # pipeline.ingest_json("./data/raw_corpus/cbn_regulatory_corpus.json", source_type="circular")
    # stats = pipeline.get_statistics()
    # print(json.dumps(stats, indent=2))
    logger.info("DataIngestionPipeline ready. Call methods to ingest data.")
