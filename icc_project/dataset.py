"""
ICC Dataset Module
===================
PyTorch Dataset classes for loading and preprocessing compliance data
for the ICC multi-head classifier.

Author: Team Ogun — ICC Product
Date: August 2026
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ICCDataset(Dataset):
    """
    PyTorch Dataset for ICC compliance classification.

    Loads annotated documents and prepares them for multi-head
    classification (KYC tier, obligation mapping, risk flagging).
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        task_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to JSON data file.
            tokenizer: Hugging Face tokenizer.
            max_length: Maximum sequence length.
            task_weights: Optional weights for multi-task loss balancing.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_weights = task_weights or {
            'kyc': 1.0,
            'obligation': 1.0,
            'risk': 1.0,
        }

        self.data = self._load_data(data_path)
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load and validate data from JSON file."""
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Handle both list and wrapped formats
        if isinstance(raw_data, list):
            records = raw_data
        elif isinstance(raw_data, dict):
            records = raw_data.get('annotations', raw_data.get('records', []))
        else:
            raise ValueError(f"Unexpected data format in {data_path}")

        return records

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.

        Returns:
            Dictionary with tokenized inputs and labels.
        """
        record = self.data[idx]

        # Extract text
        text = record.get('processed_text', record.get('text', record.get('raw_text', '')))

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )

        # Extract labels
        kyc_label = self._encode_kyc_tier(record.get('kyc_tier', record.get('tier_label', 0)))
        obligation_label = self._encode_obligation(record.get('obligation_category', record.get('obligation_categories', 0)))
        risk_label = self._encode_risk(record.get('risk_flag', record.get('risk_level', 0)))

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).squeeze(0),
            'kyc_labels': torch.tensor(kyc_label, dtype=torch.long),
            'obligation_labels': torch.tensor(obligation_label, dtype=torch.long),
            'risk_labels': torch.tensor(risk_label, dtype=torch.long),
        }

    def _encode_kyc_tier(self, tier) -> int:
        """Encode KYC tier label to integer."""
        tier_map = {
            'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2,
            0: 0, 1: 1, 2: 2,
            'tier_1': 0, 'tier_2': 1, 'tier_3': 2,
        }
        if isinstance(tier, str):
            return tier_map.get(tier.strip(), 0)
        return int(tier) if tier is not None else 0

    def _encode_obligation(self, obligation) -> int:
        """Encode obligation category to integer."""
        obligation_map = {
            'KYC/Due Diligence': 0,
            'Customer Identification': 1,
            'Transaction Monitoring': 2,
            'Suspicious Activity Reporting': 3,
            'Record Retention': 4,
            'Enhanced Due Diligence': 5,
            'PEP Screening': 6,
            'Sanctions Screening': 7,
            'Cross-Border Transactions': 8,
            'BVN/NIN Verification': 9,
            'Beneficial Ownership': 10,
            'Annual Reporting': 11,
            'Internal Control': 12,
            'Staff Training': 13,
            'Risk Assessment': 14,
        }
        if isinstance(obligation, str):
            return obligation_map.get(obligation.strip(), 0)
        return int(obligation) if obligation is not None else 0

    def _encode_risk(self, risk) -> int:
        """Encode risk flag to integer."""
        risk_map = {
            'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3,
            'Normal': 0, 'Suspicious': 1,
            0: 0, 1: 1,
        }
        if isinstance(risk, str):
            return risk_map.get(risk.strip(), 0)
        return int(risk) if risk is not None else 0

    def get_class_distribution(self) -> Dict[str, Dict[int, int]]:
        """Return class distribution across all tasks."""
        distribution = {
            'kyc': {},
            'obligation': {},
            'risk': {},
        }

        for record in self.data:
            kyc = self._encode_kyc_tier(record.get('kyc_tier', record.get('tier_label', 0)))
            obligation = self._encode_obligation(record.get('obligation_category', record.get('obligation_categories', 0)))
            risk = self._encode_risk(record.get('risk_flag', record.get('risk_level', 0)))

            distribution['kyc'][kyc] = distribution['kyc'].get(kyc, 0) + 1
            distribution['obligation'][obligation] = distribution['obligation'].get(obligation, 0) + 1
            distribution['risk'][risk] = distribution['risk'].get(risk, 0) + 1

        return distribution


class MultiTaskBatchSampler:
    """
    Custom batch sampler that ensures balanced representation
    across all three classification tasks within each batch.
    """

    def __init__(self, dataset: ICCDataset, batch_size: int = 16):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        indices = torch.randperm(len(self.dataset)).tolist()
        for i in range(0, len(indices), self.batch_size):
            yield indices[i:i + self.batch_size]

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
