"""
Annotation Schema & Tools for ICC System
=========================================
Defines the annotation schema for KYC tiers, obligation categories,
and risk flags. Includes tools for creating and managing annotations.

Author: Team Ogun — ICC Product
Date: August 2026
"""

import json
import csv
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Annotation Schema Enums
# =============================================================================

class KYCTier(str, Enum):
    """CBN KYC Tier classifications."""
    TIER_1 = "Tier 1"
    TIER_2 = "Tier 2"
    TIER_3 = "Tier 3"


class ObligationCategory(str, Enum):
    """CBN regulatory obligation categories."""
    KYC_DUE_DILIGENCE = "KYC/Due Diligence"
    CUSTOMER_IDENTIFICATION = "Customer Identification"
    TRANSACTION_MONITORING = "Transaction Monitoring"
    SUSPICIOUS_ACTIVITY_REPORTING = "Suspicious Activity Reporting"
    RECORD_RETENTION = "Record Retention"
    ENHANCED_DUE_DILIGENCE = "Enhanced Due Diligence"
    PEP_SCREENING = "PEP Screening"
    SANCTIONS_SCREENING = "Sanctions Screening"
    CROSS_BORDER_TRANSACTIONS = "Cross-Border Transactions"
    BVN_NIN_VERIFICATION = "BVN/NIN Verification"
    BENEFICIAL_OWNERSHIP = "Beneficial Ownership"
    ANNUAL_REPORTING = "Annual Reporting"
    INTERNAL_CONTROL = "Internal Control"
    STAFF_TRAINING = "Staff Training"
    RISK_ASSESSMENT = "Risk Assessment"


class RiskLevel(str, Enum):
    """Risk flag levels for compliance decisions."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class RiskFlag(str, Enum):
    """Specific risk flag types."""
    STRUCTURING = "Structuring/Smurfing"
    UNUSUAL_TRANSACTION = "Unusual Transaction Pattern"
    PEP_MATCH = "PEP Match"
    SANCTIONS_HIT = "Sanctions Hit"
    IDENTITY_MISMATCH = "Identity Mismatch"
    HIGH_RISK_JURISDICTION = "High-Risk Jurisdiction"
    LARGE_CASH_TRANSACTION = "Large Cash Transaction"
    RAPID_MOVEMENT = "Rapid Fund Movement"
    SHELL_COMPANY = "Shell Company Indicator"
    COMPLIANT = "Compliant"


class EntityRecognitionType(str, Enum):
    """Types of Nigerian entities to recognize."""
    BVN = "BVN"
    NIN = "NIN"
    PERSON_NAME = "PERSON"
    COMPANY_NAME = "ORG"
    ADDRESS = "ADDRESS"
    PHONE = "PHONE"
    ACCOUNT_NUMBER = "ACCOUNT"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Annotation:
    """A single annotation for a compliance document."""
    annotation_id: str
    document_id: str
    kyc_tier: str
    obligation_categories: list[str]
    risk_flags: list[str]
    risk_level: str
    entities: list[dict]
    annotator_id: str
    confidence: float = 0.0
    notes: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class AnnotationBatch:
    """A batch of annotations with metadata."""
    batch_id: str
    annotations: list[Annotation]
    created_at: str = ""
    dataset_version: str = "1.0"

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


# =============================================================================
# Annotation Tools
# =============================================================================

class AnnotationManager:
    """
    Manages creation, validation, and export of compliance annotations.
    """

    VALID_TIERS = {e.value for e in KYCTier}
    VALID_OBLIGATIONS = {e.value for e in ObligationCategory}
    VALID_RISK_LEVELS = {e.value for e in RiskLevel}
    VALID_RISK_FLAGS = {e.value for e in RiskFlag}

    def __init__(self, output_dir: str = "./data/annotations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_annotation(
        self,
        document_id: str,
        kyc_tier: str,
        obligation_categories: list[str],
        risk_flags: list[str],
        risk_level: str,
        entities: list[dict] = None,
        annotator_id: str = "system",
        confidence: float = 0.0,
        notes: str = ""
    ) -> Annotation:
        """
        Create a new annotation with validation.

        Args:
            document_id: ID of the document being annotated.
            kyc_tier: KYC tier classification.
            obligation_categories: List of obligation categories.
            risk_flags: List of risk flags.
            risk_level: Overall risk level.
            entities: List of recognized entities.
            annotator_id: ID of the annotator.
            confidence: Confidence score (0-1).
            notes: Optional notes.

        Returns:
            Validated Annotation object.

        Raises:
            ValueError: If any field fails validation.
        """
        # Validate fields
        if kyc_tier not in self.VALID_TIERS:
            raise ValueError(f"Invalid KYC tier: {kyc_tier}. Must be one of {self.VALID_TIERS}")

        for ob in obligation_categories:
            if ob not in self.VALID_OBLIGATIONS:
                raise ValueError(f"Invalid obligation category: {ob}")

        for rf in risk_flags:
            if rf not in self.VALID_RISK_FLAGS:
                raise ValueError(f"Invalid risk flag: {rf}")

        if risk_level not in self.VALID_RISK_LEVELS:
            raise ValueError(f"Invalid risk level: {risk_level}")

        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")

        annotation = Annotation(
            annotation_id=f"ANN-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{document_id}",
            document_id=document_id,
            kyc_tier=kyc_tier,
            obligation_categories=obligation_categories,
            risk_flags=risk_flags,
            risk_level=risk_level,
            entities=entities or [],
            annotator_id=annotator_id,
            confidence=confidence,
            notes=notes,
        )

        logger.info(f"Created annotation: {annotation.annotation_id}")
        return annotation

    def validate_annotation(self, annotation: Annotation) -> bool:
        """Validate an existing annotation against the schema."""
        try:
            self.create_annotation(
                document_id=annotation.document_id,
                kyc_tier=annotation.kyc_tier,
                obligation_categories=annotation.obligation_categories,
                risk_flags=annotation.risk_flags,
                risk_level=annotation.risk_level,
                entities=annotation.entities,
                annotator_id=annotation.annotator_id,
                confidence=annotation.confidence,
            )
            return True
        except ValueError:
            return False

    def export_annotations(
        self,
        annotations: list[Annotation],
        format: str = "json",
        filename: Optional[str] = None
    ) -> str:
        """
        Export annotations to a file.

        Args:
            annotations: List of annotations to export.
            format: Output format ('json' or 'csv').
            filename: Optional filename.

        Returns:
            Path to the exported file.
        """
        if format == "json":
            filepath = filename or str(self.output_dir / "annotations.json")
            data = [asdict(a) for a in annotations]
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'total_annotations': len(annotations),
                        'exported_at': datetime.utcnow().isoformat(),
                        'schema_version': '1.0',
                    },
                    'annotations': data,
                }, f, indent=2, ensure_ascii=False)
        elif format == "csv":
            filepath = filename or str(self.output_dir / "annotations.csv")
            fieldnames = [
                'annotation_id', 'document_id', 'kyc_tier',
                'obligation_categories', 'risk_flags', 'risk_level',
                'annotator_id', 'confidence', 'notes', 'timestamp'
            ]
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for ann in annotations:
                    row = asdict(ann)
                    row['obligation_categories'] = '|'.join(row['obligation_categories'])
                    row['risk_flags'] = '|'.join(row['risk_flags'])
                    row['entities'] = json.dumps(row['entities'])
                    del row['entities']
                    writer.writerow(row)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported {len(annotations)} annotations to {filepath}")
        return filepath


class DataAugmenter:
    """
    Generates augmented training data from existing annotations
    using synonym replacement, back-translation simulation, and
    Nigerian Pidgin code-mixing.
    """

    # Compliance-related synonyms for augmentation
    COMPLIANCE_SYNONYMS = {
        'suspicious': ['unusual', 'dubious', 'questionable', 'irregular'],
        'transaction': ['transfer', 'movement', 'exchange', 'payment'],
        'customer': ['client', 'account holder', 'subscriber'],
        'verify': ['confirm', 'validate', 'authenticate', 'check'],
        'identity': ['identification', 'ID', 'credentials'],
        'risk': ['danger', 'threat', 'hazard', 'exposure'],
        'fraud': ['deception', 'scam', 'swindle', 'deceit'],
        'compliance': ['adherence', 'conformity', 'observance'],
        'monitor': ['track', 'observe', 'watch', 'oversee'],
        'report': ['flag', 'notify', 'alert', 'escalate'],
    }

    # Pidgin code-mixing patterns
    PIDGIN_PATTERNS = {
        'suspicious': 'wahala',
        'problem': 'wahala',
        'money': 'money',
        'alert': 'alert',
        'complaint': 'complaint',
        'trouble': 'wahala',
        'yes': 'yes o',
        'no': 'no be',
        'good': 'good o',
    }

    def __init__(self, augmentation_factor: float = 2.0):
        """
        Initialize the data augmenter.

        Args:
            augmentation_factor: Multiplier for dataset size (e.g., 2.0 = double the data).
        """
        self.augmentation_factor = augmentation_factor

    def augment_text(self, text: str, strategy: str = "synonym") -> str:
        """
        Augment a text string using the specified strategy.

        Args:
            text: Original text to augment.
            strategy: Augmentation strategy ('synonym', 'pidgin_mix', 'delete', 'swap').

        Returns:
            Augmented text string.
        """
        import random

        if strategy == "synonym":
            words = text.split()
            for i, word in enumerate(words):
                word_lower = word.lower().strip('.,!?;:')
                if word_lower in self.COMPLIANCE_SYNONYMS:
                    replacements = self.COMPLIANCE_SYNONYMS[word_lower]
                    if random.random() < 0.3:  # 30% chance of replacement
                        words[i] = random.choice(replacements)
            return ' '.join(words)

        elif strategy == "pidgin_mix":
            for pidgin_term, english_term in self.PIDGIN_PATTERNS.items():
                if pidgin_term in text.lower() and random.random() < 0.2:
                    text = text.replace(pidgin_term, english_term)
            return text

        elif strategy == "delete":
            words = text.split()
            if len(words) > 5:
                delete_idx = random.randint(0, len(words) - 1)
                words.pop(delete_idx)
            return ' '.join(words)

        elif strategy == "swap":
            words = text.split()
            if len(words) > 2:
                idx1 = random.randint(0, len(words) - 2)
                idx2 = idx1 + random.randint(1, min(2, len(words) - idx1 - 1))
                if idx2 < len(words):
                    words[idx1], words[idx2] = words[idx2], words[idx1]
            return ' '.join(words)

        return text

    def augment_dataset(
        self,
        documents: list[dict],
        target_size: int = None
    ) -> list[dict]:
        """
        Augment a dataset of documents.

        Args:
            documents: List of document dictionaries with 'text' key.
            target_size: Target number of documents after augmentation.

        Returns:
            Augmented list of documents.
        """
        strategies = ["synonym", "pidgin_mix", "delete", "swap"]
        augmented = list(documents)

        target = target_size or int(len(documents) * self.augmentation_factor)
        needed = target - len(documents)

        for i in range(needed):
            original = documents[i % len(documents)]
            strategy = strategies[i % len(strategies)]
            augmented_text = self.augment_text(original['text'], strategy)

            augmented_doc = dict(original)
            augmented_doc['text'] = augmented_text
            augmented_doc['record_id'] = f"AUG-{i}_{original.get('record_id', 'unknown')}"
            augmented_doc['augmented'] = True
            augmented_doc['augmentation_strategy'] = strategy
            augmented.append(augmented_doc)

        logger.info(f"Augmented dataset from {len(documents)} to {len(augmented)} documents")
        return augmented
