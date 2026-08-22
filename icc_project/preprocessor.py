"""
ICC Preprocessing Pipeline
===========================
Full text preprocessing pipeline for Nigerian financial compliance data.
Handles normalization, Nigerian Pidgin translation, BVN/NIN entity recognition,
and tokenization for the ICC transformer model.

Author: Team Ogun — ICC Product
Date: August 2026
"""

import re
import unicodedata
import logging
from typing import Optional
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """Represents a fully processed document ready for model input."""
    original_text: str
    normalized_text: str
    pidgin_translated: str
    masked_text: str
    entities: list[dict]
    token_count: int
    status: str = "processed"


class NigerianEntityRecognizer:
    """
    Recognizes and extracts Nigerian-specific entities from text,
    including BVN, NIN, Nigerian names, and address patterns.
    """

    # BVN: exactly 11 digits
    BVN_PATTERN = re.compile(
        r'(?<!\d)\d{3}\s?\d{3}\s?\d{2}\s?\d{3}(?!\d)'
    )

    # NIN: 11 digits, may appear with "NIN" label
    NIN_PATTERN = re.compile(
        r'(?i)(?:NIN|National\s+Identification\s+Number)?\s*[:\-]?\s*(?<!\d)\d{3}\s?\d{3}\s?\d{2}\s?\d{3}(?!\d)'
    )

    # Nigerian name patterns (Yoruba, Hausa, Igbo common structures)
    NAME_PATTERN = re.compile(
        r'(?:(?:Olu|Ade|Abi|Tai|Ife|Oluwaseun|Chukwu|Emeka|Ibrahim|Musa|Aisha|'
        r'Olumide|Abimbola|Chinedu|Yusuf|Fatima|Ngozi|Obinna|Babatunde|'
        r'Adebayo|Okechukwu|Abubakar|Aminu|Chinwe|Ifeanyi)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
    )

    # Nigerian phone number patterns
    PHONE_PATTERN = re.compile(
        r'(?:(?:\+234|0)\s?(?:[789]\d)\s?\d{3}\s?\d{3}\s?\d{3})'
    )

    # Nigerian address patterns
    ADDRESS_PATTERN = re.compile(
        r'(?:\d+|[A-Z][a-z]+\s+\d+|(?:No\.?|Plot|Plot\s+\d+|House\s+No\.?))'
        r'\s+.+?\s+'
        r'(?:Lagos|Abuja|Kano|Ibadan|Port\s+Harcourt|Benin|Enugu|Jos|'
        r'Ilorin|Warri|Owerri|Abeokuta|Akure|Osogbo|Maiduguri|Sokoto|'
        r'Calabar|Uyo|Lokoja|Makurdi|Yenagoa|Asaba|Awka|Damaturu|Dutse|'
        r'Gusau|Jalingo|Lafia|Damask|Minna|Gombe|Yola|Bauchi|Birnin\s+Kebbi|'
        r'Wukari|Lokoja)\s+State'
    )

    def extract_entities(self, text: str) -> list[dict]:
        """Extract all recognized entities from text."""
        entities = []

        # Extract BVN
        for match in self.BVN_PATTERN.finditer(text):
            clean_bvn = re.sub(r'\s+', '', match.group(0))
            entities.append({
                'type': 'BVN',
                'text': match.group(0),
                'normalized': clean_bvn,
                'start': match.start(),
                'end': match.end(),
            })

        # Extract NIN
        for match in self.NIN_PATTERN.finditer(text):
            digits = re.findall(r'\d+', match.group(0))
            cleaned = ''.join(digits) if digits else match.group(0)
            entities.append({
                'type': 'NIN',
                'text': match.group(0),
                'normalized': cleaned,
                'start': match.start(),
                'end': match.end(),
            })

        # Extract names
        for match in self.NAME_PATTERN.finditer(text):
            entities.append({
                'type': 'PERSON_NG',
                'text': match.group(0),
                'normalized': match.group(0).strip(),
                'start': match.start(),
                'end': match.end(),
            })

        # Extract phone numbers
        for match in self.PHONE_PATTERN.finditer(text):
            entities.append({
                'type': 'PHONE_NG',
                'text': match.group(0),
                'normalized': re.sub(r'\s+', '', match.group(0)),
                'start': match.start(),
                'end': match.end(),
            })

        return entities

    def mask_entities(self, text: str, entities: list[dict]) -> str:
        """Replace entities with placeholder tokens for model input."""
        # Sort by position descending to avoid offset shifts
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)

        for entity in sorted_entities:
            start, end = entity['start'], entity['end']
            replacement = f"<{entity['type']}>"
            text = text[:start] + replacement + text[end:]

        return text


class NigerianPidginNormalizer:
    """
    Translates and normalizes Nigerian Pidgin English text
    to standard English for compliance processing.
    """

    # Comprehensive Pidgin-to-English mapping for compliance context
    PIDGIN_TO_ENGLISH = {
        # Verbs
        r'\bdey\b': 'are/is',
        r'\bcomot\b': 'remove',
        r'\bbring\b': 'bring',
        r'\bcom\b': 'come',
        r'\bgo\b(?!\w)': 'will',
        r'\bwak\b': 'wake',
        r'\bsleep\b': 'sleep',
        r'\beat\b': 'eat',
        r'\bchop\b': 'eat',
        r'\bdrink\b': 'drink',
        r'\brun\b': 'run',
        r'\bsmall\b': 'small',
        r'\bshow\b': 'show',
        r'\btell\b': 'tell',
        r'\bsay\b': 'say',
        r'\bgive\b': 'give',
        r'\btake\b': 'take',
        r'\bfind\b': 'find',
        r'\bknow\b': 'know',
        r'\bsabey\b': 'understand',
        r'\bunderstand\b': 'understand',
        r'\bhear\b': 'hear',
        r'\bsee\b': 'see',
        r'\bwahala\b': 'problem',
        r'\btrouble\b': 'trouble',
        r'\bmoney\b': 'money',
        r'\balert\b': 'bank transfer notification',
        r'\bcash\b': 'cash',
        r'\baccount\b': 'account',
        r'\bbank\b': 'bank',

        # Pidgin-specific constructs
        r'\bna\b': 'is/it is',
        r'\bwey\b': 'that/which',
        r'\bwho\b': 'who',
        r'\bwetin\b': 'what',
        r'\bwhere\b': 'where',
        r'\bwhen\b': 'when',
        r'\bhow\b': 'how',
        r'\bwhy\b': 'why',
        r'\bbcos\b': 'because',
        r'\bsef\b': 'even',
        r'\bsha\b': 'anyway',
        r'\babi\b': 'right?',
        r'\bno be\b': 'is not',
        r'\bno dey\b': 'does not',
        r'\bI don\b': 'I have',
        r'\bI dey\b': 'I am',
        r'\bhe don\b': 'he has',
        r'\bshe don\b': 'she has',
        r'\bthey don\b': 'they have',
        r'\bwe don\b': 'we have',
        r'\byou don\b': 'you have',
        r'\bmake I\b': 'let me',
        r'\bmake we\b': 'let us',
        r'\bmake you\b': 'let you',
        r'\bmake e\b': 'let it',
    }

    # Pidgin negation patterns
    NEGATION_PATTERNS = [
        (re.compile(r'\bno\s+be\b', re.IGNORECASE), 'is not'),
        (re.compile(r'\bno\s+dey\b', re.IGNORECASE), 'does not'),
        (re.compile(r'\bno\s+get\b', re.IGNORECASE), 'does not have'),
        (re.compile(r'\bno\s+fit\b', re.IGNORECASE), 'cannot'),
        (re.compile(r'\bno\s+go\b', re.IGNORECASE), 'will not'),
        (re.compile(r'\bno\s+want\b', re.IGNORECASE), 'does not want'),
        (re.compile(r'\bno\s+like\b', re.IGNORECASE), 'does not like'),
    ]

    def __init__(self, aggressive: bool = False):
        """
        Initialize the normalizer.

        Args:
            aggressive: If True, applies all translations. If False,
                       only translates high-confidence compliance-relevant terms.
        """
        self.aggressive = aggressive

    def translate(self, text: str) -> str:
        """Translate Nigerian Pidgin to English."""
        if not self.aggressive:
            # Conservative mode: only translate key compliance terms
            text = self._translate_compliance_terms(text)
        else:
            # Aggressive mode: translate all Pidgin patterns
            text = self._translate_full(text)
        return text

    def _translate_compliance_terms(self, text: str) -> str:
        """Translate only compliance-critical Pidgin terms."""
        compliance_terms = {
            r'\bwahala\b': 'problem',
            r'\balert\b': 'bank transfer',
            r'\bcash\b': 'cash',
            r'\baccount\b': 'account',
            r'\bmoney\b': 'money',
        }
        for pidgin, english in compliance_terms.items():
            text = re.sub(pidgin, english, text, flags=re.IGNORECASE)
        return text

    def _translate_full(self, text: str) -> str:
        """Apply full Pidgin translation."""
        # First handle negation patterns
        for pattern, replacement in self.NEGATION_PATTERNS:
            text = pattern.sub(replacement, text)

        # Then apply word-level translations
        for pidgin, english in self.PIDGIN_TO_ENGLISH.items():
            text = re.sub(pidgin, english, text, flags=re.IGNORECASE)

        return text


class TextNormalizer:
    """Handles Unicode normalization, whitespace cleanup, and general text cleaning."""

    def __init__(self):
        # Patterns to clean
        self.url_pattern = re.compile(
            r'https?://[^\s<>\"{}|\\^`\[\]]+'
        )
        self.email_pattern = re.compile(
            r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
        )
        self.multispace_pattern = re.compile(r'\s{2,}')
        self.numeric_reference_pattern = re.compile(r'\[\d+\]')

    def normalize(self, text: str) -> str:
        """Apply all normalization steps to text."""
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)

        # Remove URLs
        text = self.url_pattern.sub('', text)

        # Remove email addresses
        text = self.email_pattern.sub('[EMAIL]', text)

        # Remove citation references
        text = self.numeric_reference_pattern.sub('', text)

        # Normalize whitespace
        text = self.multispace_pattern.sub(' ', text).strip()

        # Normalize case for consistency (preserve for some tasks)
        text = re.sub(r'\s+', ' ', text)

        return text


class ICCPreprocessor:
    """
    Main preprocessing orchestrator that combines normalization,
    Pidgin handling, and entity recognition.
    """

    def __init__(
        self,
        mask_entities: bool = True,
        translate_pidgin: bool = True,
        pidgin_aggressive: bool = False,
        max_length: int = 512,
    ):
        """
        Initialize the ICC preprocessor.

        Args:
            mask_entities: Whether to mask BVN/NIN entities.
            translate_pidgin: Whether to translate Nigerian Pidgin.
            pidgin_aggressive: Level of Pidgin translation aggressiveness.
            max_length: Maximum text length after processing.
        """
        self.mask_entities = mask_entities
        self.translate_pidgin = translate_pidgin
        self.pidgin_aggressive = pidgin_aggressive
        self.max_length = max_length

        self.normalizer = TextNormalizer()
        self.pidgin_normalizer = NigerianPidginNormalizer(aggressive=pidgin_aggressive)
        self.ner = NigerianEntityRecognizer()

    def process_document(self, doc: dict) -> ProcessedDocument:
        """
        Process a document through the full preprocessing pipeline.

        Args:
            doc: Dictionary with 'text' key and optional metadata.

        Returns:
            ProcessedDocument with all processing stages completed.
        """
        original_text = doc.get('text', '')

        if not original_text or not isinstance(original_text, str):
            logger.warning(f"Empty or invalid text in document: {doc.get('record_id', 'unknown')}")
            return ProcessedDocument(
                original_text=str(original_text),
                normalized_text="",
                pidgin_translated="",
                masked_text="",
                entities=[],
                token_count=0,
                status="error"
            )

        # Step 1: Normalize
        normalized = self.normalizer.normalize(original_text)

        # Step 2: Translate Pidgin
        pidgin_translated = self.pidgin_normalizer.translate(normalized) if self.translate_pidgin else normalized

        # Step 3: Extract entities
        entities = self.ner.extract_entities(pidgin_translated)

        # Step 4: Mask entities
        masked = self.ner.mask_entities(pidgin_translated, entities) if self.mask_entities else pidgin_translated

        # Step 5: Truncate
        words = masked.split()
        if len(words) > self.max_length:
            masked = ' '.join(words[:self.max_length])

        token_count = len(masked.split())

        result = ProcessedDocument(
            original_text=original_text,
            normalized_text=normalized,
            pidgin_translated=pidgin_translated,
            masked_text=masked,
            entities=entities,
            token_count=token_count,
        )

        return result

    def process_batch(self, documents: list[dict]) -> list[ProcessedDocument]:
        """Process a batch of documents."""
        results = []
        for doc in documents:
            result = self.process_document(doc)
            results.append(result)
        logger.info(f"Processed batch of {len(results)} documents")
        return results
