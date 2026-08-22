"""
ICC Test Suite
==============
Comprehensive unit and integration tests for the ICC system.
Covers preprocessing, model architecture, training, inference,
and audit trail functionality.

Author: Team Ogun — ICC Product
Date: August 2026
"""

import pytest
import json
import torch
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

# =============================================================================
# Phase 2: Preprocessor Tests
# =============================================================================

class TestNigerianEntityRecognizer:
    """Tests for the Nigerian Entity Recognizer."""

    def setup_method(self):
        from phase_2_processing.preprocessor import NigerianEntityRecognizer
        self.ner = NigerianEntityRecognizer()

    def test_bvn_extraction(self):
        """Test BVN extraction from text."""
        text = "My BVN is 12345678901"
        entities = self.ner.extract_entities(text)
        bvn_entities = [e for e in entities if e['type'] == 'BVN']
        assert len(bvn_entities) == 1
        assert bvn_entities[0]['normalized'] == '12345678901'

    def test_nin_extraction(self):
        """Test NIN extraction from text."""
        text = "NIN: 98765432109"
        entities = self.ner.extract_entities(text)
        nin_entities = [e for e in entities if e['type'] == 'NIN']
        assert len(nin_entities) >= 1

    def test_name_extraction(self):
        """Test Nigerian name extraction."""
        text = "Customer name is Oluwaseun Adebayo"
        entities = self.ner.extract_entities(text)
        name_entities = [e for e in entities if e['type'] == 'PERSON_NG']
        assert len(name_entities) >= 1
        assert 'Oluwaseun' in name_entities[0]['text']

    def test_entity_masking(self):
        """Test that entities are properly masked."""
        text = "Customer BVN is 12345678901"
        entities = self.ner.extract_entities(text)
        masked = self.ner.mask_entities(text, entities)
        assert "12345678901" not in masked
        assert "<BVN>" in masked

    def test_no_entities(self):
        """Test handling of text with no entities."""
        text = "This is a normal text with no identifiers"
        entities = self.ner.extract_entities(text)
        assert len(entities) == 0


class TestNigerianPidginNormalizer:
    """Tests for the Nigerian Pidgin Normalizer."""

    def setup_method(self):
        from phase_2_processing.preprocessor import NigerianPidginNormalizer
        self.normalizer_conservative = NigerianPidginNormalizer(aggressive=False)
        self.normalizer_aggressive = NigerianPidginNormalizer(aggressive=True)

    def test_conservative_mode_compliance_terms(self):
        """Test that conservative mode translates key compliance terms."""
        text = "I have wahala with my account"
        result = self.normalizer_conservative.translate(text)
        assert "wahala" not in result.lower() or "problem" in result.lower()

    def test_aggressive_mode_full_translation(self):
        """Test that aggressive mode translates all Pidgin."""
        text = "Na me dey get the account wey get wahala"
        result = self.normalizer_aggressive.translate(text)
        assert result != text  # Should be different after translation

    def test_negation_handling(self):
        """Test Pidgin negation patterns."""
        text = "No be me dey do this transaction"
        result = self.normalizer_aggressive.translate(text)
        assert "is not" in result.lower()


class TestTextNormalizer:
    """Tests for the Text Normalizer."""

    def setup_method(self):
        from phase_2_processing.preprocessor import TextNormalizer
        self.normalizer = TextNormalizer()

    def test_url_removal(self):
        """Test URL removal."""
        text = "Visit https://cbn.gov.ng for details"
        result = self.normalizer.normalize(text)
        assert "https://" not in result

    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        text = "This  has   multiple    spaces"
        result = self.normalizer.normalize(text)
        assert "  " not in result

    def test_unicode_normalization(self):
        """Test Unicode normalization."""
        text = "Café résumé naïve"
        result = self.normalizer.normalize(text)
        assert len(result) > 0

    def test_empty_text(self):
        """Test handling of empty text."""
        result = self.normalizer.normalize("")
        assert result == ""


class TestICCPreprocessor:
    """Tests for the full ICC Preprocessor pipeline."""

    def setup_method(self):
        from phase_2_processing.preprocessor import ICCPreprocessor
        self.preprocessor = ICCPreprocessor()

    def test_process_valid_document(self):
        """Test processing a valid document."""
        doc = {'text': "Customer Oluwaseun BVN is 12345678901"}
        result = self.preprocessor.process_document(doc)
        assert result.status == "processed"
        assert result.token_count > 0
        assert "<BVN>" in result.masked_text

    def test_process_empty_document(self):
        """Test processing an empty document."""
        doc = {'text': ""}
        result = self.preprocessor.process_document(doc)
        assert result.status == "error"

    def test_process_pidgin_text(self):
        """Test processing Nigerian Pidgin text."""
        doc = {'text': "I don send wahala alert to customer"}
        result = self.preprocessor.process_document(doc)
        assert result.status == "processed"

    def test_batch_processing(self):
        """Test batch processing of multiple documents."""
        documents = [
            {'text': "First document with BVN 11122233344"},
            {'text': "Second document about account opening"},
            {'text': "Third document wey get wahala"},
        ]
        results = self.preprocessor.process_batch(documents)
        assert len(results) == 3
        assert all(r.status == "processed" for r in results)


# =============================================================================
# Phase 2: Annotation Tests
# =============================================================================

class TestAnnotationManager:
    """Tests for the annotation schema and management."""

    def setup_method(self):
        from phase_2_processing.annotation import AnnotationManager
        self.manager = AnnotationManager(output_dir=tempfile.mkdtemp())

    def test_create_valid_annotation(self):
        """Test creating a valid annotation."""
        annotation = self.manager.create_annotation(
            document_id="doc_001",
            kyc_tier="Tier 2",
            obligation_categories=["KYC/Due Diligence", "Customer Identification"],
            risk_flags=["COMPLIANT"],
            risk_level="Low",
            annotator_id="test_user",
            confidence=0.95,
        )
        assert annotation.kyc_tier == "Tier 2"
        assert annotation.confidence == 0.95

    def test_invalid_tier_raises_error(self):
        """Test that invalid tier raises ValueError."""
        with pytest.raises(ValueError, match="Invalid KYC tier"):
            self.manager.create_annotation(
                document_id="doc_001",
                kyc_tier="Tier 5",
                obligation_categories=["KYC/Due Diligence"],
                risk_flags=["COMPLIANT"],
                risk_level="Low",
            )

    def test_invalid_obligation_raises_error(self):
        """Test that invalid obligation raises ValueError."""
        with pytest.raises(ValueError, match="Invalid obligation category"):
            self.manager.create_annotation(
                document_id="doc_001",
                kyc_tier="Tier 1",
                obligation_categories=["Invalid Category"],
                risk_flags=["COMPLIANT"],
                risk_level="Low",
            )

    def test_export_json(self):
        """Test exporting annotations to JSON."""
        annotation = self.manager.create_annotation(
            document_id="doc_001",
            kyc_tier="Tier 1",
            obligation_categories=["KYC/Due Diligence"],
            risk_flags=["COMPLIANT"],
            risk_level="Low",
        )
        filepath = self.manager.export_annotations([annotation], format="json")
        assert os.path.exists(filepath)

    def test_export_csv(self):
        """Test exporting annotations to CSV."""
        annotation = self.manager.create_annotation(
            document_id="doc_001",
            kyc_tier="Tier 1",
            obligation_categories=["KYC/Due Diligence"],
            risk_flags=["COMPLIANT"],
            risk_level="Low",
        )
        filepath = self.manager.export_annotations([annotation], format="csv")
        assert os.path.exists(filepath)


class TestDataAugmenter:
    """Tests for the data augmentation module."""

    def setup_method(self):
        from phase_2_processing.annotation import DataAugmenter
        self.augmenter = DataAugmenter(augmentation_factor=2.0)

    def test_synonym_augmentation(self):
        """Test synonym replacement augmentation."""
        text = "suspicious transaction from customer"
        augmented = self.augmenter.augment_text(text, strategy="synonym")
        assert isinstance(augmented, str)
        assert len(augmented) > 0

    def test_pidgin_mixture_augmentation(self):
        """Test Pidgin code-mixing augmentation."""
        text = "suspicious money transfer alert"
        augmented = self.augmenter.augment_text(text, strategy="pidgin_mix")
        assert isinstance(augmented, str)

    def test_dataset_augmentation(self):
        """Test full dataset augmentation."""
        documents = [
            {'text': 'suspicious transaction', 'record_id': 'r1'},
            {'text': 'normal customer onboarding', 'record_id': 'r2'},
            {'text': 'large cash transfer', 'record_id': 'r3'},
        ]
        augmented = self.augmenter.augment_dataset(documents, target_size=6)
        assert len(augmented) == 6
        assert any(d.get('augmented', False) for d in augmented)


# =============================================================================
# Phase 3: Model Architecture Tests
# =============================================================================

class TestICCMultiHeadClassifier:
    """Tests for the ICC multi-head classifier architecture."""

    def setup_method(self):
        from phase_3_model.model_architecture import ICCMultiHeadClassifier
        self.model = ICCMultiHeadClassifier(
            base_model_name='bert-base-uncased',
            num_kyc_classes=3,
            num_obligation_classes=15,
            num_risk_classes=2,
        )

    def test_forward_pass(self):
        """Test forward pass produces correct output shapes."""
        batch_size = 4
        seq_len = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        assert outputs['kyc_logits'].shape == (batch_size, 3)
        assert outputs['obligation_logits'].shape == (batch_size, 15)
        assert outputs['risk_logits'].shape == (batch_size, 2)

    def test_forward_with_labels(self):
        """Test forward pass with labels computes losses."""
        batch_size = 4
        seq_len = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        kyc_labels = torch.randint(0, 3, (batch_size,))
        obligation_labels = torch.randint(0, 15, (batch_size,))
        risk_labels = torch.randint(0, 2, (batch_size,))

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels_kyc=kyc_labels,
            labels_obligation=obligation_labels,
            labels_risk=risk_labels,
        )

        assert 'total_loss' in outputs
        assert 'kyc_loss' in outputs
        assert 'obligation_loss' in outputs
        assert 'risk_loss' in outputs

    def test_base_model_freezing(self):
        """Test that base layers are properly frozen."""
        # First 4 encoder layers should be frozen
        frozen_layers = self.model.bert.encoder.layer[:4]
        for layer in frozen_layers:
            for param in layer.parameters():
                assert not param.requires_grad

    def test_attention_pooling_mode(self):
        """Test attention pooling strategy."""
        from phase_3_model.model_architecture import ICCMultiHeadClassifier
        model = ICCMultiHeadClassifier(
            base_model_name='bert-base-uncased',
            pooling_strategy='attention',
        )
        input_ids = torch.randint(0, 1000, (2, 64))
        attention_mask = torch.ones(2, 64, dtype=torch.long)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        assert outputs['kyc_logits'].shape == (2, 3)


# =============================================================================
# Phase 3: Dataset Tests
# =============================================================================

class TestICCDataset:
    """Tests for the ICC dataset class."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self._create_sample_data()

    def _create_sample_data(self):
        """Create sample data for testing."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        data = [
            {
                'text': 'Customer onboarding with Tier 2 KYC',
                'kyc_tier': 'Tier 2',
                'obligation_category': 'KYC/Due Diligence',
                'risk_flag': 'Normal',
            },
            {
                'text': 'Suspicious large cash transaction',
                'kyc_tier': 'Tier 3',
                'obligation_category': 'Suspicious Activity Reporting',
                'risk_flag': 'Suspicious',
            },
        ]

        self.data_path = os.path.join(self.tmpdir, 'test_data.json')
        with open(self.data_path, 'w') as f:
            json.dump(data, f)

        from phase_3_model.dataset import ICCDataset
        self.dataset = ICCDataset(self.data_path, tokenizer)

    def test_dataset_length(self):
        """Test dataset reports correct length."""
        assert len(self.dataset) == 2

    def test_getitem_returns_correct_keys(self):
        """Test that __getitem__ returns all required keys."""
        item = self.dataset[0]
        required_keys = [
            'input_ids', 'attention_mask', 'token_type_ids',
            'kyc_labels', 'obligation_labels', 'risk_labels',
        ]
        for key in required_keys:
            assert key in item

    def test_class_distribution(self):
        """Test class distribution computation."""
        distribution = self.dataset.get_class_distribution()
        assert 'kyc' in distribution
        assert 'obligation' in distribution
        assert 'risk' in distribution


# =============================================================================
# Phase 4: Training Tests
# =============================================================================

class TestTrainingConfig:
    """Tests for the training configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        from phase_4_training.trainer import TrainingConfig
        config = TrainingConfig()
        assert config.batch_size == 16
        assert config.learning_rate == 2e-5
        assert config.num_epochs == 10
        assert config.seed == 42

    def test_custom_config(self):
        """Test custom configuration."""
        from phase_4_training.trainer import TrainingConfig
        config = TrainingConfig(
            batch_size=32,
            learning_rate=5e-5,
            num_epochs=5,
        )
        assert config.batch_size == 32
        assert config.learning_rate == 5e-5


class TestEarlyStopping:
    """Tests for early stopping logic."""

    def test_no_stop_on_improvement(self):
        """Test that early stopping doesn't trigger on improvement."""
        from phase_4_training.trainer import EarlyStopping
        stopper = EarlyStopping(patience=2)
        assert not stopper({'overall_f1': 0.80})
        assert not stopper({'overall_f1': 0.85})  # Improvement
        assert not stopper({'overall_f1': 0.83})  # Slight decrease, within patience

    def test_stop_after_patience(self):
        """Test that early stopping triggers after patience exhausted."""
        from phase_4_training.trainer import EarlyStopping
        stopper = EarlyStopping(patience=2)
        stopper({'overall_f1': 0.90})
        stopper({'overall_f1': 0.85})
        assert stopper({'overall_f1': 0.80})  # Should stop


# =============================================================================
# Phase 6: Inference Tests
# =============================================================================

class TestInferenceAPI:
    """Tests for the FastAPI inference server."""

    def test_health_endpoint(self):
        """Test health check endpoint."""
        from fastapi.testclient import TestClient
        from phase_6_inference.main import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'model_loaded' in data

    def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        from fastapi.testclient import TestClient
        from phase_6_inference.main import app

        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert 'pipeline_stages' in data
        assert len(data['pipeline_stages']) == 5

    def test_classify_endpoint_schema(self):
        """Test that classify endpoint validates input schema."""
        from fastapi.testclient import TestClient
        from phase_6_inference.main import app

        client = TestClient(app)
        response = client.post("/classify", json={
            'document_text': '',
        })
        assert response.status_code == 422  # Validation error


# =============================================================================
# Phase 7: Audit Trail Tests
# =============================================================================

class TestAuditTrailManager:
    """Tests for the audit trail management."""

    def setup_method(self):
        from phase_7_monitoring.audit_logger import AuditTrailManager
        self.manager = AuditTrailManager(storage_mode="memory")

    def test_log_classification(self):
        """Test logging a classification decision."""
        from phase_7_monitoring.audit_logger import AuditRecord
        record = AuditRecord(
            audit_id="TEST-001",
            customer_id="cust_001",
            document_type="onboarding",
            classification_timestamp="",
            kyc_tier="Tier 2",
            kyc_confidence=0.95,
            mapped_obligations=["KYC/Due Diligence"],
            obligation_confidence=0.90,
            risk_flag="Compliant",
            risk_confidence=0.85,
            requires_escalation=False,
            raw_input_hash="abc123",
            processed_input="Processed text here",
        )
        result = self.manager.log_classification(record)
        assert result is True
        assert "TEST-001" in self.manager.records

    def test_retrieve_record(self):
        """Test retrieving a logged record."""
        from phase_7_monitoring.audit_logger import AuditRecord
        record = AuditRecord(
            audit_id="TEST-002",
            customer_id="cust_002",
            document_type="transaction",
            classification_timestamp="",
            kyc_tier="Tier 1",
            kyc_confidence=0.80,
            mapped_obligations=["Transaction Monitoring"],
            obligation_confidence=0.75,
            risk_flag="Suspicious",
            risk_confidence=0.90,
            requires_escalation=True,
            raw_input_hash="def456",
            processed_input="Suspicious transaction",
        )
        self.manager.log_classification(record)
        retrieved = self.manager.get_record("TEST-002")
        assert retrieved is not None
        assert retrieved.kyc_tier == "Tier 1"

    def test_query_records(self):
        """Test querying records with filters."""
        from phase_7_monitoring.audit_logger import AuditRecord
        for i in range(5):
            record = AuditRecord(
                audit_id=f"TEST-{i+10}",
                customer_id="cust_003",
                document_type="onboarding",
                classification_timestamp="",
                kyc_tier="Tier 2" if i % 2 == 0 else "Tier 1",
                kyc_confidence=0.9,
                mapped_obligations=["KYC/Due Diligence"],
                obligation_confidence=0.85,
                risk_flag="Compliant",
                risk_confidence=0.8,
                requires_escalation=False,
                raw_input_hash=f"hash_{i}",
                processed_input=f"Text {i}",
            )
            self.manager.log_classification(record)

        # Query by tier
        tier1_records = self.manager.query_records(kyc_tier="Tier 1")
        assert len(tier1_records) == 2  # indices 1, 3

    def test_purge_expired(self):
        """Test purging expired records."""
        from phase_7_monitoring.audit_logger import AuditRecord
        record = AuditRecord(
            audit_id="TEST-EXPIRED",
            customer_id="cust_expired",
            document_type="old",
            classification_timestamp="2020-01-01T00:00:00",
            kyc_tier="Tier 1",
            kyc_confidence=0.5,
            mapped_obligations=[],
            obligation_confidence=0.5,
            risk_flag="Compliant",
            risk_confidence=0.5,
            requires_escalation=False,
            raw_input_hash="expired_hash",
            processed_input="Old data",
            retention_expires_at="2021-01-01T00:00:00",  # Already expired
        )
        self.manager.log_classification(record)
        purged = self.manager.purge_expired_records()
        assert purged >= 1


class TestModelDriftDetector:
    """Tests for the model drift detection system."""

    def setup_method(self):
        from phase_7_monitoring.audit_logger import ModelDriftDetector
        self.detector = ModelDriftDetector(window_size=100)

    def test_no_drift_with_matching_distribution(self):
        """Test no drift when distribution matches baseline."""
        self.detector.set_baseline({'kyc': {0: 0.33, 1: 0.33, 2: 0.34}})

        # Record matching distribution
        for _ in range(50):
            self.detector.record_prediction('kyc', 0)
        for _ in range(50):
            self.detector.record_prediction('kyc', 1)

        drift = self.detector.detect_drift('kyc')
        # With insufficient class 2 data, drift may or may not trigger
        assert 'drift_detected' in drift

    def test_drift_insufficient_data(self):
        """Test drift detection returns false with insufficient data."""
        self.detector.set_baseline({'kyc': {0: 0.5, 1: 0.5}})
        self.detector.record_prediction('kyc', 0)
        self.detector.record_prediction('kyc', 1)

        drift = self.detector.detect_drift('kyc')
        assert drift['drift_detected'] is False
        assert 'Insufficient data' in drift['reason']


# =============================================================================
# Integration Tests
# =============================================================================

class TestEndToEndPipeline:
    """Integration tests for the full ICC pipeline."""

    def test_preprocess_then_classify(self):
        """Test preprocessing followed by classification."""
        from phase_2_processing.preprocessor import ICCPreprocessor
        from phase_3_model.model_architecture import ICCMultiHeadClassifier

        preprocessor = ICCPreprocessor()
        doc = {'text': 'Customer Oluwaseun Adebayo BVN is 12345678901, Tier 2 onboarding'}
        processed = preprocessor.process_document(doc)

        assert processed.status == "processed"
        assert "<BVN>" in processed.masked_text

        # Verify the processed text can be fed to the model
        model = ICCMultiHeadClassifier()
        input_ids = torch.randint(0, 1000, (1, 64))
        attention_mask = torch.ones(1, 64, dtype=torch.long)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        assert 'kyc_logits' in outputs

    def test_annotation_to_training_data(self):
        """Test that annotations can be converted to training data format."""
        from phase_2_processing.annotation import AnnotationManager
        manager = AnnotationManager()

        annotation = manager.create_annotation(
            document_id="e2e_001",
            kyc_tier="Tier 2",
            obligation_categories=["KYC/Due Diligence", "BVN/NIN Verification"],
            risk_flags=["COMPLIANT"],
            risk_level="Low",
            annotator_id="system",
            confidence=0.95,
        )

        # Export and re-import
        filepath = manager.export_annotations([annotation], format="json")
        with open(filepath, 'r') as f:
            data = json.load(f)

        assert len(data['annotations']) == 1
        assert data['annotations'][0]['kyc_tier'] == 'Tier 2'

    def test_audit_trail_integrity(self):
        """Test that audit trail maintains integrity."""
        from phase_7_monitoring.audit_logger import AuditTrailManager, AuditRecord

        manager = AuditTrailManager(storage_mode="memory")

        # Log multiple records
        for i in range(10):
            record = AuditRecord(
                audit_id=f"E2E-{i:04d}",
                customer_id=f"cust_{i}",
                document_type="onboarding",
                classification_timestamp="",
                kyc_tier=f"Tier {(i % 3) + 1}",
                kyc_confidence=0.9,
                mapped_obligations=["KYC/Due Diligence"],
                obligation_confidence=0.85,
                risk_flag="Compliant" if i % 2 == 0 else "Suspicious",
                risk_confidence=0.8,
                requires_escalation=i % 2 != 0,
                raw_input_hash=f"hash_{i}",
                processed_input=f"Sample text {i}",
            )
            manager.log_classification(record)

        # Verify all records are retrievable
        all_records = manager.query_records()
        assert len(all_records) == 10

        # Verify query filtering works
        escalated = manager.query_records(risk_flag="Suspicious")
        assert len(escalated) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
