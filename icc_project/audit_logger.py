"""
ICC Audit Trail & Monitoring System
=====================================
Implements 5-year audit trail compliance, model drift detection,
and comprehensive logging for the ICC system.

Author: Team Ogun — ICC Product
Date: August 2026
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict, field
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Audit Trail Database Schema (SQL)
# =============================================================================

AUDIT_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS icc_audit_trail (
    id SERIAL PRIMARY KEY,
    audit_id VARCHAR(50) UNIQUE NOT NULL,
    customer_id VARCHAR(100),
    document_type VARCHAR(50),
    classification_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    kyc_tier VARCHAR(20) NOT NULL,
    kyc_confidence FLOAT,
    mapped_obligations JSONB,
    obligation_confidence FLOAT,
    risk_flag VARCHAR(50),
    risk_confidence FLOAT,
    requires_escalation BOOLEAN DEFAULT FALSE,
    raw_input_hash VARCHAR(64),
    processed_input TEXT,
    model_version VARCHAR(50),
    processing_time_ms FLOAT,
    retention_expires_at TIMESTAMP,
    reviewed_by VARCHAR(100),
    review_timestamp TIMESTAMP,
    review_decision VARCHAR(50),
    review_notes TEXT,
    checksum VARCHAR(64)
);

CREATE INDEX IF NOT EXISTS idx_audit_customer ON icc_audit_trail(customer_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON icc_audit_trail(classification_timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_kyc_tier ON icc_audit_trail(kyc_tier);
CREATE INDEX IF NOT EXISTS idx_audit_risk_flag ON icc_audit_trail(risk_flag);
CREATE INDEX IF NOT EXISTS idx_audit_retention ON icc_audit_trail(retention_expires_at);

-- Immutable log table for regulatory compliance
CREATE TABLE IF NOT EXISTS icc_audit_log (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    event_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    actor VARCHAR(100),
    target_id VARCHAR(100),
    action VARCHAR(200),
    details JSONB,
    ip_address VARCHAR(45),
    user_agent TEXT,
    checksum VARCHAR(64)
);

-- Model performance tracking
CREATE TABLE IF NOT EXISTS icc_model_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50),
    metric_name VARCHAR(100),
    metric_value FLOAT,
    dataset_slice VARCHAR(100),
    data_drift_score FLOAT
);
"""


class EventSeverity(str, Enum):
    """Severity levels for audit events."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AuditEventType(str, Enum):
    """Types of auditable events."""
    CLASSIFICATION = "CLASSIFICATION"
    ESCALATION = "ESCALATION"
    REVIEW = "REVIEW"
    OVERRIDE = "OVERRIDE"
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    MODEL_UPDATE = "MODEL_UPDATE"
    DATA_EXPORT = "DATA_EXPORT"
    ACCESS_LOG = "ACCESS_LOG"


@dataclass
class AuditRecord:
    """A single audit trail record."""
    audit_id: str
    customer_id: str
    document_type: str
    classification_timestamp: str
    kyc_tier: str
    kyc_confidence: float
    mapped_obligations: List[str]
    obligation_confidence: float
    risk_flag: str
    risk_confidence: float
    requires_escalation: bool
    raw_input_hash: str
    processed_input: str
    model_version: str = "1.0.0"
    processing_time_ms: float = 0.0
    retention_expires_at: str = ""
    reviewed_by: Optional[str] = None
    review_timestamp: Optional[str] = None
    review_decision: Optional[str] = None
    review_notes: Optional[str] = None

    def __post_init__(self):
        if not self.retention_expires_at:
            expiry = datetime.utcnow() + timedelta(days=5 * 365)
            self.retention_expires_at = expiry.isoformat()
        if not self.classification_timestamp:
            self.classification_timestamp = datetime.utcnow().isoformat()


@dataclass
class AuditEvent:
    """A system audit event (not a classification record)."""
    event_type: str
    actor: str
    target_id: str
    action: str
    details: Dict = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class AuditTrailManager:
    """
    Manages the immutable audit trail for ICC classification decisions.
    Supports both in-memory (development) and PostgreSQL (production) modes.
    """

    RETENTION_PERIOD_DAYS = 5 * 365  # 5 years

    def __init__(
        self,
        storage_mode: str = "memory",
        db_config: Optional[Dict] = None,
        storage_dir: str = "./data/audit_logs",
    ):
        """
        Initialize the audit trail manager.

        Args:
            storage_mode: 'memory', 'file', or 'postgres'.
            db_config: Database configuration for PostgreSQL mode.
            storage_dir: Directory for file-based storage.
        """
        self.storage_mode = storage_mode
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.records: Dict[str, AuditRecord] = {}
        self.events: List[AuditEvent] = []

        if storage_mode == "postgres":
            self._init_postgres(db_config)
        elif storage_mode == "file":
            logger.info(f"File-based audit storage: {self.storage_dir}")

    def _init_postgres(self, db_config: dict):
        """Initialize PostgreSQL connection and create tables."""
        try:
            import psycopg2
            self.conn = psycopg2.connect(**db_config)
            with self.conn.cursor() as cur:
                cur.execute(AUDIT_TABLE_SCHEMA)
                self.conn.commit()
            logger.info("PostgreSQL audit tables created")
        except ImportError:
            logger.error("psycopg2 not installed. Install with: pip install psycopg2-binary")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise

    def _compute_checksum(self, record: AuditRecord) -> str:
        """Compute SHA-256 checksum for integrity verification."""
        data = json.dumps({
            'audit_id': record.audit_id,
            'kyc_tier': record.kyc_tier,
            'risk_flag': record.risk_flag,
            'timestamp': record.classification_timestamp,
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def log_classification(self, record: AuditRecord) -> bool:
        """
        Log a classification decision to the audit trail.

        Args:
            record: AuditRecord with classification details.

        Returns:
            True if successfully logged.
        """
        if self.storage_mode == "memory":
            self.records[record.audit_id] = record
            return True

        elif self.storage_mode == "file":
            filepath = self.storage_dir / f"{record.audit_id}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(asdict(record), f, indent=2)
            return True

        elif self.storage_mode == "postgres":
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO icc_audit_trail (
                            audit_id, customer_id, document_type,
                            kyc_tier, kyc_confidence, mapped_obligations,
                            obligation_confidence, risk_flag, risk_confidence,
                            requires_escalation, raw_input_hash, processed_input,
                            model_version, processing_time_ms, retention_expires_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        record.audit_id,
                        record.customer_id,
                        record.document_type,
                        record.kyc_tier,
                        record.kyc_confidence,
                        json.dumps(record.mapped_obligations),
                        record.obligation_confidence,
                        record.risk_flag,
                        record.risk_confidence,
                        record.requires_escalation,
                        record.raw_input_hash,
                        record.processed_input,
                        record.model_version,
                        record.processing_time_ms,
                        record.retention_expires_at,
                    ))
                    self.conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to log to PostgreSQL: {e}")
                self.conn.rollback()
                return False

        return False

    def log_event(self, event: AuditEvent) -> bool:
        """Log a system event to the audit event log."""
        self.events.append(event)

        if self.storage_mode == "postgres":
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO icc_audit_log (
                            event_type, actor, target_id, action,
                            details, ip_address, user_agent
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        event.event_type,
                        event.actor,
                        event.target_id,
                        event.action,
                        json.dumps(event.details),
                        event.ip_address,
                        event.user_agent,
                    ))
                    self.conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to log event: {e}")
                self.conn.rollback()
                return False

        return True

    def get_record(self, audit_id: str) -> Optional[AuditRecord]:
        """Retrieve a specific audit record."""
        if self.storage_mode == "memory":
            return self.records.get(audit_id)
        elif self.storage_mode == "file":
            filepath = self.storage_dir / f"{audit_id}.json"
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                return AuditRecord(**data)
        return None

    def query_records(
        self,
        customer_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        kyc_tier: Optional[str] = None,
        risk_flag: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditRecord]:
        """Query audit records with filters."""
        # Simplified in-memory query
        results = list(self.records.values())

        if customer_id:
            results = [r for r in results if r.customer_id == customer_id]
        if kyc_tier:
            results = [r for r in results if r.kyc_tier == kyc_tier]
        if risk_flag:
            results = [r for r in results if r.risk_flag == risk_flag]

        return results[:limit]

    def purge_expired_records(self) -> int:
        """Remove records past their retention period."""
        purged = 0
        if self.storage_mode == "memory":
            now = datetime.utcnow()
            expired_ids = [
                aid for aid, record in self.records.items()
                if datetime.fromisoformat(record.retention_expires_at) < now
            ]
            for aid in expired_ids:
                del self.records[aid]
                purged += 1
        return purged


class ModelDriftDetector:
    """
    Monitors model predictions for drift over time.
    Detects when the model's prediction distribution diverges
    from its training distribution.
    """

    def __init__(self, window_size: int = 1000, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.recent_predictions: Dict[str, list] = {
            'kyc': [],
            'obligation': [],
            'risk': [],
        }
        self.baseline_distributions: Dict[str, list] = {}

    def set_baseline(self, baseline: Dict[str, list]):
        """Set the baseline distribution from training data."""
        self.baseline_distributions = baseline

    def record_prediction(self, task: str, prediction: int):
        """Record a prediction for drift monitoring."""
        self.recent_predictions[task].append(prediction)
        # Keep only recent window
        if len(self.recent_predictions[task]) > self.window_size:
            self.recent_predictions[task] = self.recent_predictions[task][-self.window_size:]

    def detect_drift(self, task: str) -> Dict:
        """
        Detect distributional drift for a specific task.

        Uses a simple chi-squared test approximation.
        """
        if task not in self.baseline_distributions:
            return {'drift_detected': False, 'reason': 'No baseline set'}

        recent = self.recent_predictions[task]
        if len(recent) < 100:
            return {'drift_detected': False, 'reason': 'Insufficient data'}

        baseline = self.baseline_distributions[task]

        # Compute current distribution
        from collections import Counter
        current_counts = Counter(recent)
        total = len(recent)
        current_dist = {k: v / total for k, v in current_counts.items()}

        # Compare distributions
        drift_score = 0.0
        for class_id in set(list(current_dist.keys()) + list(baseline.keys())):
            baseline_prob = baseline.get(class_id, 0.0)
            current_prob = current_dist.get(class_id, 0.0)
            drift_score += abs(current_prob - baseline_prob)

        drift_detected = drift_score > self.threshold

        return {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'threshold': self.threshold,
            'sample_size': len(recent),
            'current_distribution': current_dist,
            'baseline_distribution': baseline,
        }
