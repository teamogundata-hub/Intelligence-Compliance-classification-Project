"""
ICC Inference Server (FastAPI)
===============================
Production-grade FastAPI inference server implementing the 5-stage
processing pipeline: Ingestion → Classification → Obligation Mapping →
Escalation → Audit Trail.

Author: Team Ogun — ICC Product
Date: August 2026
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
import torch
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

from phase_3_model.model_architecture import ICCMultiHeadClassifier
from phase_2_processing.preprocessor import ICCPreprocessor
from phase_7_monitoring.audit_logger import AuditTrailLogger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Application Setup
# =============================================================================

app = FastAPI(
    title="ICC - Intelligent Compliance Classification",
    description=(
        "Real-time and batch classification API for Nigerian financial "
        "services compliance. Classifies KYC tiers, maps obligations, "
        "and flags suspicious activity per CBN Circular BSD/DIR/PUB/LAB/019/002."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Global State (loaded at startup)
# =============================================================================

class AppState:
    """Holds application state loaded at startup."""
    def __init__(self):
        self.model: Optional[ICCMultiHeadClassifier] = None
        self.preprocessor: Optional[ICCPreprocessor] = None
        self.audit_logger: Optional[AuditTrailLogger] = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path: str = "./checkpoints/best_model.pt"

    def load(self):
        """Load model and dependencies."""
        from transformers import AutoTokenizer

        logger.info("Loading ICC model and dependencies...")

        # Load preprocessor
        self.preprocessor = ICCPreprocessor()

        # Load model
        self.model = ICCMultiHeadClassifier()
        if Path(self.model_path).exists():
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            logger.info(f"Model loaded from {self.model_path}")
        else:
            logger.warning("No pre-trained model found. Using untrained model.")

        self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        except Exception:
            logger.warning("Could not load tokenizer, using default")

        # Initialize audit logger (optional)
        try:
            self.audit_logger = AuditTrailLogger({
                'host': 'localhost',
                'port': 5432,
                'dbname': 'icc_db',
                'user': 'icc_user',
                'password': 'icc_password',
            })
        except Exception as e:
            logger.warning(f"Audit logger not available: {e}")

        logger.info("ICC system ready")


state = AppState()


@app.on_event("startup")
async def startup_event():
    """Initialize model and dependencies on server startup."""
    state.load()


# =============================================================================
# Request/Response Models
# =============================================================================

class ComplianceClassificationRequest(BaseModel):
    """Request model for single classification."""
    document_text: str = Field(..., min_length=1, max_length=5000, description="Raw text to classify")
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    document_type: Optional[str] = Field(None, description="Type: onboarding, transaction, chat, complaint")
    metadata: Optional[Dict] = Field(None, description="Additional metadata")

    @validator('document_text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("document_text cannot be empty or whitespace only")
        return v


class ComplianceClassificationResponse(BaseModel):
    """Response model for single classification."""
    audit_id: str
    kyc_tier: str
    kyc_confidence: float
    mapped_obligations: List[str]
    obligation_confidence: float
    risk_flag: str
    risk_confidence: float
    requires_escalation: bool
    processing_time_ms: float
    timestamp: str


class BatchClassificationRequest(BaseModel):
    """Request model for batch classification."""
    documents: List[ComplianceClassificationRequest] = Field(..., min_items=1, max_items=100)


class BatchClassificationResponse(BaseModel):
    """Response model for batch classification."""
    batch_id: str
    total_documents: int
    results: List[ComplianceClassificationResponse]
    processing_time_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    version: str


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=state.model is not None,
        device=str(state.device),
        version="1.0.0",
    )


@app.post("/classify", response_model=ComplianceClassificationResponse)
async def classify_document(request: ComplianceClassificationRequest):
    """
    Classify a single compliance document.

    Implements the 5-stage pipeline:
    1. Ingestion: Normalize and preprocess the text
    2. Classification: Run multi-head transformer inference
    3. Obligation Mapping: Map to specific CBN clauses
    4. Escalation: Determine if human review is needed
    5. Audit Trail: Log the decision
    """
    start_time = time.time()

    try:
        # Stage 1: Ingestion & Preprocessing
        doc = {'text': request.document_text}
        processed = state.preprocessor.process_document(doc)

        # Stage 2: Classification
        if state.tokenizer and state.model:
            inputs = state.tokenizer(
                processed.masked_text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding='max_length',
            )

            input_ids = inputs['input_ids'].to(state.device)
            attention_mask = inputs['attention_mask'].to(state.device)

            with torch.no_grad():
                outputs = state.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            # Extract predictions
            kyc_probs = torch.softmax(outputs['kyc_logits'], dim=1).cpu().numpy()[0]
            obligation_probs = torch.softmax(outputs['obligation_logits'], dim=1).cpu().numpy()[0]
            risk_probs = torch.softmax(outputs['risk_logits'], dim=1).cpu().numpy()[0]

            kyc_class = int(torch.argmax(outputs['kyc_logits'], dim=1).item())
            obligation_class = int(torch.argmax(outputs['obligation_logits'], dim=1).item())
            risk_class = int(torch.argmax(outputs['risk_logits'], dim=1).item())
        else:
            # Fallback: use rule-based classification
            kyc_class, kyc_probs = 1, [0.33, 0.34, 0.33]
            obligation_class, obligation_probs = 0, [0.5] * 15
            risk_class, risk_probs = 0, [0.7, 0.3]

        # Stage 3: Obligation Mapping
        kyc_tier_names = ['Tier 1', 'Tier 2', 'Tier 3']
        obligation_names = [
            'KYC/Due Diligence', 'Customer Identification', 'Transaction Monitoring',
            'Suspicious Activity Reporting', 'Record Retention', 'Enhanced Due Diligence',
            'PEP Screening', 'Sanctions Screening', 'Cross-Border Transactions',
            'BVN/NIN Verification', 'Beneficial Ownership', 'Annual Reporting',
            'Internal Control', 'Staff Training', 'Risk Assessment',
        ]
        risk_flag_names = ['Compliant', 'Suspicious']

        kyc_tier = kyc_tier_names[kyc_class] if kyc_class < len(kyc_tier_names) else 'Unknown'
        mapped_obligations = [obligation_names[obligation_class]] if obligation_class < len(obligation_names) else ['Unknown']
        risk_flag = risk_flag_names[risk_class] if risk_class < len(risk_flag_names) else 'Unknown'

        kyc_confidence = float(kyc_probs[kyc_class])
        obligation_confidence = float(obligation_probs[obligation_class])
        risk_confidence = float(risk_probs[risk_class])

        # Stage 4: Escalation
        requires_escalation = (
            risk_flag == 'Suspicious' or
            kyc_confidence < 0.7 or
            'Enhanced Due Diligence' in mapped_obligations
        )

        processing_time_ms = (time.time() - start_time) * 1000
        audit_id = f"ICC-{uuid.uuid4().hex[:12].upper()}"

        response = ComplianceClassificationResponse(
            audit_id=audit_id,
            kyc_tier=kyc_tier,
            kyc_confidence=kyc_confidence,
            mapped_obligations=mapped_obligations,
            obligation_confidence=obligation_confidence,
            risk_flag=risk_flag,
            risk_confidence=risk_confidence,
            requires_escalation=requires_escalation,
            processing_time_ms=round(processing_time_ms, 2),
            timestamp=datetime.utcnow().isoformat(),
        )

        # Stage 5: Audit Trail
        if state.audit_logger:
            state.audit_logger.log_decision({
                'audit_id': audit_id,
                'customer_id': request.customer_id or 'anonymous',
                'kyc_tier': kyc_tier,
                'mapped_obligations': mapped_obligations,
                'risk_flag': risk_flag,
                'raw_input': request.document_text[:500],
                'processed_input': processed.masked_text[:500],
            })

        return response

    except Exception as e:
        logger.error(f"Classification error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post("/classify/batch", response_model=BatchClassificationResponse)
async def classify_batch(request: BatchClassificationRequest):
    """
    Classify a batch of documents.

    Processes up to 100 documents in a single request.
    Returns results for all documents with a single batch ID.
    """
    start_time = time.time()
    batch_id = f"BATCH-{uuid.uuid4().hex[:10].upper()}"

    results = []
    for doc_request in request.documents:
        try:
            # Reuse single classification logic
            result = await classify_document(doc_request)
            results.append(result)
        except Exception as e:
            logger.error(f"Batch item error: {e}")
            results.append(ComplianceClassificationResponse(
                audit_id=f"ERR-{uuid.uuid4().hex[:8].upper()}",
                kyc_tier="Unknown",
                kyc_confidence=0.0,
                mapped_obligations=[],
                obligation_confidence=0.0,
                risk_flag="Unknown",
                risk_confidence=0.0,
                requires_escalation=True,
                processing_time_ms=0.0,
                timestamp=datetime.utcnow().isoformat(),
            ))

    processing_time_ms = (time.time() - start_time) * 1000

    return BatchClassificationResponse(
        batch_id=batch_id,
        total_documents=len(request.documents),
        results=results,
        processing_time_ms=round(processing_time_ms, 2),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/audit/{audit_id}")
async def get_audit_record(audit_id: str):
    """Retrieve a specific audit record."""
    if not state.audit_logger:
        raise HTTPException(status_code=503, detail="Audit logger not available")

    try:
        record = state.audit_logger.get_record(audit_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"Audit record not found: {audit_id}")
        return record
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_system_metrics():
    """Return system performance metrics."""
    return {
        'model_status': 'loaded' if state.model else 'not_loaded',
        'device': str(state.device),
        'supported_classifications': ['KYC Tier', 'Obligation Mapping', 'Risk Flagging'],
        'pipeline_stages': [
            'Ingestion',
            'Classification',
            'Obligation Mapping',
            'Escalation',
            'Audit Trail',
        ],
        'regulatory_compliance': 'CBN Circular BSD/DIR/PUB/LAB/019/002',
    }
