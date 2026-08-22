# ICC — Intelligent Compliance Classification

**A Localised Transformer Architecture for Nigerian Financial-Services Compliance**

## Overview

ICC is a domain-adapted transformer system that classifies KYC tiers (1/2/3), maps documents and transactions to CBN regulatory obligations, and flags suspicious activity within 24 hours. It handles Nigerian identifiers (BVN, NIN), naming conventions, and bilingual text (English + Nigerian Pidgin).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest phase_8_testing/ -v

# Start API server
uvicorn phase_6_inference.main:app --host 0.0.0.0 --port 8000

# Or use Docker
docker-compose up -d
```

## Project Structure

```
icc_project/
├── phase_1_data_gathering/     # Data collection & scraping
│   ├── cbn_scraper.py          # CBN circular scraper
│   └── data_ingestion.py       # Multi-format ingestion pipeline
├── phase_2_processing/         # Text preprocessing & annotation
│   ├── preprocessor.py         # Full preprocessing pipeline
│   └── annotation.py           # Annotation schema & augmentation
├── phase_3_model/              # Model architecture
│   ├── model_architecture.py   # Multi-head transformer
│   └── dataset.py              # PyTorch dataset classes
├── phase_4_training/           # Training pipeline
│   └── trainer.py              # Training loop with early stopping
├── phase_5_evaluation/         # Evaluation & benchmarking
│   └── evaluator.py            # Metrics, confusion matrices, baselines
├── phase_6_inference/          # API server
│   └── main.py                 # FastAPI 5-stage pipeline
├── phase_7_monitoring/         # Audit trail & monitoring
│   └── audit_logger.py         # 5-year retention, drift detection
├── phase_8_testing/            # Test suite
│   └── test_icc.py             # Unit & integration tests
├── Dockerfile                  # Production container
├── docker-compose.yml          # Full stack deployment
├── init_db.sql                 # Database schema
├── prometheus.yml              # Monitoring config
└── requirements.txt            # All dependencies
```

## Regulatory Compliance

- CBN Circular BSD/DIR/PUB/LAB/019/002
- 24-hour suspicious activity reporting to NFIU
- 5-year record retention
- Three-tier KYC framework (Tier 1/2/3)
- BVN/NIN verification via NIBSS/NIMC

## License

Proprietary — Team Ogun / Cohort 10
