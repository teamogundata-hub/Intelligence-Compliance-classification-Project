# ICC System Dockerfile
# Multi-stage build for production deployment
# ========================

FROM python:3.11-slim as base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "phase_6_inference.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]


# =============================================================================
# Development Stage
# =============================================================================

FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-cov pytest-asyncio httpx mypy

# Run tests on build
CMD ["pytest", "phase_8_testing/", "-v", "--cov=."]
