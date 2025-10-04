# Multi-stage Dockerfile for Cognitive Assessment Backend
# Production-ready Flask application with ML models

# ================================
# Stage 1: Python Base Image
# ================================
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install system dependencies for Python and ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# ================================
# Stage 2: Dependencies Installation
# ================================
FROM base AS dependencies

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ================================
# Stage 3: Model Download & Setup
# ================================
FROM dependencies AS model-setup

# Copy model bundle
COPY model_bundle/ ./model_bundle/

# Copy any additional setup scripts
COPY setup_dependencies.py ./

# Pre-download models and setup (optional - uncomment if needed)
# RUN python setup_dependencies.py

# ================================
# Stage 4: Production Image
# ================================
FROM model-setup AS production

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/recordings /app/uploads /app/storage /app/logs /tmp/cognitive_cache

# Set proper permissions
RUN chown -R appuser:appuser /app && \
    chmod -R 755 /app/recordings && \
    chmod -R 755 /app/uploads && \
    chmod -R 755 /app/storage && \
    chmod -R 755 /app/logs

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Default command (can be overridden by Railway)
CMD ["python", "app.py"]

# ================================
# Stage 5: Development Image (Optional)
# ================================
FROM dependencies AS development

# Copy application code
COPY . .

# Install development dependencies
RUN pip install --no-cache-dir pytest black flake8 mypy

# Create non-root user
RUN groupadd -r devuser && useradd -r -g devuser devuser

# Switch to non-root user
USER devuser

# Expose port
EXPOSE 5001

# Development command
CMD ["python", "app.py"]