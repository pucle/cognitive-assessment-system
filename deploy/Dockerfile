FROM python:3.10-slim

# System deps needed for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (production)
COPY backend/requirements.txt /app/requirements.txt
COPY deploy/requirements-prod.txt /app/requirements-prod.txt
RUN pip install --no-cache-dir -r /app/requirements-prod.txt

# Copy backend code
COPY backend /app/backend

# Runtime configuration
ENV HOST=0.0.0.0 \
    PORT=8000 \
    DEBUG=false \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Start with gunicorn for production
CMD ["gunicorn", "-w", "2", "-k", "gthread", "-t", "180", "-b", "0.0.0.0:8000", "backend.app:app"]


