# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_URL=http://api:8000

WORKDIR /app

# Install evaluator dependencies
COPY evaluator/requirements.txt /app/evaluator/requirements.txt
RUN pip install --no-cache-dir -r /app/evaluator/requirements.txt

# Copy evaluator application code
COPY evaluator/ /app/evaluator/

# Ensure directories exist
RUN mkdir -p /app/models

# Execute evaluation suite
CMD ["python", "evaluator/evaluate.py"]
