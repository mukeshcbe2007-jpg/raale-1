# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Set environment variables for reproducibility and unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY trainer/requirements.txt /app/trainer/requirements.txt
RUN pip install --no-cache-dir -r /app/trainer/requirements.txt

# Copy dataset and trainer module
COPY data/ /app/data/
COPY trainer/ /app/trainer/

# Ensure directory for model artifacts exists
RUN mkdir -p /app/models

# Execute the training pipeline
CMD ["python", "trainer/train.py"]
