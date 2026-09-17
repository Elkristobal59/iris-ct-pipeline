FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code and data
COPY train.py .
COPY data/ data/

# Environment variables (MLFLOW_TRACKING_URI, AWS_ACCESS_KEY_ID,
# AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION) are injected at runtime
# by `docker run -e ...` from GitHub Actions secrets.

CMD ["python", "train.py"]
