# Iris Continuous Training Pipeline

Continuous training pipeline: GitHub Actions builds a Docker image, pushes it to Docker Hub, pulls it back, runs training inside the container, and logs the best model to MLflow (S3 artifact store).

## How it works

```
push to data/iris.csv  ──┐
                         ├──▶ build image ──▶ push to Docker Hub ──▶ pull image ──▶ docker run (train.py)
manual trigger ──────────┘                                                               │
                                                                                         ▼
                                                                          MLflow tracking server
                                                                          (artifacts → S3)
```

- **`train.py`** — loads `data/iris.csv`, runs `GridSearchCV` (5-fold, `f1_weighted`) over a RandomForest grid, keeps the best mean-F1 model, evaluates on a held-out test set, then logs params/metrics/model to MLflow and registers it as `iris-classifier`.
- **`Dockerfile`** — Python 3.11 slim image with all dependencies; runs `train.py` by default.
- **`.github/workflows/continuous-training.yml`** — two jobs:
  1. `build-and-push`: builds the image and pushes it to Docker Hub (`latest` + commit SHA tags).
  2. `train`: pulls the image from Docker Hub and runs the container with the MLflow/AWS environment variables injected from secrets.

## Triggers

- **Manually**: Actions tab → *Continuous Training* → *Run workflow*.
- **Automatically**: any push that modifies `data/iris.csv` (i.e., new data added).

## Required GitHub secrets

Set these in *Settings → Secrets and variables → Actions*:

| Secret | Purpose |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub account (also used as the image namespace) |
| `DOCKERHUB_TOKEN` | Docker Hub access token (Account Settings → Security) |
| `MLFLOW_TRACKING_URI` | URL of your MLflow tracking server |
| `AWS_ACCESS_KEY_ID` | AWS credentials for the S3 artifact store |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials for the S3 artifact store |
| `AWS_DEFAULT_REGION` | Region of the S3 bucket |

The MLflow tracking server must itself be configured with an S3 artifact root, e.g.:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://<your-bucket>/mlflow-artifacts \
  --host 0.0.0.0 --port 5000
```

## Run locally

```bash
docker build -t iris-trainer .
docker run --rm \
  -e MLFLOW_TRACKING_URI=... \
  -e AWS_ACCESS_KEY_ID=... \
  -e AWS_SECRET_ACCESS_KEY=... \
  -e AWS_DEFAULT_REGION=... \
  iris-trainer
```

## Adding new data

Append rows to `data/iris.csv` (columns: `sepal_length,sepal_width,petal_length,petal_width,species`) and push — the pipeline retrains automatically.
