# Environment Variables

## GitHub Actions secrets

Set these in *Settings → Secrets and variables → Actions*. Used by the `healthcheck`, `build-and-push`, and `train` jobs in `.github/workflows/continuous-training.yml`.

| Secret | Purpose |
|---|---|
| `MLFLOW_TRACKING_URI` | URL of the MLflow tracking server |
| `AWS_ACCESS_KEY_ID` | AWS credentials for the S3 artifact store |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials for the S3 artifact store |
| `AWS_DEFAULT_REGION` | Region of the S3 bucket |
| `S3_BUCKET` | Bucket used as the MLflow artifact store; must match the tracking server's `--default-artifact-root`. Used by `healthcheck.py` for the write/read/delete permission check |
| `DOCKERHUB_USERNAME` | Docker Hub account (also used as the image namespace) |
| `DOCKERHUB_TOKEN` | Docker Hub access token (Account Settings → Security) |

## Optional variables (train.py)

Not required — each has a default. Override only if you need a different value.

| Variable | Default | Purpose |
|---|---|---|
| `DATA_PATH` | `data/iris.csv` | Path to the training CSV |
| `MLFLOW_EXPERIMENT_NAME` | `iris-continuous-training` | MLflow experiment name |
| `MLFLOW_MODEL_NAME` | `iris-classifier` | Name the trained model is registered under |

## Running locally

```bash
export MLFLOW_TRACKING_URI=...
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=...
export S3_BUCKET=...

python healthcheck.py
python train.py
```
