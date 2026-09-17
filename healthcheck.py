import os
import sys

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


def check_mlflow(tracking_uri: str) -> None:
    client = MlflowClient(tracking_uri=tracking_uri)
    experiments = client.search_experiments(max_results=1)
    print(f"MLflow reachable at {tracking_uri} ({len(experiments)} experiment(s) visible)")


def check_aws_identity() -> None:
    identity = boto3.client("sts").get_caller_identity()
    print(f"AWS identity OK: account={identity['Account']} arn={identity['Arn']}")


def main() -> int:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        print("ERROR: MLFLOW_TRACKING_URI is not set.", file=sys.stderr)
        return 1

    try:
        check_mlflow(tracking_uri)
        check_aws_identity()
    except (MlflowException, ClientError, BotoCoreError) as exc:
        print(f"ERROR: healthcheck failed: {exc}", file=sys.stderr)
        return 1

    print("All healthchecks passed.")
    return 0


main()
