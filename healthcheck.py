import os
import sys
import uuid

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

S3_HEALTHCHECK_PREFIX = "healthcheck/"


def check_mlflow(tracking_uri: str) -> None:
    client = MlflowClient(tracking_uri=tracking_uri)
    experiments = client.search_experiments(max_results=1)
    print(f"MLflow reachable at {tracking_uri} ({len(experiments)} experiment(s) visible)")


def check_aws_identity() -> None:
    identity = boto3.client("sts").get_caller_identity()
    print(f"AWS identity OK: account={identity['Account']} arn={identity['Arn']}")


def check_s3_round_trip(bucket: str) -> None:
    s3 = boto3.client("s3")
    key = f"{S3_HEALTHCHECK_PREFIX}{uuid.uuid4().hex}.txt"
    body = b"healthcheck"

    s3.put_object(Bucket=bucket, Key=key, Body=body)
    try:
        fetched = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        if fetched != body:
            raise RuntimeError(f"S3 read returned unexpected content for s3://{bucket}/{key}")
        print(f"S3 write/read/delete round-trip OK: s3://{bucket}/{key}")
    finally:
        s3.delete_object(Bucket=bucket, Key=key)


def main() -> int:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    bucket = os.getenv("S3_BUCKET")

    if not tracking_uri:
        print("ERROR: MLFLOW_TRACKING_URI is not set.", file=sys.stderr)
        return 1
    if not bucket:
        print("ERROR: S3_BUCKET is not set.", file=sys.stderr)
        return 1

    try:
        check_mlflow(tracking_uri)
        check_aws_identity()
        check_s3_round_trip(bucket)
    except (MlflowException, ClientError, BotoCoreError, RuntimeError) as exc:
        print(f"ERROR: healthcheck failed: {exc}", file=sys.stderr)
        return 1

    print("All healthchecks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
