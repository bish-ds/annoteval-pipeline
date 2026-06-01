"""Handle AWS S3 operations for pipeline outputs.

This module manages uploading, downloading, listing, and connection checks
for the pipeline's S3 storage layer.
"""

from __future__ import annotations

import os
from pathlib import Path

import boto3
import dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PIPELINE_OUTPUT_FILES = [
    DATA_DIR / "validated_annotations.csv",
    DATA_DIR / "validation_errors.csv",
    DATA_DIR / "annotator_stats.csv",
    DATA_DIR / "cohen_kappa_scores.csv",
    DATA_DIR / "agreement_summary.json",
    DATA_DIR / "percent_agreement.csv",
    DATA_DIR / "contested_questions.csv",
    DATA_DIR / "llm_eval_scores.csv",
]



def load_aws_config() -> dict[str, str]:
    """Load AWS credentials and bucket configuration from the .env file."""

    dotenv.load_dotenv(PROJECT_ROOT / ".env")
    return {
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID", "").strip(),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", "").strip(),
        "aws_bucket_name": os.getenv("AWS_BUCKET_NAME", "").strip(),
        "aws_region": os.getenv("AWS_REGION", "").strip(),
    }



def initialize_s3_client() -> tuple[object | None, str]:
    """Initialize and return a boto3 S3 client plus the configured bucket name."""

    config = load_aws_config()
    bucket_name = config["aws_bucket_name"]
    required_values = [
        config["aws_access_key_id"],
        config["aws_secret_access_key"],
        bucket_name,
        config["aws_region"],
    ]
    if any(not value for value in required_values):
        print("AWS configuration is incomplete. Check the .env file values.")
        return None, bucket_name

    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=config["aws_access_key_id"],
            aws_secret_access_key=config["aws_secret_access_key"],
            region_name=config["aws_region"],
        )
        return s3_client, bucket_name
    except Exception as exc:
        print(f"Failed to initialize S3 client: {exc}")
        return None, bucket_name



def upload_file(local_path: str | Path, s3_key: str) -> bool:
    """Upload a local file to S3 and report whether the upload succeeded."""

    s3_client, bucket_name = initialize_s3_client()
    local_path = Path(local_path)
    if s3_client is None:
        return False
    if not local_path.exists():
        print(f"Upload failed: local file not found: {local_path}")
        return False

    try:
        s3_client.upload_file(str(local_path), bucket_name, s3_key)
        print(f"Uploaded {local_path} → s3://{bucket_name}/{s3_key}")
        return True
    except Exception as exc:
        print(f"Upload failed for {local_path}: {exc}")
        return False



def download_file(s3_key: str, local_path: str | Path) -> bool:
    """Download a file from S3 to a local path and report the result."""

    s3_client, bucket_name = initialize_s3_client()
    local_path = Path(local_path)
    if s3_client is None:
        return False

    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket_name, s3_key, str(local_path))
        print(f"Downloaded s3://{bucket_name}/{s3_key} → {local_path}")
        return True
    except Exception as exc:
        print(f"Download failed for s3://{bucket_name}/{s3_key}: {exc}")
        return False



def upload_pipeline_outputs() -> dict[str, str]:
    """Upload all available pipeline output files to the pipeline_outputs/ S3 folder."""

    results: dict[str, str] = {}
    for local_path in PIPELINE_OUTPUT_FILES:
        filename = local_path.name
        if not local_path.exists():
            print(f"Skipped {local_path}: file does not exist")
            results[filename] = "failed"
            continue

        s3_key = f"pipeline_outputs/{filename}"
        success = upload_file(local_path, s3_key)
        status = "success" if success else "failed"
        print(f"{filename}: {status}")
        results[filename] = status
    return results



def list_pipeline_outputs() -> list[dict[str, object]]:
    """List all files under pipeline_outputs/ in S3 and print a formatted table."""

    s3_client, bucket_name = initialize_s3_client()
    if s3_client is None:
        return []

    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix="pipeline_outputs/")
        contents = response.get("Contents", [])
        results = [
            {
                "key": item["Key"],
                "size_kb": round(item["Size"] / 1024, 2),
                "last_modified": item["LastModified"].isoformat(),
            }
            for item in contents
            if item["Key"] != "pipeline_outputs/"
        ]

        if not results:
            print("No pipeline outputs found in S3.")
            return []

        key_width = max(len("key"), max(len(str(item["key"])) for item in results))
        size_width = max(len("size_kb"), max(len(f"{item['size_kb']:.2f}") for item in results))
        last_modified_width = max(
            len("last_modified"),
            max(len(str(item["last_modified"])) for item in results),
        )
        header = (
            f"{'key'.ljust(key_width)}  "
            f"{'size_kb'.ljust(size_width)}  "
            f"{'last_modified'.ljust(last_modified_width)}"
        )
        print(header)
        print("-" * len(header))
        for item in results:
            print(
                f"{str(item['key']).ljust(key_width)}  "
                f"{item['size_kb']:.2f}".ljust(size_width + 2)
                + f"  {str(item['last_modified']).ljust(last_modified_width)}"
            )
        return results
    except Exception as exc:
        print(f"Failed to list pipeline outputs: {exc}")
        return []



def check_bucket_connection() -> bool:
    """Verify that the configured AWS bucket can be listed with the current credentials."""

    s3_client, bucket_name = initialize_s3_client()
    if s3_client is None:
        return False

    try:
        s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
        print("AWS connection successful")
        return True
    except Exception as exc:
        print(f"AWS connection failed: {exc}")
        return False



def main() -> None:
    """Check AWS connectivity, upload available outputs, and list remote results."""

    if check_bucket_connection():
        upload_pipeline_outputs()
        list_pipeline_outputs()


if __name__ == "__main__":
    main()
