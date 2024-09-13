import os
import logging
import json
from google.cloud import storage
import filetype
from typing import Union, Optional


class GcsUtils:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = self.initialize_gcs_client()

    def initialize_gcs_client(self) -> storage.Client:
        try:
            client = storage.Client(project=self.project_id)
            return client
        except Exception as e:
            raise Exception(f"Error initializing Google Cloud Storage client: {str(e)}")

    def save_to_bucket(
        self, data: Union[str, bytes, dict, list], filename: str, bucket_name: str
    ) -> None:
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(filename)

            if isinstance(data, str):
                content_type = "text/plain"
                data = data.encode("utf-8")
            elif isinstance(data, bytes):
                kind = filetype.guess(data)
                content_type = kind.mime if kind else "application/octet-stream"
            elif isinstance(data, (dict, list)):
                content_type = "application/json"
                data = json.dumps(data).encode("utf-8")
            else:
                raise TypeError(
                    f"Unsupported data type for saving: {type(data)}. Supported types: str, bytes, dict, list"
                )

            blob.upload_from_string(data, content_type=content_type)
        except Exception as e:
            raise Exception(f"Error saving data to bucket {bucket_name}: {str(e)}")

    def download_from_gcs_url(
        self, url: str, as_text: bool = False
    ) -> Optional[Union[str, bytes]]:
        try:
            if not url.startswith("gs://"):
                raise ValueError("Invalid GCS URL. Must start with 'gs://'")

            bucket_name, blob_name = url[5:].split("/", 1)
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            if not blob.exists():
                raise FileNotFoundError(
                    f"File {blob_name} not found in bucket {bucket_name}"
                )

            return blob.download_as_text() if as_text else blob.download_as_bytes()

        except FileNotFoundError as e:
            print(str(e))
            return None
        except Exception as e:
            raise Exception(f"Error downloading from URL {url}: {str(e)}")

    def download_from_gcs_bucket(
        self, filename: str, bucket_name: str, as_text: bool = False
    ) -> Optional[Union[str, bytes]]:
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(filename)

            if not blob.exists():
                raise FileNotFoundError(
                    f"File {filename} not found in bucket {bucket_name}"
                )

            return blob.download_as_text() if as_text else blob.download_as_bytes()

        except FileNotFoundError as e:
            print(str(e))
            return None
        except Exception as e:
            raise Exception(
                f"Error downloading {filename} from bucket {bucket_name}: {str(e)}"
            )
