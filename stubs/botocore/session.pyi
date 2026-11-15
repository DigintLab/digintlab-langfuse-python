"""Types for creating the Bedrock Runtime client used in tests."""

from typing import Literal
from botocore.client import BedrockRuntimeClient

class Session:
    def create_client(
        self,
        service_name: Literal["bedrock-runtime"],
        *,
        region_name: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
    ) -> BedrockRuntimeClient: ...

def get_session() -> Session: ...
