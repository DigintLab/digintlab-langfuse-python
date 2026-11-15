"""Typed botocore Stubber surface used for Bedrock Converse tests."""

from types import TracebackType
from typing import Literal, Self

from botocore.client import BaseClient, ConverseRequest, ConverseResponse

class Stubber:
    def __init__(self, client: BaseClient) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def add_response(
        self,
        method: Literal["converse"],
        service_response: ConverseResponse,
        expected_params: ConverseRequest,
    ) -> None: ...
