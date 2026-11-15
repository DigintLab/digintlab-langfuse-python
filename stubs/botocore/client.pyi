"""Bedrock Runtime operations exercised by the local integration tests."""

from typing import Literal, TypedDict, Unpack

class TextBlock(TypedDict):
    text: str

class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: list[TextBlock]

class ConverseRequest(TypedDict):
    modelId: str
    messages: list[Message]

class ConverseUsage(TypedDict):
    inputTokens: int
    outputTokens: int
    totalTokens: int
    cacheReadInputTokens: int
    cacheWriteInputTokens: int

class ConverseOutput(TypedDict):
    message: Message

class ConverseMetrics(TypedDict):
    latencyMs: int

class ConverseResponse(TypedDict):
    output: ConverseOutput
    stopReason: str
    usage: ConverseUsage
    metrics: ConverseMetrics

class BaseClient: ...

class BedrockRuntimeClient(BaseClient):
    def converse(self, **kwargs: Unpack[ConverseRequest]) -> ConverseResponse: ...
