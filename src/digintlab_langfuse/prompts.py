"""Prompt management helpers for the Langfuse SDK."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol

from langfuse import Langfuse
from langfuse.model import ChatMessageDict, ChatMessageWithPlaceholdersDict, ChatPromptClient, TextPromptClient

from digintlab_langfuse.json_values import JsonValue, to_json_value

__all__ = [
    "ChatManagedPrompt",
    "ChatPromptMessage",
    "ChatPromptPart",
    "ChatPromptRequest",
    "ManagedPrompt",
    "PromptClient",
    "PromptKind",
    "PromptRequest",
    "TextManagedPrompt",
    "TextPromptRequest",
    "create_or_update_prompt",
    "get_prompt",
    "langchain_prompt_metadata",
]

type ChatPromptMessage = ChatMessageDict
type ChatPromptPart = ChatMessageDict | ChatMessageWithPlaceholdersDict
type PromptClient = TextPromptClient | ChatPromptClient


class PromptKind(StrEnum):
    """Prompt types supported by Langfuse prompt management."""

    CHAT = "chat"
    TEXT = "text"


class PromptRequest[TClient: PromptClient](Protocol):
    """Typed interface for fetching a managed Langfuse prompt."""

    prompt_kind: PromptKind

    def fetch(self, lf: Langfuse) -> TClient:
        """Fetch this prompt through the Langfuse SDK cache."""
        ...


class ManagedPrompt[TClient: PromptClient](Protocol):
    """Typed interface for creating a managed Langfuse prompt version."""

    prompt_kind: PromptKind

    def create(self, lf: Langfuse) -> TClient:
        """Create this prompt version in Langfuse."""
        ...


@dataclass(frozen=True, slots=True)
class TextPromptRequest(PromptRequest[TextPromptClient]):
    """Request parameters for fetching a managed text prompt."""

    name: str
    prompt_kind: PromptKind = field(default=PromptKind.TEXT, init=False)
    version: int | None = None
    label: str | None = None
    cache_ttl_seconds: int | None = None
    fallback: str | None = None
    max_retries: int | None = None
    fetch_timeout_seconds: int | None = None

    @property
    def sdk_type(self) -> Literal["text"]:
        """Return the Langfuse SDK prompt type literal."""
        return "text"

    def fetch(self, lf: Langfuse) -> TextPromptClient:
        """Fetch this text prompt through the Langfuse SDK cache."""
        return lf.get_prompt(
            self.name,
            type=self.sdk_type,
            version=self.version,
            label=self.label,
            cache_ttl_seconds=self.cache_ttl_seconds,
            fallback=self.fallback,
            max_retries=self.max_retries,
            fetch_timeout_seconds=self.fetch_timeout_seconds,
        )


@dataclass(frozen=True, slots=True)
class ChatPromptRequest(PromptRequest[ChatPromptClient]):
    """Request parameters for fetching a managed chat prompt."""

    name: str
    prompt_kind: PromptKind = field(default=PromptKind.CHAT, init=False)
    version: int | None = None
    label: str | None = None
    cache_ttl_seconds: int | None = None
    fallback: list[ChatPromptMessage] | None = None
    max_retries: int | None = None
    fetch_timeout_seconds: int | None = None

    @property
    def sdk_type(self) -> Literal["chat"]:
        """Return the Langfuse SDK prompt type literal."""
        return "chat"

    def fetch(self, lf: Langfuse) -> ChatPromptClient:
        """Fetch this chat prompt through the Langfuse SDK cache."""
        return lf.get_prompt(
            self.name,
            type=self.sdk_type,
            version=self.version,
            label=self.label,
            cache_ttl_seconds=self.cache_ttl_seconds,
            fallback=self.fallback,
            max_retries=self.max_retries,
            fetch_timeout_seconds=self.fetch_timeout_seconds,
        )


@dataclass(frozen=True, slots=True)
class TextManagedPrompt(ManagedPrompt[TextPromptClient]):
    """Text prompt version to create in Langfuse."""

    name: str
    prompt: str
    prompt_kind: PromptKind = field(default=PromptKind.TEXT, init=False)
    labels: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    config: JsonValue = None
    commit_message: str | None = None

    @property
    def sdk_type(self) -> Literal["text"]:
        """Return the Langfuse SDK prompt type literal."""
        return "text"

    def create(self, lf: Langfuse) -> TextPromptClient:
        """Create this text prompt version in Langfuse."""
        return lf.create_prompt(
            name=self.name,
            type=self.sdk_type,
            prompt=self.prompt,
            labels=list(self.labels),
            tags=list(self.tags),
            config=to_json_value(self.config),
            commit_message=self.commit_message,
        )


@dataclass(frozen=True, slots=True)
class ChatManagedPrompt(ManagedPrompt[ChatPromptClient]):
    """Chat prompt version to create in Langfuse."""

    name: str
    prompt: list[ChatPromptPart]
    prompt_kind: PromptKind = field(default=PromptKind.CHAT, init=False)
    labels: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    config: JsonValue = None
    commit_message: str | None = None

    @property
    def sdk_type(self) -> Literal["chat"]:
        """Return the Langfuse SDK prompt type literal."""
        return "chat"

    def create(self, lf: Langfuse) -> ChatPromptClient:
        """Create this chat prompt version in Langfuse."""
        return lf.create_prompt(
            name=self.name,
            type=self.sdk_type,
            prompt=self.prompt,
            labels=list(self.labels),
            tags=list(self.tags),
            config=to_json_value(self.config),
            commit_message=self.commit_message,
        )


def get_prompt[TClient: PromptClient](lf: Langfuse, request: PromptRequest[TClient]) -> TClient:
    """Fetch a managed prompt through the Langfuse SDK cache."""
    return request.fetch(lf)


def create_or_update_prompt[TClient: PromptClient](lf: Langfuse, prompt: ManagedPrompt[TClient]) -> TClient:
    """Create a new managed prompt version in Langfuse."""
    return prompt.create(lf)


def langchain_prompt_metadata(prompt: PromptClient) -> dict[str, PromptClient]:
    """Return metadata used by LangChain integrations to link generations to a prompt."""
    return {"langfuse_prompt": prompt}
