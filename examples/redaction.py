"""Mask credentials and PII before data leaves the process.

Requires the `redaction` extra. Run with LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL set.
"""

import asyncio
import os
import sys

from digintlab_langfuse import LangfuseClientConfig, SensitiveDataRedactionConfig, init_langfuse_telemetry
from digintlab_langfuse.json_values import JsonValue
from digintlab_langfuse.redaction import redact_sensitive_data


def redaction_policy() -> SensitiveDataRedactionConfig:
    """Return the example policy: default credential fields plus email and phone PII."""
    return SensitiveDataRedactionConfig(pii_entities=frozenset({"EMAIL_ADDRESS", "PHONE_NUMBER"}))


def config_from_env() -> LangfuseClientConfig:
    """Build a client config with redaction enabled, exiting with a clear message when the environment is incomplete."""
    names = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        sys.exit(f"Set these environment variables before running this example: {', '.join(missing)}")
    return LangfuseClientConfig(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        base_url=os.environ["LANGFUSE_BASE_URL"],
        sensitive_data_redaction=redaction_policy(),
    )


async def run(config: LangfuseClientConfig) -> JsonValue:
    """Trace a call whose input contains sensitive values, and return what the policy does to the same payload."""
    payload: dict[str, JsonValue] = {"api_key": "sk-secret", "note": "Contact jane.doe@example.com"}
    telemetry = init_langfuse_telemetry(config)
    async with telemetry:
        with telemetry.trace_call(
            name="example-redaction",
            session_id="example-session",
            user_id="example-user",
            root_observation=True,
            root_input=payload,
        ):
            pass
        await telemetry.flush()
    return redact_sensitive_data(payload, redaction_policy())


if __name__ == "__main__":
    sys.stdout.write(f"{asyncio.run(run(config_from_env()))}\n")
