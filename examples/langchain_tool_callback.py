"""Record a LangChain tool call as a Langfuse tool observation.

Run with LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL set.
"""

import asyncio
import os
import sys

from langchain_core.tools import tool

from digintlab_langfuse import LangfuseClientConfig, ObservationKind, init_langfuse_telemetry


@tool
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


def config_from_env() -> LangfuseClientConfig:
    """Build a client config from the environment, exiting with a clear message when it is incomplete."""
    names = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        sys.exit(f"Set these environment variables before running this example: {', '.join(missing)}")
    return LangfuseClientConfig(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        base_url=os.environ["LANGFUSE_BASE_URL"],
    )


async def run(config: LangfuseClientConfig) -> int:
    """Invoke a tool inside a node observation so the callback has an active parent to attach to."""
    telemetry = init_langfuse_telemetry(config)
    async with telemetry:
        with (
            telemetry.trace_call(name="example-tools", session_id="example-session", user_id="example-user", root_observation=True),
            telemetry.observe_node_span("tools-node", enabled=True, kind=ObservationKind.CHAIN),
        ):
            result: int = add.invoke({"left": 2, "right": 3}, config={"callbacks": [telemetry.tool_span_callback()]})
        await telemetry.flush()
    return result


if __name__ == "__main__":
    sys.stdout.write(f"add(2, 3) = {asyncio.run(run(config_from_env()))}\n")
