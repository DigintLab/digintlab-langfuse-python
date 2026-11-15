"""Trace one call and attach scores.

Run with LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL set.
"""

import asyncio
import os
import sys

from digintlab_langfuse import LangfuseClientConfig, init_langfuse_telemetry
from digintlab_langfuse.scores import CategoricalScore, NumericScore, ScoreTarget, score_elapsed


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


async def run(config: LangfuseClientConfig) -> None:
    """Open a root observation, score the active trace, then emit scores against an explicit trace id."""
    telemetry = init_langfuse_telemetry(config)
    async with telemetry:
        with telemetry.trace_call(name="example-run", session_id="example-session", user_id="example-user", root_observation=True):
            telemetry.score_current_trace(score=NumericScore(name="quality", value=0.95))
            runtime_context = telemetry.trace_runtime_context()
            if runtime_context is not None:
                target = ScoreTarget(trace_id=runtime_context.trace_id)
                with score_elapsed(telemetry.client, name="work_seconds", target=target):
                    await asyncio.sleep(0.01)
                telemetry.emit_score(score=CategoricalScore(name="outcome", value="ok"), target=target)
        await telemetry.flush()


if __name__ == "__main__":
    asyncio.run(run(config_from_env()))
