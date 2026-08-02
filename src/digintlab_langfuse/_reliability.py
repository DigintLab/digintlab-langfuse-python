"""Internal fail-open context handling for runtime telemetry operations."""

import logging
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Never


def _raise_application_error(error: BaseException) -> Never:
    raise error.with_traceback(error.__traceback__)


@contextmanager
def fail_open_context[TValue](
    start: Callable[[], AbstractContextManager[TValue]],
    *,
    logger: logging.Logger,
    operation: str,
) -> Iterator[TValue | None]:
    """Run a telemetry context without allowing its failures to replace application behavior."""
    application_error: BaseException | None = None
    body_started = False
    try:
        with start() as value:
            body_started = True
            try:
                yield value
            except BaseException as exc:
                application_error = exc
                raise
        if application_error is not None:
            _raise_application_error(application_error)
    except BaseException as exc:
        if application_error is not None:
            if exc is application_error:
                raise
            if not isinstance(exc, Exception):
                raise
            logger.exception("%s failed while handling an application error", operation)
            _raise_application_error(application_error)
        if not isinstance(exc, Exception):
            raise
        logger.exception("%s failed", operation)
        if not body_started:
            yield None
