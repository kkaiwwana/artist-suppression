"""Scoped suppression of noisy third-party model output."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
import io
import logging
from typing import Iterator
import warnings


@contextmanager
def quiet_external_output(
    enabled: bool = True,
    *,
    capture_stdout: bool = True,
    capture_stderr: bool = True,
    suppress_warnings: bool = True,
) -> Iterator[None]:
    """Temporarily hide dependency INFO logs, prints, progress, and warnings.

    Exceptions are never swallowed. The process-wide logging disable threshold
    is restored exactly, including when model construction or inference fails.
    This context is intentionally scoped to heavyweight third-party calls; it
    must not wrap the surrounding training/evaluation control flow.
    """

    if not enabled:
        yield
        return

    previous_disable = logging.root.manager.disable
    logging.disable(max(previous_disable, logging.INFO))
    try:
        with ExitStack() as stack:
            if capture_stdout:
                stack.enter_context(redirect_stdout(io.StringIO()))
            if capture_stderr:
                stack.enter_context(redirect_stderr(io.StringIO()))
            if suppress_warnings:
                stack.enter_context(warnings.catch_warnings())
                warnings.simplefilter("ignore")
            yield
    finally:
        logging.disable(previous_disable)


__all__ = ["quiet_external_output"]
