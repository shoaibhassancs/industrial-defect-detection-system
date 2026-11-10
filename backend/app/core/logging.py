"""JSON logging with structlog (PLAN.md Phase 5 step 2).

One JSON object per line on standard output, with the request ID bound for the whole request.
Never logged: image bytes, uploaded file names, or the full text of anything the client sent.
Stack traces are logged here and never returned to the client.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    numeric = logging.getLevelNamesMapping()[level.upper()]
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=numeric, force=True)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str = "app") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
