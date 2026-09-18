"""Structured logging.

One JSON object per line on stdout, which is what a container platform expects and what
a log aggregator can parse. The standard library does this with a small formatter, so no
logging dependency is added.

Nothing here logs document text, question text or key material: the fields are counts,
timings and identifiers, so logs stay safe to ship to a third-party aggregator even
though the documents are customer compliance data.
"""

import json
import logging
import sys
import time
from contextlib import contextmanager

# Rates per million tokens, gpt-4o-mini and text-embedding-3-small, as published in
# September 2026. Logged spend is an estimate for observability, not billing.
CHAT_INPUT_PER_MILLION = 0.15
CHAT_OUTPUT_PER_MILLION = 0.60

LOGGER = logging.getLogger("document_qa")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
        }
        # Anything passed as extra={...} rides along, which is where the metrics live.
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    LOGGER.handlers = [handler]
    LOGGER.setLevel(level)
    LOGGER.propagate = False


def log_event(event: str, **fields: object) -> None:
    LOGGER.info(event, extra={"fields": fields})


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    dollars = (
        input_tokens * CHAT_INPUT_PER_MILLION + output_tokens * CHAT_OUTPUT_PER_MILLION
    ) / 1_000_000
    return round(dollars, 6)


@contextmanager
def timed():
    """Measure a block in milliseconds: `with timed() as elapsed: ...`."""
    started = time.perf_counter()
    marker = {}
    try:
        yield marker
    finally:
        marker["ms"] = round((time.perf_counter() - started) * 1000)
