"""Failure types, so the HTTP layer can tell whose fault a failure is.

Library exceptions are caught at the edge where they happen and re-raised as one of
these, which keeps status-code decisions out of the pipeline and stack traces out of
responses.
"""


class InputError(Exception):
    """The caller sent something unusable. Maps to 400."""


class TooLargeError(InputError):
    """The caller sent more than the service accepts. Maps to 413."""


class UpstreamError(Exception):
    """A model or embedding call failed. Maps to 502."""
