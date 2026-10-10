"""Errors raised by the labeling layer. The API maps them to HTTP status codes."""

from __future__ import annotations


class LabelingError(Exception):
    pass


class BadRequest(LabelingError):
    """The request cannot be carried out as stated (HTTP 400)."""


class Conflict(LabelingError):
    """The request conflicts with the clip's current state (HTTP 409)."""


class PropagationFailed(LabelingError):
    """Propagation could not complete (HTTP 500 / job error)."""
