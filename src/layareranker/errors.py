"""Exception hierarchy. Each type maps to one caller action; nothing is wrapped into a catch-all."""


class LayaRerankerError(Exception):
    """Base class for every error raised by layareranker."""


class ConfigurationError(LayaRerankerError):
    """Invalid settings or preset, or a prompt that does not fit the checkpoint's budget."""


class InputError(LayaRerankerError, ValueError):
    """The caller's query or passages are invalid (empty, too large, or unscoreable)."""


class ScoringError(LayaRerankerError):
    """The model returned an answer that cannot be turned into a score (missing, NaN, out of range)."""


class Overloaded(LayaRerankerError):
    """The batching queue is full; retry later."""


class DeadlineExceeded(LayaRerankerError):
    """The request's deadline passed before it could be scored."""


class BackendFatal(LayaRerankerError):
    """The inference device is in an unrecoverable state; the process must be restarted."""
