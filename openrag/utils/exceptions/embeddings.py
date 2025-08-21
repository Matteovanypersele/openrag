from .base import EmbeddingError


class EmbeddingAPIError(EmbeddingError):
    """Raised when there's an API error with the embedding provider."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            code="EMBEDDING_API_ERROR",
            status_code=500,
            details=kwargs,
        )


class EmbeddingResponseError(EmbeddingError):
    """Raised when the response from the embedding provider is invalid or unexpected."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            code="EMBEDDING_RESPONSE_ERROR",
            status_code=422,
            details=kwargs,
        )


class UnexpectedEmbeddingError(EmbeddingError):
    """Raised for unexpected errors in embedding operations."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            code="UNEXPECTED_EMBEDDING_ERROR",
            status_code=500,
            details=kwargs,
        )
