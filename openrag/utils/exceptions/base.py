from fastapi import status


class OpenRAGError(Exception):
    """Base class for all OpenRAG exceptions."""

    def __init__(
        self,
        message: str,
        code: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        **kwargs,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.extra = kwargs or {}
        super().__init__(f"{self.code}: {self.message}")

    def to_dict(self) -> dict:
        return {
            "detail": f"[{self.code}]: {self.message}",
            "extra": self.extra,
        }


# Subclass exceptions for specific error types
class EmbeddingError(OpenRAGError):
    """Base exception for all embedding-related errors."""

    def __init__(
        self, message, code, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, **kwargs
    ):
        super().__init__(message, code, status_code, **kwargs)


class VDBError(OpenRAGError):
    """Base exception for all vector database-related errors."""

    def __init__(
        self, message, code, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, **kwargs
    ):
        super().__init__(message, code, status_code, **kwargs)
