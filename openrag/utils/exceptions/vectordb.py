from .base import VDBError


class VDBConnectionError(VDBError):
    """Raised when connection to vector database fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            code="VECTOR_DB_CONNECTION_ERROR",
            status_code=503,
            details=kwargs,
        )


class VDBCreateOrLoadCollectionError(VDBError):
    """Raised when there's an issue with collection operations."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            code="VECTOR_DB_COLLECTION_ERROR",
            status_code=422,
            details=kwargs,
        )


class VDBInsertError(VDBError):
    """Raised when data insertion fails."""

    def __init__(self, message: str, status_code: int = 422, **kwargs):
        super().__init__(
            message=message,
            code="VECTOR_DB_INSERT_ERROR",
            status_code=status_code,
            details=kwargs,
        )


class VDBFileIDAlreadyExistsError(VDBError):
    """Raised when a file already exists in the vector database."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            code="VECTOR_DB_FILE_ALREADY_EXISTS",
            status_code=409,
            details=kwargs,
        )


class VDBDeleteError(VDBError):
    """Raised when data deletion fails."""

    def __init__(
        self,
        message: str,
        status_code=422,
        **kwargs,
    ):
        super().__init__(
            message=message,
            code="VECTOR_DB_DELETE_ERROR",
            status_code=status_code,
            details=kwargs,
        )


class VDBSearchError(VDBError):
    """Raised when vector search fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            code="VECTOR_DB_SEARCH_ERROR",
            status_code=422,
            details=kwargs,
        )


class VDBPartitionNotFound(VDBError):
    """Raised when a partition is not found in the vector database."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            code="VECTOR_DB_PARTITION_NOT_FOUND",
            status_code=404,
            details=kwargs,
        )


class VDBFileNotFoundError(VDBError):
    """Raised when a file is not found in the vector database."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            code="VECTOR_DB_FILE_NOT_FOUND",
            status_code=404,
            details=kwargs,
        )


class UnexpectedVDBError(VDBError):
    """Raised for unexpected errors in vector database operations."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            code="UNEXPECTED_VECTOR_DB_ERROR",
            status_code=500,
            details=kwargs,
        )
