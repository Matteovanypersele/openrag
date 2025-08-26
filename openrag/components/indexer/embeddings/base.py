from langchain_core.documents.base import Document


class BaseEmbedding:
    """Base class for all embedding models."""

    async def embed_documents(self, chunks: list[Document], logger) -> list[dict]:
        """
        Asynchronously embed documents using the configured embedder.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    async def embed_query(self, query: str, logger) -> list[float]:
        """
        Asynchronously embed a query using the configured embedder.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @property
    def embedding_dimension(self) -> int:
        """
        Returns the dimension of the embedding vector.
        This is used to validate the vector size during indexing and searching.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
