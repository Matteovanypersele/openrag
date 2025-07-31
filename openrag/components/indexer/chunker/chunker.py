import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from omegaconf import OmegaConf
from tqdm.asyncio import tqdm
from utils.logger import get_logger

from ...utils import llmSemaphore, load_config, load_sys_template
from .utills import split_md_elements, combine_chunks, add_overlap


logger = get_logger()
config = load_config()
prompt_paths = Path(config.paths.get("prompts_dir"))
chunk_contextualizer_pmpt = config.prompt.get("chunk_contextualizer_pmpt")
CHUNK_CONTEXTUALIZER = load_sys_template(prompt_paths / chunk_contextualizer_pmpt)


class BaseChunker(ABC):
    """Base class for document chunkers with built-in contextualization capability."""

    def __init__(
        self,
        contextual_retrieval: bool = False,
        llm: Optional[ChatOpenAI] = None,
        **kwargs,
    ):
        self.contextual_retrieval = contextual_retrieval
        self.llm = llm
        self.context_generator = None
        self._page_pattern = re.compile(r"\[PAGE_(\d+)\]")

        if self.contextual_retrieval:
            if not isinstance(llm, ChatOpenAI):
                raise ValueError(
                    "The `llm` should be of type `ChatOpenAI` if contextual_retrieval is `True`"
                )
            prompt = ChatPromptTemplate.from_template(template=CHUNK_CONTEXTUALIZER)
            self.context_generator = (prompt | llm | StrOutputParser()).with_retry(
                retry_if_exception_type=(Exception,),
                wait_exponential_jitter=False,
                stop_after_attempt=2,
            )

    async def _generate_context(
        self, first_chunks: str, prev_chunk: str, chunk: str, source: str
    ) -> str:
        """Generate context for a given chunk of text."""
        async with llmSemaphore:
            try:
                return await self.context_generator.ainvoke(
                    {
                        "first_chunks": first_chunks,
                        "prev_chunk": prev_chunk,
                        "chunk": chunk,
                        "source": source,
                    }
                )
            except Exception as e:
                logger.warning(
                    f"Error when contextualizing a chunk of document `{source}`: {e}"
                )
                return ""

    async def _contextualize_chunks(self, chunks: list[str], source: str) -> list[str]:
        """Contextualize a list of document chunks."""
        if not self.contextual_retrieval or len(chunks) < 2:
            return chunks

        try:
            tasks = []
            for i in range(len(chunks)):
                prev_chunk = chunks[i - 1] if i > 0 else ""
                curr_chunk = chunks[i]
                first_chunks = "\n".join(chunks[:4])

                tasks.append(
                    self._generate_context(
                        first_chunks=first_chunks,
                        prev_chunk=prev_chunk,
                        chunk=curr_chunk,
                        source=source,
                    )
                )

            contexts = await tqdm.gather(
                *tasks,
                total=len(tasks),
                desc=f"Contextualizing chunks of *{Path(source).name}*",
            )

            # Format contextualized chunks
            chunk_format = """Context: {chunk_context}\n\nChunk: {chunk}"""
            contexts = [
                chunk_format.format(
                    chunk=chunk, chunk_context=context, source=Path(source).name
                )
                for chunk, context in zip(chunks, contexts)
            ]

            return contexts

        except Exception as e:
            logger.warning(f"Error when contextualizing chunks from `{source}`: {e}")
            return chunks

    def _get_chunk_page_info(self, chunk_str: str, previous_page=1):
        """
        Determine the start and end pages for a text chunk containing [PAGE_N] separators.
        PAGE_N marks the end of page N - text before separator is on page N.
        """
        # Find all page separator matches in the chunk
        matches = list(self._page_pattern.finditer(chunk_str))

        if not matches:
            # No separators found - entire chunk is on previous page
            return {"start_page": previous_page, "end_page": previous_page}

        first_match = matches[0]
        last_match = matches[-1]
        last_char_idx = len(chunk_str) - 1

        # Determine start page
        if first_match.start() == 0:
            # Chunk starts with a separator - begins on next page
            start_page = int(first_match.group(1)) + 1
        else:
            # Text precedes first separator - starts on previous page
            start_page = previous_page

        # Determine end page
        if last_match.end() - 1 == last_char_idx:
            # Chunk ends exactly at a separator - ends on that page
            end_page = int(last_match.group(1))
        else:
            # Chunk ends after separator - ends on next page
            end_page = int(last_match.group(1)) + 1

        return {"start_page": start_page, "end_page": end_page}

    @abstractmethod
    async def split_document(self, docs: list[Document], task_id: str = None):
        pass


class RecursiveSplitter(BaseChunker):
    """RecursiveSplitter splits documents into chunks using recursive character splitting."""

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 20, **kwargs):
        super().__init__(**kwargs)

        from langchain.text_splitter import RecursiveCharacterTextSplitter

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda x: self.llm.get_num_tokens(x)
            if self.llm
            else len(x),
        )

    async def split_document(self, doc: Document, task_id: str = None):
        metadata = doc.metadata
        log = logger.bind(
            file_id=metadata.get("file_id"),
            partition=metadata.get("partition"),
            task_id=task_id,
        )
        log.info("Starting document chunking")
        source = metadata["source"]

        # Split the document into chunks of text, tables, and images
        all_content = doc.page_content.strip()
        splits = split_md_elements(all_content)

        # Add overlap to image and table chunks
        splits = add_overlap(
            chunks=splits,
            target_chunk_types=["table", "image"],
            add_before=True,
            add_after=True,
            chunk_overlap=self.chunk_overlap,
        )

        # only split text elements into chunks
        chunks = []
        for chunk_type, content in splits:
            if chunk_type == "text":
                chunks.extend(self.splitter.split_text(content))
            else:
                chunks.append(content)

        # regrouping chunks based on token length
        chunks = combine_chunks(
            chunks=chunks, llm=self.llm, chunk_max_size=self.chunk_size
        )

        chunks_w_context = chunks  # Default to original chunks if no contextualization

        if self.contextual_retrieval:
            log.info("Contextualizing chunks")
            chunks_w_context = await self._contextualize_chunks(chunks, source=source)

        filtered_chunks = []
        prev_page_num = 1
        for chunk, chunk_w_context in zip(chunks, chunks_w_context):
            page_info = self._get_chunk_page_info(
                chunk_str=chunk, previous_page=prev_page_num
            )
            start_page = page_info["start_page"]
            end_page = page_info["end_page"]
            prev_page_num = end_page

            if len(chunk.strip()) > 10:
                filtered_chunks.append(
                    Document(
                        page_content=chunk_w_context,
                        metadata={**metadata, "page": start_page},
                    )
                )
        log.info("Document chunking completed")
        return filtered_chunks


class SemanticSplitter(BaseChunker):
    """SemanticSplitter splits documents into semantically meaningful chunks."""

    def __init__(
        self,
        min_chunk_size: int = 1000,
        embeddings=None,
        breakpoint_threshold_amount=85,
        **kwargs,
    ):
        super().__init__(**kwargs)

        from langchain_experimental.text_splitter import SemanticChunker

        self.semantic_splitter = SemanticChunker(
            embeddings=embeddings,
            buffer_size=1,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            min_chunk_size=min_chunk_size,
        )

        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100,
            length_function=lambda x: self.llm.get_num_tokens(x)
            if self.llm
            else len(x),
        )

        self.chunk_size = 512
        self.chunk_overlap = 100

    def split_text(self, text: str):
        # split sematically meaningful chunks
        splits = self.semantic_splitter.split_text(text)

        # apply recursive character splitter to each chunk (this would add overlapping between text chunks)
        splits_l = [self.recursive_splitter.split_text(s) for s in splits]
        splits = sum(splits_l, [])
        return splits

    async def split_document(self, doc: Document, task_id: str = None):
        metadata = doc.metadata
        log = logger.bind(
            file_id=metadata.get("file_id"),
            partition=metadata.get("partition"),
            task_id=task_id,
        )
        log.info("Starting document chunking")
        source = metadata["source"]

        # Split the document into chunks of text, tables, and images
        all_content = doc.page_content.strip()
        splits = split_md_elements(all_content)

        # Add overlap image and table chunks
        splits = add_overlap(
            chunks=splits,
            target_chunk_types=["table", "image"],
            add_before=True,
            add_after=True,
            chunk_overlap=self.chunk_overlap,
        )

        # only split text elements into chunks
        chunks = []
        for chunk_type, content in splits:
            if chunk_type == "text":
                chunks.extend(self.split_text(content))
            else:
                chunks.append(content)

        # regrouping chunks based on token length
        chunks = combine_chunks(
            chunks=chunks, llm=self.llm, chunk_max_size=self.chunk_size
        )

        # regrouping chunks based on token length
        chunks = combine_chunks(
            chunks=chunks, llm=self.llm, chunk_max_size=self.chunk_size
        )

        chunks_w_context = chunks  # Default to original chunks if no contextualization
        if self.contextual_retrieval:
            log.info("Contextualizing chunks")
            chunks_w_context = await self._contextualize_chunks(chunks, source=source)

        filtered_chunks = []
        prev_page_num = 1
        for chunk, chunk_w_context in zip(chunks, chunks_w_context):
            page_info = self._get_chunk_page_info(
                chunk_str=chunk, previous_page=prev_page_num
            )
            start_page = page_info["start_page"]
            end_page = page_info["end_page"]
            prev_page_num = end_page

            if len(chunk.strip()) > 10:
                filtered_chunks.append(
                    Document(
                        page_content=chunk_w_context,
                        metadata={**metadata, "page": start_page},
                    )
                )
        log.info("Document chunking completed")
        return filtered_chunks


class MarkDownSplitter(BaseChunker):
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 20, **kwargs):
        super().__init__(**kwargs)

        self.chunk_size = chunk_size
        self.overlap = chunk_overlap

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            # ("###", "Header 3"),
            # ("####", "Header 4"),
        ]
        self.md_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

        self.recurive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda x: self.llm.get_num_tokens(x),
        )

        self.chunk_overlap = chunk_overlap

    def split_md_chunks(self, text: str) -> list[str]:
        # split the text into chunks based on headers
        splits: list[Document] = self.md_header_splitter.split_text(text)

        # regrouping chunks based on token length
        combined_elements = combine_chunks(
            chunks=splits, llm=self.llm, chunk_max_size=self.chunk_size
        )

        # use recussive splitter to further split the chunks (this would add overlapping between text chunks)
        overlapped_elements = list(
            map(lambda x: self.recurive_splitter.split_text(x), combined_elements)
        )
        return sum(overlapped_elements, [])

    async def split_document(self, doc: Document, task_id: str = None):
        metadata = doc.metadata
        log = logger.bind(
            file_id=metadata.get("file_id"),
            partition=metadata.get("partition"),
            task_id=task_id,
        )
        log.info("Starting document chunking")
        source = metadata["source"]

        # Split the document into chunks of text, tables, and images
        all_content = doc.page_content.strip()
        splits = split_md_elements(all_content)

        # Add overlap image and table chunks
        splits = add_overlap(
            chunks=splits,
            target_chunk_types=["table", "image"],
            add_before=True,
            add_after=True,
            chunk_overlap=self.chunk_overlap,
        )

        # only split text elements into chunks
        chunks = []
        for chunk_type, content in splits:
            if chunk_type == "text":
                chunks.extend(self.split_md_chunks(content))
            else:
                chunks.append(content)

        # regrouping chunks based on token length
        chunks = combine_chunks(
            chunks=chunks, llm=self.llm, chunk_max_size=self.chunk_size
        )

        chunks_w_context = chunks  # Default to original chunks if no contextualization
        if self.contextual_retrieval:
            log.info("Contextualizing chunks")
            chunks_w_context = await self._contextualize_chunks(chunks, source=source)

        filtered_chunks = []
        prev_page_num = 1
        for chunk, chunk_w_context in zip(chunks, chunks_w_context):
            page_info = self._get_chunk_page_info(
                chunk_str=chunk, previous_page=prev_page_num
            )
            start_page = page_info["start_page"]
            end_page = page_info["end_page"]
            prev_page_num = end_page

            if len(chunk.strip()) > 10:
                filtered_chunks.append(
                    Document(
                        page_content=chunk_w_context,
                        metadata={**metadata, "page": start_page},
                    )
                )
        log.info("Document chunking completed")
        return filtered_chunks


class ChunkerFactory:
    CHUNKERS = {
        "recursive_splitter": RecursiveSplitter,
        "semantic_splitter": SemanticSplitter,
        "markdown_splitter": MarkDownSplitter,
    }

    @staticmethod
    def create_chunker(
        config: OmegaConf,
        embedder: Optional[HuggingFaceBgeEmbeddings | HuggingFaceEmbeddings] = None,
    ) -> BaseChunker:
        """
        Create and initialize a chunker based on the provided configuration.
        Args:
            config (OmegaConf): Configuration object containing chunker parameters.
            embedder (Optional[HuggingFaceBgeEmbeddings | HuggingFaceEmbeddings]): Optional embedder to be used if the chunker type is 'semantic_splitter'.
        """

        # Extract parameters
        chunker_params = OmegaConf.to_container(config.chunker, resolve=True)
        name = chunker_params.pop("name")

        # Initialize and return the chunker
        chunker_class: BaseChunker = ChunkerFactory.CHUNKERS.get(name)
        if not chunker_class:
            raise ValueError(f"Chunker '{name}' is not recognized.")

        # Add embeddings if semantic splitter is selected
        if name == "semantic_splitter":
            if embedder is not None:
                chunker_params.update({"embeddings": embedder})
            else:
                raise AttributeError(
                    f"{name} type chunker requires the `embedder` parameter"
                )

        # Include contextual retrieval if specified
        chunker_params["llm"] = ChatOpenAI(**config.vlm)
        return chunker_class(**chunker_params)
