"""
HTML file loader implementation.

Converts HTML to Markdown, then processes embedded images for captioning.
"""

import re
from pathlib import Path
from typing import Dict, Optional, Union

import chardet
from html_to_markdown import convert
from langchain_core.documents.base import Document
from tqdm.asyncio import tqdm
from utils.logger import get_logger

from .base import BaseLoader

logger = get_logger()


class HTMLLoader(BaseLoader):
    """
    Loader for HTML files (.html, .htm).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._http_img_pattern = re.compile(r"!\[(.*?)\]\((https?://.*?)\)")
        self._data_uri_pattern = re.compile(
            r"!\[(.*?)\]\((data:image/[^;]+;base64,[^)]+)\)"
        )

    async def aload_document(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict] = None,
        save_markdown: bool = False,
    ) -> Document:
        if metadata is None:
            metadata = {}

        path = Path(file_path)

        # Read with encoding detection (chardet is already a project dependency)
        raw_bytes = path.read_bytes()
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding", "utf-8") or "utf-8"

        try:
            html_content = raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            html_content = raw_bytes.decode("utf-8", errors="ignore")

        # Convert HTML to Markdown (html_to_markdown strips scripts/styles)
        content = convert(html_content).strip()

        # Process images if captioning is enabled
        if self.image_captioning:
            http_matches = self._http_img_pattern.findall(content)
            data_uri_matches = self._data_uri_pattern.findall(content)
            total_images = len(http_matches) + len(data_uri_matches)

            logger.debug(
                "Found images in HTML",
                http_images=len(http_matches),
                data_uri_images=len(data_uri_matches),
                total_images=total_images,
            )

            if total_images > 0:
                tasks = {}
                for alt, url in (*http_matches, *data_uri_matches):
                    markdown_syntax = f"![{alt}]({url})"
                    tasks[markdown_syntax] = self.get_image_description(url)

                descriptions = await tqdm.gather(
                    *tasks.values(), desc="Captioning HTML images"
                )
                image_to_description = dict(zip(tasks.keys(), descriptions))

                for md_syntax, description in image_to_description.items():
                    content = content.replace(md_syntax, description)

        doc = Document(page_content=content, metadata=metadata)
        if save_markdown:
            self.save_content(content, str(path))

        return doc
