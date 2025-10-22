from typing import List, Optional

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import JSONResponse
from utils.dependencies import get_indexer
from utils.logger import get_logger

from .utils import (
    current_user_or_admin_partitions_list,
    require_partition_viewer,
    require_partitions_viewer,
)

logger = get_logger()

router = APIRouter()


@router.get("")
async def search_multiple_partitions(
    request: Request,
    partitions: Optional[List[str]] = Query(
        default=["all"], description="List of partitions to search"
    ),
    text: str = Query(..., description="Text to search semantically"),
    top_k: int = Query(5, description="Number of top results to return"),
    indexer=Depends(get_indexer),
    partition_viewer=Depends(require_partitions_viewer),
    user_partitions=Depends(current_user_or_admin_partitions_list),
):
    # Fetch user partitions if "all" is specified, or all partitions if super admin
    if partitions == ["all"]:
        partitions = user_partitions

    log = logger.bind(partitions=partitions, query=text, top_k=top_k)

    results = await indexer.asearch.remote(
        query=text, top_k=top_k, partition=partitions
    )
    log.info(
        "Semantic search on multiple partitions completed.",
        result_count=len(results),
    )

    documents = [
        {
            "link": str(request.url_for("get_extract", extract_id=doc.metadata["_id"])),
            "metadata": doc.metadata,
            "content": doc.page_content,
        }
        for doc in results
    ]

    return JSONResponse(
        status_code=status.HTTP_200_OK, content={"documents": documents}
    )


@router.get("/partition/{partition}")
async def search_one_partition(
    request: Request,
    partition: str,
    text: str = Query(..., description="Text to search semantically"),
    top_k: int = Query(5, description="Number of top results to return"),
    indexer=Depends(get_indexer),
    partition_viewer=Depends(require_partition_viewer),
):
    log = logger.bind(partition=partition, query=text, top_k=top_k)
    results = await indexer.asearch.remote(query=text, top_k=top_k, partition=partition)
    log.info(
        "Semantic search on single partition completed.", result_count=len(results)
    )
    documents = [
        {
            "link": str(request.url_for("get_extract", extract_id=doc.metadata["_id"])),
            "metadata": doc.metadata,
            "content": doc.page_content,
        }
        for doc in results
    ]

    return JSONResponse(
        status_code=status.HTTP_200_OK, content={"documents": documents}
    )


@router.get("/partition/{partition}/file/{file_id}")
async def search_file(
    request: Request,
    partition: str,
    file_id: str,
    text: str = Query(..., description="Text to search semantically"),
    top_k: int = Query(5, description="Number of top results to return"),
    indexer=Depends(get_indexer),
    partition_viewer=Depends(require_partition_viewer),
):
    log = logger.bind(partition=partition, file_id=file_id, query=text, top_k=top_k)
    results = await indexer.asearch.remote(
        query=text, top_k=top_k, partition=partition, filter={"file_id": file_id}
    )
    log.info("Semantic search on specific file completed.", result_count=len(results))

    documents = [
        {
            "link": str(request.url_for("get_extract", extract_id=doc.metadata["_id"])),
            "metadata": doc.metadata,
            "content": doc.page_content,
        }
        for doc in results
    ]

    return JSONResponse(
        status_code=status.HTTP_200_OK, content={"documents": documents}
    )
