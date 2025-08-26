from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from utils.dependencies import get_vectordb
from utils.logger import get_logger

logger = get_logger()

# Create an APIRouter instance
router = APIRouter()



@router.get("/{extract_id}")
async def get_extract(extract_id: str, vectordb=Depends(get_vectordb)):
    log = logger.bind(extract_id=extract_id)
    try:
        chunk = await vectordb.get_chunk_by_id.remote(extract_id)
        if chunk is None:
            log.warning("Extract not found.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Extract '{extract_id}' not found.",
            )
        log.info("Extract successfully retrieved.")
    except Exception as e:
        log.exception("Failed to retrieve extract.", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve extract: {str(e)}",
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"page_content": chunk.page_content, "metadata": chunk.metadata},
    )
