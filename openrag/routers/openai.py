import json
from urllib.parse import quote

import consts
from components.pipeline import RagPipeline
from config import load_config
from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.documents.base import Document
from models.openai import (
    OpenAIChatCompletionRequest,
    OpenAICompletionRequest,
)
from utils.dependencies import get_vectordb
from utils.logger import get_logger

from .utils import (
    check_llm_model_availability,
    current_user,
    current_user_or_admin_partitions,
    current_user_or_admin_partitions_list,
    get_partition_name,
)

logger = get_logger()
config = load_config()
router = APIRouter()

ragpipe = RagPipeline(config=config)


@router.get(
    "/models",
    summary="OpenAI-compatible model listing endpoint",
    description="""
    OpenAI-compatible endpoint to list all available models.
    
    Returns a list of models that can be used with OpenRAG, including:
    - All available partitions formatted as 'openrag-{partition_name}'
    - A special 'openrag-all' model to query across all partitions
    
    The response format mimics the OpenAI models listing API for compatibility.
    """,
    response_description="A list of available models in OpenAI format",
)
async def list_models(
    _: None = Depends(check_llm_model_availability),
    vectordb=Depends(get_vectordb),
    user_partitions=Depends(current_user_or_admin_partitions),
):
    if [p["partition"] for p in user_partitions] == ["all"]:
        user_partitions = await vectordb.list_partitions.remote()
    logger.debug("Listing models", partition_count=len(user_partitions))

    models = []
    for partition in user_partitions:
        model_id = f"{consts.PARTITION_PREFIX}{partition['partition']}"
        models.append(
            {
                "id": model_id,
                "object": "model",
                "created": partition["created_at"],
                "owned_by": "OpenRAG",
            }
        )

    models.append(
        {
            "id": f"{consts.PARTITION_PREFIX}all",
            "object": "model",
            "created": 0,
            "owned_by": "OpenRAG",
        }
    )
    return JSONResponse(content={"object": "list", "data": models})


def __prepare_sources(request: Request, docs: list[Document]):
    links = []
    for doc in docs:
        doc_metadata = dict(doc.metadata)
        file_url = str(request.url_for("static", path=doc_metadata["filename"]))
        encoded_url = quote(file_url, safe=":/")
        links.append(
            {
                "file_url": encoded_url,
                "chunk_url": str(
                    request.url_for("get_extract", extract_id=doc_metadata["_id"])
                ),
                **doc_metadata,
            }
        )
    return links


@router.post(
    "/chat/completions",
    summary="OpenAI compatible chat completion endpoint using RAG",
    description="""
    OpenAI-compatible chat completion endpoint that leverages Retrieval-Augmented Generation (RAG).
    
    This endpoint accepts chat messages in OpenAI format and uses the specified model to generate
    a completion. The model selection determines which document partition(s) will be queried:
    - 'openrag-{partition_name}': Queries only the specified partition
    - 'openrag-all': Queries across all available partitions
    
    Previous messages provide conversation context. The system enriches the prompt with relevant documents retrieved
    from the vector database before sending to the LLM.
    """,
)
async def openai_chat_completion(
    request2: Request,
    request: OpenAIChatCompletionRequest = Body(...),
    _: None = Depends(check_llm_model_availability),
    user=Depends(current_user),
    user_partitions=Depends(current_user_or_admin_partitions_list),
):
    model_name = request.model
    log = logger.bind(model=model_name, endpoint="/chat/completions")

    if (
        not request.messages
        or request.messages[-1].role != "user"
        or not request.messages[-1].content
    ):
        log.warning("Invalid request: missing or malformed user message.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The last message must be a non-empty user message",
        )

    try:
        partitions = await get_partition_name(
            model_name, user_partitions, is_admin=user["is_admin"]
        )
        log.debug(f"Using partitions: {partitions}")
    except Exception as e:
        log.warning("Invalid model or partition", error=str(e))
        raise

    try:
        llm_output, docs = await ragpipe.chat_completion(
            partition=partitions, payload=request.model_dump()
        )
        log.debug("RAG chat completion pipeline executed.")
    except Exception as e:
        log.exception("Chat completion failed.", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat completion failed: {str(e)}",
        )

    metadata = __prepare_sources(request2, docs)
    metadata_json = json.dumps({"sources": metadata})

    if request.stream:

        async def stream_response():
            async for line in llm_output:
                if line.startswith("data:"):
                    if "[DONE]" in line:
                        yield f"{line}\n\n"
                    else:
                        try:
                            data_str = line[len("data: ") :]
                            data = json.loads(data_str)
                            data["model"] = model_name
                            data["extra"] = metadata_json
                            yield f"data: {json.dumps(data)}\n\n"
                        except json.JSONDecodeError as e:
                            log.exception(
                                "Failed to decode streamed chunk.", error=str(e)
                            )
                            raise

        return StreamingResponse(stream_response(), media_type="text/event-stream")
    else:
        try:
            chunk = await llm_output.__anext__()
            chunk["model"] = model_name
            chunk["extra"] = metadata_json
            log.debug("Returning non-streaming completion chunk.")
            return JSONResponse(content=chunk)
        except Exception as e:
            log.warning("Error while generating answer", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error while generating answer: {str(e)}",
            )


@router.post(
    "/completions",
    summary="OpenAI compatible completion endpoint using RAG",
    description="""
    OpenAI-compatible text completion endpoint that leverages Retrieval-Augmented Generation (RAG).
    
    This endpoint accepts a prompt in OpenAI format and uses the specified model to generate
    a text completion. The model selection determines which document partition(s) will be queried:
    - 'openrag-{partition_name}': Queries only the specified partition
    - 'openrag-all': Queries across all available partitions
    
    The system enriches the prompt with relevant documents retrieved from the vector database
    before sending to the LLM, allowing the completion to include information from your document store.
    """,
)
async def openai_completion(
    request2: Request,
    request: OpenAICompletionRequest,
    _: None = Depends(check_llm_model_availability),
    user=Depends(current_user),
    user_partitions=Depends(current_user_or_admin_partitions_list),
):
    model_name = request.model
    log = logger.bind(model=model_name, endpoint="/completions")

    if not request.prompt:
        log.warning("Prompt is missing.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The prompt is required",
        )

    if request.stream:
        log.warning("Streaming not supported for this endpoint.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Streaming is not supported for this endpoint",
        )

    try:
        partitions = await get_partition_name(
            model_name, user_partitions, is_admin=user["is_admin"]
        )

    except Exception as e:
        log.warning(f"Invalid model or partition: {e}")
        raise

    try:
        llm_output, docs = await ragpipe.completions(
            partition=partitions, payload=request.model_dump()
        )
        log.debug("RAG completion pipeline executed.")
    except Exception as e:
        log.exception("Completion request failed.", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Completion failed: {str(e)}",
        )

    metadata = __prepare_sources(request2, docs)
    metadata_json = json.dumps({"sources": metadata})

    try:
        complete_response = await llm_output.__anext__()
        complete_response["extra"] = metadata_json
        log.debug("Returning completion response.")
        return JSONResponse(content=complete_response)
    except Exception as e:
        log.warning("No response from LLM.", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"No response from LLM: {str(e)}",
        )
