from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from utils.dependencies import get_vectordb
from utils.logger import get_logger

from .utils import require_admin

logger = get_logger()
router = APIRouter()


@router.get("/")
async def list_users(vectordb=Depends(get_vectordb), admin_user=Depends(require_admin)):
    users = await vectordb.list_users.remote()
    logger.debug("Returned list of users.", user_count=len(users))
    return JSONResponse(status_code=status.HTTP_200_OK, content={"users": users})


@router.get("/info")
async def get_current_user(request: Request):
    """Get current authenticated user info"""
    user = request.state.user
    return user


@router.post("/")
async def create_user(
    display_name: Optional[str] = Form(None),
    external_user_id: Optional[str] = Form(None),
    is_admin: bool = Form(False),
    vectordb=Depends(get_vectordb),
    admin_user=Depends(require_admin),
):
    """
    Create a new user and generate a token.
    """
    user = await vectordb.create_user.remote(
        display_name=display_name,
        external_user_id=external_user_id,
        is_admin=is_admin,
    )
    logger.info("Created new user", user_id=user["id"])
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=user)


@router.get("/{user_id}")
async def get_user(
    user_id: int, vectordb=Depends(get_vectordb), admin_user=Depends(require_admin)
):
    """
    Get details of a specific user (without exposing token).
    """
    user = await vectordb.get_user.remote(user_id)
    return JSONResponse(status_code=status.HTTP_200_OK, content=user)


@router.delete("/{user_id}")
async def delete_user(
    user_id: int, vectordb=Depends(get_vectordb), admin_user=Depends(require_admin)
):
    """
    Delete a user.
    """
    if user_id == 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete default admin user."
        )
    await vectordb.delete_user.remote(user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{user_id}/regenerate_token")
async def regenerate_user_token(user_id: int, vectordb=Depends(get_vectordb)):
    """
    Regenerate a user's token.
    """
    user = await vectordb.regenerate_user_token.remote(user_id)
    logger.info("Regenerated user token", user_id=user_id)
    return JSONResponse(status_code=status.HTTP_200_OK, content=user)
