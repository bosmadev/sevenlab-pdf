"""Document upload and management. Full implementation in T-004 / T-013."""
from fastapi import APIRouter, HTTPException, status

router = APIRouter()


@router.post("/upload")
async def upload() -> dict:
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="T-004")


@router.get("/")
async def list_documents() -> list:
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="T-013")
