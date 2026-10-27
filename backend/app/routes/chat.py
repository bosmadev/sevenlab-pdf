"""Chat / RAG endpoints. Full implementation in T-011 / T-012."""
from fastapi import APIRouter, HTTPException, status

router = APIRouter()


@router.post("/ask/stream")
async def ask_stream() -> dict:
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="T-012")
