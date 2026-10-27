"""Password-only login (scraper pattern). Full implementation lands in T-002."""
from fastapi import APIRouter, HTTPException, status

router = APIRouter()


@router.post("")
async def login() -> dict:
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="T-002")


@router.delete("")
async def logout() -> dict:
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="T-002")
