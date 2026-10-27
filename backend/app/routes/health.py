"""Health endpoint. Deep DB checks land in T-105."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/api/health/status")
async def health() -> dict:
    return {"ok": True, "service": "sevenlab-pdf", "version": "0.1.0"}
