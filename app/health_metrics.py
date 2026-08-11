
from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import time
import psutil

router = APIRouter()
_start = time.time()

@router.get("/health")
def health():
    return {"status": "ok", "uptime_s": int(time.time() - _start)}

@router.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
