"""
Auto-scaling hooks for GPU router.
Triggers model pulls, lane warm-ups, and capacity alerts.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from typing import Any

import httpx

from app.config import RouterConfig
from app.state import ClusterState, LaneState

log = logging.getLogger("ollama-router.autoscale")

PULL_COOLDOWN = 300  # seconds between pull attempts for same model
_last_pull: dict[tuple[str, str], float] = {}  # (lane, model) -> timestamp


async def ensure_model_on_lane(lane: LaneState, model: str, config: RouterConfig) -> bool:
    """Ensure model exists on lane, pull if missing. Returns True if available."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"http://127.0.0.1:{lane.cfg.port}/api/tags")
            if resp.status_code != 200:
                return False
            models = {m["name"] for m in resp.json().get("models", [])}
            if model in models:
                return True
    except Exception:
        pass

    # Check cooldown
    key = (lane.cfg.name, model)
    now = time.time()
    if _last_pull.get(key, 0) > now - PULL_COOLDOWN:
        return False

    _last_pull[key] = now
    log.info(f"Pulling model {model} to lane {lane.cfg.name}...")

    try:
        # Use ollama pull via subprocess (requires ollama CLI in PATH)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = lane.cuda_visible_devices
        env["OLLAMA_VULKAN"] = "false"
        result = subprocess.run(
            ["ollama", "pull", model],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            log.info(f"Successfully pulled {model} to {lane.cfg.name}")
            return True
        else:
            log.error(f"Failed to pull {model}: {result.stderr}")
            return False
    except Exception as e:
        log.error(f"Exception pulling {model}: {e}")
        return False


async def warmup_lane(lane: LaneState, config: RouterConfig) -> None:
    """Pre-load common models on a lane to reduce cold-start latency."""
    common_models = ["llama3.1:8b", "nomic-embed-text:latest"]
    for model in common_models:
        await ensure_model_on_lane(lane, model, config)
        await asyncio.sleep(1)  # stagger pulls


def check_capacity_pressure(state: ClusterState, threshold: float = 0.85) -> list[dict[str, Any]]:
    """Check for GPUs under memory pressure. Returns list of pressured GPUs."""
    pressured = []
    for gpu_idx, gpu in state.gpus.items():
        if gpu.total_mib > 0:
            used_pct = 1.0 - (gpu.free_mib / gpu.total_mib)
            if used_pct > threshold:
                pressured.append({
                    "gpu": gpu_idx,
                    "name": gpu.name,
                    "used_pct": used_pct,
                    "free_mib": gpu.free_mib,
                    "total_mib": gpu.total_mib,
                })
    return pressured


async def trigger_scale_alert(pressured: list[dict[str, Any]]) -> None:
    """Hook for external alerting (webhook, email, etc.)."""
    if not pressured:
        return
    # TODO: wire to alertmanager, pagerduty, slack, etc.
    log.warning(f"GPU capacity pressure detected: {pressured}")