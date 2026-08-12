from __future__ import annotations

import re
import subprocess
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import LaneConfig, RouterConfig

# ---------------------------------------------------------------------------
# Physical GPU model (from nvidia-smi). Index aligns with CUDA_VISIBLE_DEVICES.
# ---------------------------------------------------------------------------


@dataclass
class GpuInfo:
    index: int
    name: str = ""
    total_mib: int = 0
    used_mib: int = 0
    free_mib: int = 0

    @property
    def free_gib(self) -> float:
        return round(self.free_mib / 1024, 2)

    @property
    def total_gib(self) -> float:
        return round(self.total_mib / 1024, 2)


def read_gpus() -> dict[int, GpuInfo]:
    """Read physical GPU inventory + free VRAM from nvidia-smi (source of truth)."""
    gpus: dict[int, GpuInfo] = {}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return gpus
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        idx, name, used, total, free = parts[0], parts[1], parts[2], parts[3], parts[4]
        gpus[int(idx)] = GpuInfo(
            index=int(idx),
            name=name,
            used_mib=int(used),
            total_mib=int(total),
            free_mib=int(free),
        )
    return gpus


# ---------------------------------------------------------------------------
# Lane runtime state (per ollama lane).
# ---------------------------------------------------------------------------


@dataclass
class LaneState:
    cfg: LaneConfig
    healthy: bool = False
    last_error: str = ""
    models: set[str] = field(default_factory=set)
    loaded: dict[str, int] = field(default_factory=dict)  # model -> bytes resident
    catalog: dict[str, int] = field(default_factory=dict)  # model -> on-disk bytes (from /api/tags)
    loaded_vram: dict[str, int] = field(default_factory=dict)  # model -> bytes actually in VRAM
    # model -> epoch seconds when Ollama will evict it (from /api/ps expires_at).
    # 0.0 / missing = pinned ("never") and must NOT be reclaimed.
    loaded_expires: dict[str, float] = field(default_factory=dict)
    gpu: GpuInfo | None = None
    failures: int = 0
    circuit_open_until: float = 0.0
    last_recovery: float = 0.0
    recovery_attempts: int = 0

    @property
    def physical_gpu(self) -> int:
        return self.cfg.physical_gpu

    def fits(self, vram_gib: float, margin_gib: float = 0.5) -> bool:
        if self.gpu is None:
            return False
        # Free VRAM minus already-loaded models on this lane minus safety margin.
        reserved = sum(self.loaded.values()) / (1024**3)
        available = self.gpu.free_gib - reserved - margin_gib
        return available >= vram_gib

    @property
    def resident_gib(self) -> float:
        return round(sum(self.loaded.values()) / (1024**3), 2)


# ---------------------------------------------------------------------------
# Cluster state: lanes + physical GPUs, refreshed on a timer.
# ---------------------------------------------------------------------------


class ClusterState:
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg
        self.gpus: dict[int, GpuInfo] = {}
        self.lanes: dict[str, LaneState] = {lane.name: LaneState(cfg=lane) for lane in cfg.lanes}
        self.last_refresh: float = 0.0
        self.refresh()

    def refresh(self) -> None:
        self.gpus = read_gpus()
        # Attach each lane to its physical GPU.
        for _name, ls in self.lanes.items():
            ls.gpu = self.gpus.get(ls.physical_gpu)
        # Introspect each lane concurrently.
        with httpx.Client(timeout=httpx.Timeout(connect=self.cfg.connect_timeout, read=10, write=10, pool=10)) as client:
            for _name, ls in self.lanes.items():
                self._probe_lane(client, ls)
        self.last_refresh = time.time()

    def _probe_lane(self, client: httpx.Client, ls: LaneState) -> None:
        base = f"http://127.0.0.1:{ls.cfg.port}"
        try:
            r = client.get(f"{base}/api/tags", timeout=8)
            r.raise_for_status()
            data = r.json()
            ls.models = {m["name"] for m in data.get("models", [])}
            # On-disk size per model is the single best VRAM predictor we have
            # before a model is resident -- name-based guessing badly
            # under-estimates MoE models (a "30b-a3b" is 25GB on disk, not 15).
            ls.catalog = {m["name"]: int(m.get("size", 0)) for m in data.get("models", [])}
            ps = client.get(f"{base}/api/ps", timeout=8)
            ps.raise_for_status()
            running = ps.json().get("models", [])
            ls.loaded = {m["name"]: m.get("size", 0) for m in running}
            # size_vram < size means layers spilled to CPU; a heavily spilled
            # model runs ~25x slower, so the router needs to see this.
            ls.loaded_vram = {m["name"]: m.get("size_vram", 0) for m in running}
            # expires_at is Ollama's own eviction countdown. A model whose timer
            # is within the idle grace is unused and will be evicted the moment a
            # bigger model needs the room -- so its VRAM is reclaimable when
            # judging fit. "never" (keep_alive=-1) is pinned: not reclaimable.
            ls.loaded_expires = {}
            for m in running:
                exp = m.get("expires_at")
                if not exp or str(exp).lower() == "never":
                    continue
                try:
                    ls.loaded_expires[m["name"]] = datetime.fromisoformat(str(exp)).timestamp()
                except Exception:  # noqa: BLE001 -- unusual timestamp; treat as pinned
                    ls.loaded_expires[m["name"]] = 0.0
            ls.healthy = True
            ls.last_error = ""
        except Exception as e:  # noqa: BLE001 — broad by design: any lane error = unhealthy
            ls.healthy = False
            ls.last_error = str(e)
            ls.models = set()
            ls.loaded = {}
            ls.catalog = {}
            ls.loaded_vram = {}
            ls.loaded_expires = {}

    def gpu_fraction(self, ls: LaneState, model: str) -> float:
        """Fraction of a resident model's bytes that live in VRAM (1.0 = fully on GPU)."""
        total = ls.loaded.get(model, 0)
        if not total:
            return 0.0
        return ls.loaded_vram.get(model, 0) / total

    def reclaimable_gib(self, ls: LaneState, model: str) -> float:
        """VRAM (GiB) Ollama will evict to load `model`: any resident model
        other than the target that is NOT pinned (expires_at != "never"). Ollama
        proactively evicts idle/least-recently-used models to make room for a new
        load, so their VRAM is available on demand. Pinned models (keep_alive=-1)
        are the only ones that are not reclaimable."""
        tot = 0.0
        for name, vram in ls.loaded_vram.items():
            if name == model:
                continue
            if ls.loaded_expires.get(name, 0.0) != 0.0:  # 0.0 == pinned/"never"
                tot += vram / (1024**3)
        return round(tot, 2)

    def is_circuit_open(self, ls: LaneState, now: float | None = None) -> bool:
        now = now or time.time()
        return ls.circuit_open_until > now

    def mark_failure(self, ls: LaneState) -> None:
        ls.failures += 1
        if ls.failures >= self.cfg.circuit_breaker_threshold:
            ls.circuit_open_until = time.time() + self.cfg.circuit_breaker_cooldown_sec
            ls.failures = 0

    def mark_success(self, ls: LaneState) -> None:
        ls.failures = 0
        ls.circuit_open_until = 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "last_refresh": self.last_refresh,
            "gpus": {i: g.__dict__ for i, g in self.gpus.items()},
            "lanes": {
                n: {
                    "port": ls.cfg.port,
                    "role": ls.cfg.role,
                    "physical_gpu": ls.physical_gpu,
                    "healthy": ls.healthy,
                    "error": ls.last_error,
                    "model_count": len(ls.models),
                    "loaded": ls.loaded,
                    "resident_gib": ls.resident_gib,
                    "free_gib": ls.gpu.free_gib if ls.gpu else None,
                    "circuit_open": self.is_circuit_open(ls),
                    "recovery_attempts": ls.recovery_attempts,
                }
                for n, ls in self.lanes.items()
            },
        }


# ---------------------------------------------------------------------------
# Model VRAM estimation (fallback only; real sizes come live from /api/ps size).
# ---------------------------------------------------------------------------

_SIZE_PATTERN = re.compile(r"(\d+)\s*[bm]", re.IGNORECASE)
# Bytes-per-parameter in GiB by quantization (1 param ~ 1 byte at 8-bit).
_QUANT_MAP = {
    "iq1": 0.15, "iq2": 0.22, "iq3": 0.32, "q2": 0.25, "q3": 0.4, "iq4": 0.45,
    "q4": 0.5, "q5": 0.62, "q6": 0.75, "q8": 1.0, "f16": 2.0, "f32": 4.0,
}


def estimate_vram_gib(model: str, known_bytes: int | None = None) -> float:
    """Estimate VRAM in GiB for a model name. Prefer known_bytes when available."""
    if known_bytes:
        return round(known_bytes / (1024**3), 2)
    m = _SIZE_PATTERN.findall(model)
    if not m:
        return 8.0
    params = max(int(x) for x in m)  # billions
    quant = 0.5  # default assume q4 (most local models) if unspecified
    low = model.lower()
    for token, factor in _QUANT_MAP.items():
        if token in low:
            quant = factor
            break
    return round(params * quant, 2)
