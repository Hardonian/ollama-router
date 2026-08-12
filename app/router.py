from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Iterable

from app.config import RouterConfig
from app.metrics import MetricsStore
from app.state import ClusterState, LaneState, estimate_vram_gib

logger = logging.getLogger("ollama-router")


class Router:
    """Self-correcting, self-optimizing load-splitter across Ollama GPU lanes.

    Decision order for a model request:
      1. Prefer a lane where the model is already resident (warm affinity).
      2. Prefer the lane with the best learned latency (MetricsStore).
      3. Prefer smallest GPU that *currently* has enough free VRAM (live introspection).
      4. Fall back to any healthy lane that fits, else any healthy lane (circuit-breaker open lanes excluded).
    On proxy failure: mark failure, try next-best lane (failover), open circuit breaker after N strikes.
    Dead lanes are auto-recovered (systemd restart) under a cooldown + attempt cap.
    """

    def __init__(self, cfg: RouterConfig, state: ClusterState, metrics: MetricsStore):
        self.cfg = cfg
        self.state = state
        self.metrics = metrics

    # -- selection ----------------------------------------------------------
    def select(self, model: str) -> LaneState | None:
        now = time.time()
        healthy = [ls for ls in self.state.lanes.values() if ls.healthy and not self.state.is_circuit_open(ls, now)]
        # Multiple listeners can share one physical GPU (the localhost compat
        # lane shares GPU0 with v100).  The router must never treat those as
        # independent VRAM pools; prefer the purpose-built lane and reserve
        # compat only for direct legacy callers on :11434.
        routed: dict[int | None, LaneState] = {}
        for lane in healthy:
            gpu = lane.physical_gpu
            existing = routed.get(gpu)
            if existing is None or (existing.cfg.role == "compat" and lane.cfg.role != "compat"):
                routed[gpu] = lane
        healthy = list(routed.values())
        if not healthy:
            return None

        vram = self._vram_for(model)

        # 1) warm affinity
        # A resident model is normally the cheapest choice (no reload), but a
        # badly-spilled one is not: a 30B MoE at 81% GPU measured 0.43 tok/s on
        # the V100 versus 11.9 tok/s fully resident on the P40 (~28x). So only
        # honour warm affinity when the model is mostly on the GPU, or when no
        # other lane could hold it any better.
        resident = [ls for ls in healthy if model in ls.loaded]
        good_resident = [ls for ls in resident if self.state.gpu_fraction(ls, model) >= self.cfg.min_gpu_fraction]
        if good_resident:
            return min(good_resident, key=lambda ls: self.metrics.best_lane_score(model, ls.cfg.name))
        if resident and not any(self._fits(ls, vram) for ls in healthy if ls not in resident):
            # Spilled, but nowhere better to go -- keep it where it is.
            return min(resident, key=lambda ls: self.metrics.best_lane_score(model, ls.cfg.name))

        # 2) learned best lane, if it fits there
        best = self.metrics.best_lane(model)
        if best and best in self.state.lanes:
            ls = self.state.lanes[best]
            if ls in healthy and self._fits(ls, vram):
                return ls

        # 3) smallest GPU that fits with enough FREE VRAM right now
        fitting = [ls for ls in healthy if self._fits(ls, vram)]
        if fitting:
            return min(fitting, key=lambda ls: ls.gpu.free_gib if ls.gpu else 0)

        # 3b) nothing fits now, but a lane can hold it once Ollama evicts its
        #     idle resident models. Pick the smallest such lane (most GPU
        #     headroom retained elsewhere). This is what keeps a 30B MoE off a
        #     16G V100 (crawl) when a 23G P40 has an idle model squatting it.
        fits_after_evict = [
            ls for ls in healthy
            if (ls.gpu.free_gib + self.state.reclaimable_gib(ls, model) - self.cfg.vram_margin_gb) >= vram
        ]
        if fits_after_evict:
            return min(fits_after_evict, key=lambda ls: ls.gpu.free_gib if ls.gpu else 0)

        # 4) Fits nowhere: this model will be partially offloaded to CPU no
        #    matter what, so pick the lane with the MOST total VRAM. Ollama
        #    offloads the remainder to RAM, so the biggest GPU keeps the most
        #    layers on-die and runs fastest. Picking the smallest GPU here
        #    maximised CPU offload and made big MoE models crawl.
        return max(healthy, key=lambda ls: ls.gpu.total_gib if ls.gpu else 0)

    def _vram_for(self, model: str) -> float:
        # Best -> worst signal:
        #  1. resident size reported by /api/ps (ground truth, model is loaded)
        #  2. on-disk size from /api/tags (real bytes; correct for MoE models
        #     whose names imply far fewer weights than they actually ship)
        #  3. name-based heuristic (last resort)
        for ls in self.state.lanes.values():
            if model in ls.loaded:
                return estimate_vram_gib(model, ls.loaded[model])
        for ls in self.state.lanes.values():
            size = ls.catalog.get(model)
            if size:
                # Weights must fit alongside KV cache + runtime overhead.
                return round(estimate_vram_gib(model, size) * 1.15, 2)
        return estimate_vram_gib(model)

    def _fits(self, ls: LaneState, vram_gib: float) -> bool:
        # A lane "fits" if either it has enough free VRAM now, OR it has enough
        # once Ollama evicts idle resident models. Ollama proactively evicts
        # least-recently-used models to load a new one, so a lane whose "free"
        # VRAM looks too small only because an idle model is squatting it can
        # still take the new model. Without this, a 30B MoE was wrongly sent to
        # a smaller GPU while the biggest lane sat idle (measured 3x speed loss:
        # 1.4 tok/s on V100 vs 4+ tok/s on P40).
        if ls.fits(vram_gib, self.cfg.vram_margin_gb):
            return True
        free_plus_reclaim = ls.gpu.free_gib + self.state.reclaimable_gib(ls, "")
        return (free_plus_reclaim - self.cfg.vram_margin_gb) >= vram_gib

    # -- failure / recovery ------------------------------------------------
    def on_failure(self, ls: LaneState, error: str) -> None:
        self.state.mark_failure(ls)
        logger.warning("lane %s failure #%d: %s", ls.cfg.name, ls.failures, error)
        if self.state.is_circuit_open(ls):
            self._try_recover(ls)

    def on_success(self, ls: LaneState) -> None:
        self.state.mark_success(ls)

    def _try_recover(self, ls: LaneState) -> None:
        now = time.time()
        if not ls.cfg.systemd_unit:
            return
        if now - ls.last_recovery < self.cfg.recovery_cooldown_sec:
            return
        if ls.recovery_attempts >= self.cfg.recovery_max_attempts:
            logger.error("lane %s exceeded recovery attempts; giving up", ls.cfg.name)
            return
        ls.last_recovery = now
        ls.recovery_attempts += 1
        logger.info("auto-recovering lane %s (%s) attempt %d", ls.cfg.name, ls.cfg.systemd_unit, ls.recovery_attempts)
        try:
            subprocess.run(
                ["systemctl", "--user", "restart", ls.cfg.systemd_unit],
                check=True, timeout=60,
                capture_output=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("recovery failed for %s: %s", ls.cfg.name, e)

    def eligible_lanes(self) -> Iterable[LaneState]:
        now = time.time()
        return [ls for ls in self.state.lanes.values() if ls.healthy and not self.state.is_circuit_open(ls, now)]
