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
        if not healthy:
            return None

        vram = self._vram_for(model)

        # 1) warm affinity
        resident = [ls for ls in healthy if model in ls.loaded]
        if resident:
            return min(resident, key=lambda ls: self.metrics.best_lane_score(model, ls.cfg.name))

        # 2) learned best lane, if it fits there
        best = self.metrics.best_lane(model)
        if best and best in self.state.lanes:
            ls = self.state.lanes[best]
            if ls in healthy and self._fits(ls, vram):
                return ls

        # 3) smallest GPU with enough free VRAM right now
        fitting = [ls for ls in healthy if self._fits(ls, vram)]
        if fitting:
            return min(fitting, key=lambda ls: ls.gpu.free_gib if ls.gpu else 0)

        # 4) any healthy lane (overcommit — let Ollama decide; failover handles OOM)
        return min(healthy, key=lambda ls: ls.gpu.free_gib if ls.gpu else 0)

    def _vram_for(self, model: str) -> float:
        # If we already know the resident size from a lane, use it.
        for ls in self.state.lanes.values():
            if model in ls.loaded:
                return estimate_vram_gib(model, ls.loaded[model])
        return estimate_vram_gib(model)

    def _fits(self, ls: LaneState, vram_gib: float) -> bool:
        return ls.fits(vram_gib, self.cfg.vram_margin_gb)

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
