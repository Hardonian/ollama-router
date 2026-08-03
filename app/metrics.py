from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LaneMetric:
    requests: int = 0
    errors: int = 0
    total_ms: float = 0.0
    last_ms: float = 0.0
    last_error: str = ""

    @property
    def avg_ms(self) -> float:
        return round(self.total_ms / self.requests, 1) if self.requests else 0.0

    @property
    def error_rate(self) -> float:
        return round(self.errors / self.requests, 3) if self.requests else 0.0


class MetricsStore:
    """Per-model, per-lane latency/error tracking used by the optimizer to learn."""

    def __init__(self, path: str = "/home/scott/ai-lab/logs/ollama-router-metrics.json"):
        self.path = Path(path)
        self.lock = threading.Lock()
        # model -> lane -> LaneMetric
        self.data: dict[str, dict[str, LaneMetric]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                for model, lanes in raw.items():
                    self.data[model] = {
                        lane: LaneMetric(**m) for lane, m in lanes.items()
                    }
            except Exception:  # noqa: BLE001 — corrupt metrics file is non-fatal
                self.data = {}

    def _save(self) -> None:
        try:
            out = {
                model: {lane: vars(m) for lane, m in lanes.items()}
                for model, lanes in self.data.items()
            }
            self.path.write_text(json.dumps(out, indent=2))
        except Exception:  # noqa: BLE001
            pass

    def record(self, model: str, lane: str, ms: float, error: str = "") -> None:
        with self.lock:
            self.data.setdefault(model, {}).setdefault(lane, LaneMetric())
            m = self.data[model][lane]
            m.requests += 1
            m.total_ms += ms
            m.last_ms = round(ms, 1)
            if error:
                m.errors += 1
                m.last_error = error
            else:
                m.last_error = ""
            self._save()

    def best_lane(self, model: str) -> str | None:
        """Return the lane with the lowest avg latency and an acceptable error rate."""
        lanes = self.data.get(model)
        if not lanes:
            return None
        scored = [
            (m.avg_ms + m.error_rate * 5000, lane)
            for lane, m in lanes.items()
            if m.requests >= 3
        ]
        if not scored:
            return None
        scored.sort(key=lambda x: x[0])
        return scored[0][1]

    def best_lane_score(self, model: str, lane: str) -> float:
        """Lower is better; used to pick among equally-resident lanes."""
        m = self.data.get(model, {}).get(lane)
        if not m or m.requests == 0:
            return float("inf")
        return m.avg_ms + m.error_rate * 5000

    def summary(self) -> dict:
        return {
            model: {lane: vars(m) for lane, m in lanes.items()}
            for model, lanes in self.data.items()
        }
