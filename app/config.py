from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.toml")


@dataclass
class LaneConfig:
    name: str
    port: int
    cuda_visible_devices: str = "0"
    systemd_unit: str = ""
    role: str = "general"  # large | medium | vision | compat

    @property
    def physical_gpu(self) -> int:
        """First physical GPU index this lane is pinned to (1:1 mapping here)."""
        digits = "".join(ch for ch in self.cuda_visible_devices if ch.isdigit())
        return int(digits.split(",")[0]) if digits else 0


@dataclass
class RouterConfig:
    host: str = "127.0.0.1"
    port: int = 11438
    poll_interval: float = 5.0
    request_timeout: float = 120.0
    connect_timeout: float = 10.0
    recovery_cooldown_sec: float = 600.0
    recovery_max_attempts: int = 3
    circuit_breaker_threshold: int = 3
    circuit_breaker_cooldown_sec: float = 120.0
    vram_margin_gb: float = 0.5
    hot_models: list[str] = field(default_factory=list)
    lanes: list[LaneConfig] = field(default_factory=list)

    @classmethod
    def load(cls, path: str = DEFAULT_CONFIG_PATH) -> RouterConfig:
        data: dict[str, Any] = {}
        if os.path.exists(path):
            with open(path, "rb") as fh:
                data = tomllib.load(fh)
        router = data.get("router", {})
        lanes = [LaneConfig(**lane) for lane in data.get("lanes", [])]
        return cls(
            host=router.get("host", cls.host),
            port=int(router.get("port", cls.port)),
            poll_interval=float(router.get("poll_interval", cls.poll_interval)),
            request_timeout=float(router.get("request_timeout", cls.request_timeout)),
            connect_timeout=float(router.get("connect_timeout", cls.connect_timeout)),
            recovery_cooldown_sec=float(router.get("recovery_cooldown_sec", cls.recovery_cooldown_sec)),
            recovery_max_attempts=int(router.get("recovery_max_attempts", cls.recovery_max_attempts)),
            circuit_breaker_threshold=int(router.get("circuit_breaker_threshold", cls.circuit_breaker_threshold)),
            circuit_breaker_cooldown_sec=float(router.get("circuit_breaker_cooldown_sec", cls.circuit_breaker_cooldown_sec)),
            vram_margin_gb=float(router.get("vram_margin_gb", cls.vram_margin_gb)),
            hot_models=list(router.get("hot_models", [])),
            lanes=lanes or _default_lanes(),
        )


def _default_lanes() -> list[LaneConfig]:
    return [
        LaneConfig(name="v100", port=11437, cuda_visible_devices="0", systemd_unit="ollama-v100.service", role="large"),
        LaneConfig(name="p40", port=11435, cuda_visible_devices="1", systemd_unit="ollama-p40.service", role="medium"),
        LaneConfig(name="3060", port=11436, cuda_visible_devices="2", systemd_unit="ollama-3060.service", role="vision"),
        LaneConfig(name="default", port=11434, cuda_visible_devices="0", systemd_unit="ollama-default.service", role="compat"),
    ]
