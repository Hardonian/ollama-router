from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import RouterConfig
from app.metrics import MetricsStore
from app.router import Router
from app.state import ClusterState, GpuInfo, LaneState, estimate_vram_gib


def _lane(name, port, cuda, role):
    from app.config import LaneConfig
    return LaneState(cfg=LaneConfig(name=name, port=port, cuda_visible_devices=cuda, role=role))


def test_estimate_vram():
    assert estimate_vram_gib("llama3.1:8b") > 0
    assert estimate_vram_gib("qwen3:32b", known_bytes=20 * 1024**3) == 20.0
    # 32B q4 -> ~ 3.4*32 = ~109GiB? no: q4 ~3.4 factor -> 108? sanity: bigger than 8b
    assert estimate_vram_gib("qwen3:32b") > estimate_vram_gib("llama3.1:8b")


def test_lane_fits_uses_live_free():
    ls = _lane("v100", 11437, "0", "large")
    ls.healthy = True
    ls.gpu = GpuInfo(index=0, name="V100", total_mib=16384, used_mib=0, free_mib=16384)
    assert ls.fits(15.0, 0.5) is True
    assert ls.fits(16.5, 0.5) is False


def test_router_prefers_warm_lane():
    cfg = RouterConfig(lanes=[])
    cfg.lanes = [
        _lane_cfg("v100", 11437, "0", "large"),
        _lane_cfg("p40", 11435, "1", "medium"),
        _lane_cfg("3060", 11436, "2", "vision"),
        _lane_cfg("default", 11434, "0", "compat"),
    ]
    state = ClusterState.__new__(ClusterState)
    state.lanes = {lane.cfg.name: lane for lane in cfg.lanes}
    state.gpus = {0: GpuInfo(0, "V100", 16384, 0, 16384), 1: GpuInfo(1, "P40", 23040, 0, 23040), 2: GpuInfo(2, "3060", 12288, 0, 12288)}
    for lane in state.lanes.values():
        lane.healthy = True
        lane.gpu = state.gpus[lane.physical_gpu]
    metrics = MetricsStore("/tmp/test-metrics.json")
    router = Router(cfg, state, metrics)
    # Put model resident on p40 -> warm affinity wins over smallest-fit.
    state.lanes["p40"].loaded = {"qwen3:32b": 20 * 1024**3}
    sel = router.select("qwen3:32b")
    assert sel is not None and sel.cfg.name == "p40", sel.cfg.name if sel else None


def test_router_smallest_fit():
    cfg = RouterConfig(lanes=[])
    cfg.lanes = [
        _lane_cfg("v100", 11437, "0", "large"),
        _lane_cfg("p40", 11435, "1", "medium"),
        _lane_cfg("3060", 11436, "2", "vision"),
        _lane_cfg("default", 11434, "0", "compat"),
    ]
    state = ClusterState.__new__(ClusterState)
    state.lanes = {lane.cfg.name: lane for lane in cfg.lanes}
    state.gpus = {0: GpuInfo(0, "V100", 16384, 16000, 384), 1: GpuInfo(1, "P40", 23040, 0, 23040), 2: GpuInfo(2, "3060", 12288, 0, 12288)}
    for lane in state.lanes.values():
        lane.healthy = True
        lane.gpu = state.gpus[lane.physical_gpu]
    metrics = MetricsStore("/tmp/test-metrics2.json")
    router = Router(cfg, state, metrics)
    # V100 nearly full -> 32B should NOT go to V100 even though role=large
    sel = router.select("qwen3:32b")
    assert sel is not None and sel.cfg.name in ("p40",), sel.cfg.name


def test_metrics_learning():
    m = MetricsStore("/tmp/test-metrics3.json")
    m.record("llama3.1:8b", "3060", 120.0)
    m.record("llama3.1:8b", "3060", 130.0)
    m.record("llama3.1:8b", "3060", 110.0)
    m.record("llama3.1:8b", "v100", 300.0)
    m.record("llama3.1:8b", "v100", 320.0)
    m.record("llama3.1:8b", "v100", 290.0)
    assert m.best_lane("llama3.1:8b") == "3060"


from app.config import LaneConfig


def _lane_cfg(name, port, cuda, role):
    return LaneState(cfg=LaneConfig(name=name, port=port, cuda_visible_devices=cuda, role=role))


# ── Persistence & Autoscale Tests ────────────────────────────────────────────

def test_metrics_persistence():
    import os
    import tempfile
    from pathlib import Path

    from app.metrics_persist import get_lane_stats, get_state, init_db, record_latency, set_state

    # Use temp file for isolation
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    try:
        init_db(Path(db_path))
        record_latency('test-lane', 'test-model', 100.0, True, Path(db_path))
        record_latency('test-lane', 'test-model', 200.0, False, Path(db_path))
        stats = get_lane_stats('test-lane', 'test-model', db_path=Path(db_path))
        assert stats['samples'] == 2
        assert stats['success_rate'] == 0.5
        assert abs(stats['avg_latency_ms'] - 150.0) < 0.01

        set_state('test_key', {'value': 42}, Path(db_path))
        state = get_state('test_key', db_path=Path(db_path))
        assert state == {'value': 42}
    finally:
        os.unlink(db_path)


def test_autoscale_capacity_pressure():
    from app.autoscale import check_capacity_pressure
    from app.state import ClusterState, GpuInfo

    state = ClusterState.__new__(ClusterState)
    state.gpus = {
        0: GpuInfo(0, 'V100', 16384, 15384, 1000),   # 94% used
        1: GpuInfo(1, 'P40', 23040, 100, 22940),      # 0.4% used
        2: GpuInfo(2, '3060', 12288, 11000, 1288),    # 90% used
    }
    pressured = check_capacity_pressure(state, threshold=0.85)
    assert len(pressured) == 2
    gpu_indices = {p['gpu'] for p in pressured}
    assert gpu_indices == {0, 2}

    # Test with a higher threshold that still includes the 93.9%-used V100
    pressured = check_capacity_pressure(state, threshold=0.93)
    assert len(pressured) == 1
    assert pressured[0]['gpu'] == 0
