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


def test_router_excludes_compat_lane_that_shares_v100_gpu():
    cfg = RouterConfig(lanes=[])
    cfg.lanes = [
        _lane_cfg("v100", 11437, "0", "large"),
        _lane_cfg("default", 11434, "0", "compat"),
        _lane_cfg("p40", 11435, "1", "medium"),
    ]
    state = ClusterState.__new__(ClusterState)
    state.lanes = {lane.cfg.name: lane for lane in cfg.lanes}
    state.gpus = {0: GpuInfo(0, "V100", 16384, 0, 16384), 1: GpuInfo(1, "P40", 23040, 0, 23040)}
    for lane in state.lanes.values():
        lane.healthy = True
        lane.gpu = state.gpus[lane.physical_gpu]
    metrics = MetricsStore("/tmp/test-metrics-compat.json")
    metrics.record("llama3.1:8b", "default", 1.0)
    selected = Router(cfg, state, metrics).select("llama3.1:8b")
    assert selected is not None
    assert selected.cfg.name != "default"


def test_metrics_learning():
    m = MetricsStore("/tmp/test-metrics3.json")
    m.record("llama3.1:8b", "3060", 120.0)
    m.record("llama3.1:8b", "3060", 130.0)
    m.record("llama3.1:8b", "3060", 110.0)
    m.record("llama3.1:8b", "v100", 300.0)
    m.record("llama3.1:8b", "v100", 320.0)
    m.record("llama3.1:8b", "v100", 290.0)
    assert m.best_lane("llama3.1:8b") == "3060"


def test_moe_model_sized_from_catalog_not_name():
    """A 30B-A3B MoE ships ~25GB of weights but its name implies ~15GiB.

    Routing off the name alone sends it to a GPU that cannot hold it, so the
    on-disk size reported by /api/tags must win over the name heuristic.
    """
    cfg = RouterConfig(lanes=[])
    cfg.lanes = [
        _lane_cfg("v100", 11437, "0", "large"),
        _lane_cfg("p40", 11435, "1", "medium"),
    ]
    state = ClusterState.__new__(ClusterState)
    state.lanes = {lane.cfg.name: lane for lane in cfg.lanes}
    # V100 has 15.7GiB free, P40 has 22.4GiB free. Only the P40 can hold it.
    state.gpus = {0: GpuInfo(0, "V100", 16384, 300, 16084), 1: GpuInfo(1, "P40", 23040, 100, 22940)}
    model = "nemotron-3.5-lightning:30b-a3b-q4_K_M"
    for lane in state.lanes.values():
        lane.healthy = True
        lane.gpu = state.gpus[lane.physical_gpu]
        lane.models = {model}
        lane.catalog = {model: 25 * 1000**3}  # 25GB on disk, as ollama reports it
    metrics = MetricsStore("/tmp/test-metrics-moe.json")
    router = Router(cfg, state, metrics)

    # The name heuristic alone would claim this fits in 15GiB and pick the V100.
    assert estimate_vram_gib(model) < 16.0
    assert router._vram_for(model) > 16.0
    sel = router.select(model)
    assert sel is not None and sel.cfg.name == "p40", sel.cfg.name if sel else None


def _moe_cluster(model, v100_free_mib, p40_free_mib):
    cfg = RouterConfig(lanes=[])
    cfg.lanes = [_lane_cfg("v100", 11437, "0", "large"), _lane_cfg("p40", 11435, "1", "medium")]
    state = ClusterState.__new__(ClusterState)
    state.lanes = {lane.cfg.name: lane for lane in cfg.lanes}
    state.gpus = {
        0: GpuInfo(0, "V100", 16384, 16384 - v100_free_mib, v100_free_mib),
        1: GpuInfo(1, "P40", 23040, 23040 - p40_free_mib, p40_free_mib),
    }
    for lane in state.lanes.values():
        lane.healthy = True
        lane.gpu = state.gpus[lane.physical_gpu]
        lane.models = {model}
        lane.catalog = {model: 17_500_000_000}
    return cfg, state


def test_spilled_resident_model_is_relocated_to_a_lane_that_fits():
    """Measured: muse-glimmer at 81% GPU on the V100 ran 0.43 tok/s; fully
    resident on the P40 it ran 11.9 tok/s. Warm affinity must not pin a
    badly-spilled model in place when a lane that fits it is free."""
    model = "muse-glimmer:30b-q4_K_M"
    cfg, state = _moe_cluster(model, v100_free_mib=1000, p40_free_mib=23034)
    # Resident on the V100 but only 81% of its bytes are in VRAM.
    state.lanes["v100"].loaded = {model: 17_500_000_000}
    state.lanes["v100"].loaded_vram = {model: 14_230_000_000}
    router = Router(cfg, state, MetricsStore("/tmp/test-metrics-spill.json"))

    assert state.gpu_fraction(state.lanes["v100"], model) < 0.95
    sel = router.select(model)
    assert sel is not None and sel.cfg.name == "p40", sel.cfg.name if sel else None


def test_fully_resident_model_keeps_warm_affinity():
    """The inverse: a model that IS fully on the GPU must not be bounced."""
    model = "muse-glimmer:30b-q4_K_M"
    cfg, state = _moe_cluster(model, v100_free_mib=16084, p40_free_mib=6000)
    state.lanes["p40"].loaded = {model: 16_570_000_000}
    state.lanes["p40"].loaded_vram = {model: 16_570_000_000}  # 100% on GPU
    router = Router(cfg, state, MetricsStore("/tmp/test-metrics-warm.json"))

    assert state.gpu_fraction(state.lanes["p40"], model) == 1.0
    sel = router.select(model)
    assert sel is not None and sel.cfg.name == "p40", sel.cfg.name if sel else None


def test_spilled_resident_stays_put_when_nowhere_better():
    """If every lane is too small, reloading elsewhere buys nothing -- keep it."""
    model = "muse-glimmer:30b-q4_K_M"
    cfg, state = _moe_cluster(model, v100_free_mib=1000, p40_free_mib=1000)
    state.lanes["v100"].loaded = {model: 17_500_000_000}
    state.lanes["v100"].loaded_vram = {model: 14_230_000_000}
    router = Router(cfg, state, MetricsStore("/tmp/test-metrics-stay.json"))

    sel = router.select(model)
    assert sel is not None and sel.cfg.name == "v100", sel.cfg.name if sel else None


def test_idle_resident_model_vram_is_reclaimable():
    """Regression for the 3x speed-loss bug: an idle (keep_alive-counting-down)
    model squatting the P40's VRAM must NOT stop a new 30B MoE from routing
    there. Ollama evicts idle models on demand, so their VRAM is reclaimable.

    Measured: without this, nemotron landed on the V100 at 1.38 tok/s (54% GPU)
    while the P40 sat idle with a stale muse-glimmer resident -> 3x loss.
    """
    from datetime import datetime, timedelta
    import time as _time

    model = "nemotron-3.5-lightning:30b-a3b-q4_K_M"
    other = "muse-glimmer:30b-q4_K_M"
    cfg = RouterConfig(lanes=[])
    cfg.lanes = [_lane_cfg("v100", 11437, "0", "large"), _lane_cfg("p40", 11435, "1", "medium")]
    state = ClusterState.__new__(ClusterState)
    state.lanes = {lane.cfg.name: lane for lane in cfg.lanes}
    state.gpus = {
        0: GpuInfo(0, "V100", 16384, 16384 - 16000, 384),
        1: GpuInfo(1, "P40", 23040, 16200, 6840),  # only 6.7G "free" because idle model squats it
    }
    now = _time.time()
    for lane in state.lanes.values():
        lane.healthy = True
        lane.gpu = state.gpus[lane.physical_gpu]
    # muse-glimmer idle-resident on the P40, expiring SOON (reclaimable).
    p40 = state.lanes["p40"]
    p40.models = {other}
    p40.catalog = {other: 17_500_000_000, model: 25_000_000_000}
    p40.loaded = {other: 17_500_000_000}
    p40.loaded_vram = {other: 16_570_000_000}
    p40.loaded_expires = {other: datetime.fromtimestamp(now + 30).timestamp()}  # non-zero => reclaimable
    router = Router(cfg, state, MetricsStore("/tmp/test-metrics-idle.json"))

    # Without reclaim logic, _fits(p40) would be False (only 6.7G free < 28.7G needed).
    assert not p40.fits(28.7, cfg.vram_margin_gb), "precondition: naive fit must fail"
    # But the idle model's VRAM is reclaimable (non-pinned), so route to p40.
    sel = router.select(model)
    assert sel is not None and sel.cfg.name == "p40", sel.cfg.name if sel else None


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
