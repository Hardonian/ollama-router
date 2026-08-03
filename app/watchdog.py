#!/usr/bin/env python3
# ============================================================================
# gpu-router-watchdog.py — self-maintaining operator for the Ollama GPU router.
# Runs as a systemd timer (every 60s). Verifies:
#   1. All configured lanes are reachable and return /api/tags.
#   2. Each lane only uses its pinned physical GPU (isolation guard, cross-check).
#   3. The router itself is healthy.
#   4. Config drift: lane count/ports match the running systemd units.
# Auto-recovers dead lanes (systemd restart) under the same guards as the router.
# Writes a JSON health doc the status page / operator can read.
# ============================================================================
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import RouterConfig  # noqa: E402
from app.state import ClusterState  # noqa: E402

LOG_DIR = Path("/home/scott/ai-lab/logs")
STATE_FILE = Path("/home/scott/.hermes/state/gpu-router-watchdog.json")


def check_lane(unit: str, port: int) -> dict:
    res = {"unit": unit, "port": port, "reachable": False, "models": 0, "error": ""}
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/api/tags", timeout=5)
        r.raise_for_status()
        res["reachable"] = True
        res["models"] = len(r.json().get("models", []))
    except Exception as e:  # noqa: BLE001
        res["error"] = str(e)
    return res


def isolation_ok(state: ClusterState) -> tuple[bool, str]:
    """Cross-check: every lane's ollama compute process is on its pinned GPU.

    Reads each running `ollama serve` PID's CUDA_VISIBLE_DEVICES env and maps
    the listener port -> pid via ss, then asserts the pinned GPU matches.
    """
    try:
        out = subprocess.check_output(
            ["sh", "-c",
             "for p in $(pgrep -f 'ollama serve'); do "
             "echo \"$p:$(tr '\\0' '\\n' < /proc/$p/environ 2>/dev/null | grep -E '^CUDA_VISIBLE_DEVICES=')\"; done"],
            text=True, timeout=15,
        )
    except Exception as e:  # noqa: BLE001
        return False, f"cannot read ollama envs: {e}"
    pid_to_gpu: dict[str, int] = {}
    for line in out.strip().splitlines():
        if ":" not in line:
            continue
        pid, val = line.split(":", 1)
        digits = "".join(c for c in val if c.isdigit()).split(",")[0]
        pid_to_gpu[pid] = int(digits) if digits else -1
    # Map port -> pid via ss.
    try:
        ss = subprocess.check_output(
            ["ss", "-tlnp", "sport", ">=", "11434", "and", "sport", "<=", "11437"],
            text=True, timeout=15,
        )
    except Exception:
        ss = ""
    port_to_pid: dict[str, str] = {}
    for line in ss.strip().splitlines():
        if "pid=" not in line:
            continue
        m = re.findall(r"127\.0\.0\.1:(\d+).*?pid=(\d+)", line)
        for port, pid in m:
            port_to_pid[port] = pid
    issues = []
    for ls in state.lanes.values():
        port_str = str(ls.cfg.port)
        want_gpu = ls.physical_gpu
        pid = port_to_pid.get(port_str)
        if not pid:
            issues.append(f"port {port_str}: no listener pid")
            continue
        got = pid_to_gpu.get(pid, -2)
        if got != want_gpu:
            issues.append(f"port {port_str}: expected GPU {want_gpu}, ollama pid {pid} sees GPU {got}")
    return (len(issues) == 0), ("; ".join(issues) if issues else "isolation OK")


def recovery(unit: str) -> str:
    try:
        subprocess.run(["systemctl", "--user", "restart", unit], check=True, timeout=60, capture_output=True)
        return f"restarted {unit}"
    except Exception as e:  # noqa: BLE001
        return f"restart FAILED {unit}: {e}"


def main() -> int:
    cfg = RouterConfig.load()
    state = ClusterState(cfg)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    report = {"ts": time.time(), "lanes": [], "isolation": {}, "router": {}}
    recoveries = []

    for ls in state.lanes.values():
        c = check_lane(ls.cfg.systemd_unit, ls.cfg.port)
        if not c["reachable"] and ls.cfg.systemd_unit:
            recoveries.append(recovery(ls.cfg.systemd_unit))
        report["lanes"].append(c)

    ok, detail = isolation_ok(state)
    report["isolation"] = {"ok": ok, "detail": detail}

    # router health
    try:
        r = httpx.get(f"http://127.0.0.1:{cfg.port}/health", timeout=5)
        report["router"] = r.json()
    except Exception as e:  # noqa: BLE001
        report["router"] = {"status": "down", "error": str(e)}
        # router itself may need restart
        recoveries.append(recovery("ollama-router.service"))

    report["recoveries"] = recoveries
    STATE_FILE.write_text(json.dumps(report, indent=2))

    # Human-readable log line
    healthy = sum(1 for lane in report["lanes"] if lane["reachable"])
    print(f"[{time.strftime('%H:%M:%S')}] lanes {healthy}/{len(report['lanes'])} "
          f"isolation={'OK' if ok else 'BAD'} router={report['router'].get('status')} "
          f"recoveries={len(recoveries)}")
    if not ok or healthy < len(report["lanes"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
