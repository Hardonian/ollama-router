# Ollama GPU Router

> One endpoint for all your local models — routed to the right GPU, automatically.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Local-first](https://img.shields.io/badge/deployment-local--first-2ea043)]()
[![Bootstrap-ready](https://img.shields.io/badge/bootstrap-ready-2ea043)](BOOTSTRAP.md)

---

## The problem

You have multiple GPUs (V100 / P40 / RTX 3060) and several `ollama serve`
instances — one per card. Clients shouldn't have to know *which* GPU a model
lives on, and you don't want small models OOM-ing a big-GPU lane or large
models crashing a small one.

## The fix

A thin FastAPI proxy in front of your Ollama lanes. Clients hit **one** URL;
the router picks the correct GPU by VRAM requirement and proxies transparently,
tagging the response with `X-GPU-Routed`.

## GPU lanes

| GPU  | Port  | VRAM | Used for |
|------|-------|------|----------|
| 3060 | 11437 | 12GB | Vision/VL models + small models (≤12B) |
| P40  | 11436 | 24GB | Mid models (16–24B) |
| V100 | 11435 | 16GB | Large models (≥32B) |

Routing rules (in `app/main.py`):

1. Model name contains `vl` / `vision` → 3060 (vision lane).
2. Model needs ≥ 32GB VRAM → V100.
3. Model needs ≥ 16GB VRAM → P40.
4. Otherwise → 3060.

Model size is looked up from `MODEL_SIZES`, or inferred from a `Nb` pattern in
the name (default 8B).

## Endpoints

- `GET /` — health string.
- `GET /route-info?model=<name>` — returns `{model, vram_needed_gb, target_port, target_gpu}`.
- `/*` — transparent proxy to the selected lane; responds with `X-GPU-Routed: <port>`.

## Quick start

```bash
git clone https://github.com/Hardonian/ollama-router.git
cd ollama-router
pip install fastapi uvicorn httpx
uvicorn app.main:app --host 127.0.0.1 --port 11438
```

Point your Ollama clients at this router's port instead of the raw GPU ports.

## Notes

- Each lane must be a running `ollama serve` instance on its assigned port.
- The router is **local-first** — no external calls, no telemetry.

## Part of the Hardonia stack

Ollama Router is one of the [Hardonia](https://github.com/Hardonian)
local-first AI infrastructure projects: measurable value, operator-grade
control, and zero theatre.
