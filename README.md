# Ollama GPU Router

A local FastAPI proxy that routes Ollama model requests to the correct GPU lane
based on model VRAM requirements. Sits in front of multiple `ollama serve`
instances (one per GPU) and forwards requests transparently.

<img src="/home/scott/ai-workspace/assets/repo-previews/ollama-router.png" alt="Ollama Router" width="800"/>

## Why

Running several Ollama instances on different GPUs (V100 / P40 / RTX 3060) lets
you serve small, medium, and large models without OOM. This router decides which
GPU a model belongs on and proxies the call — so clients hit one endpoint.

## GPU lanes

| GPU  | Port  | VRAM | Used for |
|------|-------|------|----------|
| 3060 | 11437 | 12GB | Vision/VL models + small models (<=12B) |
| P40  | 11436 | 24GB | Mid models (16–24B) |
| V100 | 11435 | 16GB | Large models (>=32B) |

Routing rules (in `app/main.py`):
1. Model name contains `vl` / `vision` → 3060 (vision lane).
2. Model needs >= 32GB VRAM → V100.
3. Model needs >= 16GB VRAM → P40.
4. Otherwise → 3060.

Model size is looked up from `MODEL_SIZES`, or inferred from a `Nb` pattern
in the name (default 8B).

## Endpoints

- `GET /` — health string.
- `GET /route-info?model=<name>` — returns `{model, vram_needed_gb, target_port, target_gpu}`.
- `/*` — transparent proxy to the selected lane; responds with
  `X-GPU-Routed: <port>` header.

## Run

```bash
pip install fastapi uvicorn httpx
uvicorn app.main:app --host 0.0.0.0 --port 11438
```

Point your Ollama clients at this router's port instead of the raw GPU ports.

## Notes

- Each lane must be a running `ollama serve` instance on its assigned port.
- VRAM map and lane ports live in `GPU_LANES` / `MODEL_SIZES` at the top of
  `app/main.py` — edit there to match your hardware.
