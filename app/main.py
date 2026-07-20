import re

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

app = FastAPI(title="Ollama GPU Router")

# Model size mapping (approximate VRAM in GB)
MODEL_SIZES = {
    "granite4.1:3b": 3,
    "llama3.1:8b": 8,
    "qwen2.5-coder:7b": 7,
    "dolphin3:latest": 8,
    "glm-4.7-flash:latest": 17,
    "qwen3-vl:latest": 8,
    "baytout3/Qwen3.6-27B-Uncensored-HauhauCS-Balanced:IQ4_XS": 27,
    "qwen3:32b": 32,
    "deepseek-r1:32b": 32,
    "tinyrick/gemma-4-31B-it-uncensored-heretic-vision-llmfan46:Q4_K_M": 31,
    "mistral-small3.2:latest": 24,
}

# GPU lane mappings by VRAM capacity
# Ports MUST match the live systemd lane bindings
# (ollama-v100.service=>11437, ollama-p40.service=>11435, ollama-3060.service=>11436).
# Router targets 127.0.0.1:<port>; lanes are localhost-only (no LAN exposure).
GPU_LANES = {
    "v100": {"port": 11437, "vram_gb": 16, "compute": "7.0"},
    "p40": {"port": 11435, "vram_gb": 24, "compute": "6.1"},
    "3060": {"port": 11436, "vram_gb": 12, "compute": "8.6"},
}


def get_model_size(model_name: str) -> int:
    if model_name in MODEL_SIZES:
        return MODEL_SIZES[model_name]
    match = re.search(r"(\d+)b", model_name.lower())
    return int(match.group(1)) if match else 8


def route_model(model_name: str) -> int:
    vram_needed = get_model_size(model_name)
    if "vl" in model_name.lower() or "vision" in model_name.lower():
        return GPU_LANES["3060"]["port"]
    if vram_needed >= 32:
        return GPU_LANES["v100"]["port"]
    if vram_needed >= 16:
        return GPU_LANES["p40"]["port"]
    return GPU_LANES["3060"]["port"]


@app.get("/")
async def root():
    return PlainTextResponse("Ollama GPU Router running")


@app.get("/route-info")
async def route_info(model: str):
    return {
        "model": model,
        "vram_needed_gb": get_model_size(model),
        "target_port": route_model(model),
        "target_gpu": [k for k, v in GPU_LANES.items() if v["port"] == route_model(model)][0]
    }


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(request: Request, path: str):
    model_name = request.query_params.get("model") or ""
    if not model_name and request.method == "POST":
        try:
            body = await request.json()
            model_name = body.get("model", "")
        except ValueError:
            pass
    
    target_port = route_model(model_name) if model_name else GPU_LANES["3060"]["port"]
    target_url = f"http://127.0.0.1:{target_port}/{path}"
    
    async with httpx.AsyncClient() as client:
        try:
            if request.method == "GET":
                resp = await client.get(target_url, params=request.query_params)
            else:
                body = await request.body()
                resp = await client.request(
                    request.method,
                    target_url,
                    content=body,
                    headers={k: v for k, v in request.headers.items() if k != "host"}
                )
            
            try:
                content = resp.json()
            except ValueError:
                content = resp.text
            
            return JSONResponse(
                content=content,
                status_code=resp.status_code,
                headers={"X-GPU-Routed": str(target_port)}
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail=f"GPU lane {target_port} unavailable")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))