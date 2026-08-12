from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from app.config import RouterConfig
from app.metrics import MetricsStore
from app.router import Router
from app.state import ClusterState

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("ollama-router")

cfg = RouterConfig.load()
state = ClusterState(cfg)
metrics = MetricsStore()
router = Router(cfg, state, metrics)

# Background refresh loop.
_stop = asyncio.Event()


async def _refresh_loop() -> None:
    while not _stop.is_set():
        try:
            await asyncio.to_thread(state.refresh)
        except Exception as e:  # noqa: BLE001
            log.warning("state refresh error: %s", e)
        await asyncio.sleep(cfg.poll_interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    task = loop.create_task(_refresh_loop())
    log.info("Ollama GPU Router started on :%d (%d lanes)", cfg.port, len(state.lanes))
    try:
        yield
    finally:
        _stop.set()
        task.cancel()


app = FastAPI(title="Ollama GPU Router — self-optimizing", lifespan=lifespan)


async def _model_from_request(request: Request, path: str) -> str:
    model = request.query_params.get("model") or ""
    if not model and request.method == "POST":
        try:
            body = json.loads((await request.body()).decode() or "{}")
            model = body.get("model", "")
        except Exception:  # noqa: BLE001
            model = ""
    return model


@app.get("/")
async def root() -> PlainTextResponse:
    return PlainTextResponse("Ollama GPU Router running (self-optimizing)")


@app.get("/health")
async def health() -> JSONResponse:
    healthy = sum(1 for ls in state.lanes.values() if ls.healthy)
    return JSONResponse({"status": "ok" if healthy else "degraded", "healthy_lanes": healthy, "total": len(state.lanes)})


@app.get("/route-info")
async def route_info(model: str) -> JSONResponse:
    ls = router.select(model)
    if not ls:
        raise HTTPException(status_code=503, detail="no healthy lane available")
    return JSONResponse({
        "model": model,
        "target_lane": ls.cfg.name,
        "target_port": ls.cfg.port,
        "physical_gpu": ls.physical_gpu,
        "free_gib": ls.gpu.free_gib if ls.gpu else None,
        "resident_here": model in ls.loaded,
    })


@app.get("/status")
async def status() -> HTMLResponse:
    snap = state.snapshot()
    rows = ""
    for name, ls in snap["lanes"].items():
        free = ls["free_gib"]
        rows += (
            f"<tr><td><b>{name}</b></td><td>GPU {ls['physical_gpu']}</td><td>:{ls['port']}</td>"
            f"<td style='color:{'green' if ls['healthy'] else 'red'}'>"
            f"{'HEALTHY' if ls['healthy'] else 'DOWN'} {('('+ls['error'][:40]+')') if ls['error'] else ''}</td>"
            f"<td>models:{ls['model_count']} resident:{ls['resident_gib']}GiB</td>"
            f"<td>free:{free}GiB</td>"
            f"<td style='color:{'orange' if ls['circuit_open'] else 'inherit'}'>"
            f"{'CIRCUIT-OPEN' if ls['circuit_open'] else 'ok'}</td></tr>"
        )
    gpu_rows = "".join(
        f"<tr><td>GPU {g['index']}</td><td colspan=5>{g['name']} — {g['free_mib']/1024:.1f}/{g['total_mib']/1024:.1f} GiB free</td></tr>"
        for g in snap["gpus"].values()
    )
    html = f"""
    <html><head><title>Ollama GPU Router — Status</title>
    <style>body{{font-family:monospace;background:#0d1117;color:#c9d1d9;padding:2em}}
    table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #30363d;padding:6px 10px;text-align:left}}
    h1{{color:#58a6ff}} .muted{{color:#8b949e}}</style></head>
    <body><h1>Ollama GPU Router</h1>
    <p class='muted'>last refresh: {time.strftime('%H:%M:%S', time.localtime(snap['last_refresh']))} ·
    self-optimizing · failover + auto-recovery enabled</p>
    <h3>Lanes</h3><table><tr><th>Lane</th><th>GPU</th><th>Port</th><th>State</th><th>Resident</th><th>Free</th><th>CB</th></tr>
    {rows}</table>
    <h3>Physical GPUs</h3><table><tr><th>GPU</th><th>Detail</th></tr>
    {gpu_rows}</table></body></html>
    """
    return HTMLResponse(html)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat endpoint with reasoning-model pass-through.

    Ollama's native /api/chat returns the real answer in `message.content`
    (plus `message.thinking` for reasoning models), but Ollama's OpenAI
    /v1/chat/completions returns EMPTY content for thinking models. So we
    proxy to the native endpoint and convert, merging thinking into content
    so OpenAI clients (Hermes, etc.) see the full answer.
    """
    try:
        body = json.loads((await request.body()).decode() or "{}")
    except Exception:
        body = {}
    model = body.get("model", "")
    stream = bool(body.get("stream", False))

    ls = router.select(model)
    if not ls:
        raise HTTPException(status_code=503, detail="no healthy lane available")

    # Streaming: delegate to Ollama's own OpenAI SSE (existing behavior).
    if stream:
        target = f"http://127.0.0.1:{ls.cfg.port}/v1/chat/completions"
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length", "x-lane")}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=cfg.connect_timeout, read=cfg.request_timeout, write=10, pool=10)) as client:
                resp = await client.post(target, content=json.dumps(body).encode(), headers=headers)
            router.on_success(ls)
            return JSONResponse(content=resp.json(), status_code=resp.status_code,
                                headers={"X-GPU-Routed-Lane": ls.cfg.name, "X-GPU-Routed": str(ls.cfg.port)})
        except httpx.HTTPError as e:
            router.on_failure(ls, str(e))
            raise HTTPException(status_code=503, detail=f"lane failed: {e}")

    # Non-streaming: call native /api/chat, merge thinking into content.
    native_body = {
        "model": model,
        "messages": body.get("messages", []),
        "stream": False,
        "options": {k: body[k] for k in ("temperature", "top_p", "top_k", "num_predict", "seed") if k in body},
    }
    # Allow clients to disable thinking explicitly.
    if "think" in body:
        native_body["think"] = body["think"]
    native_body = {k: v for k, v in native_body.items() if v not in (None, {}, "")}

    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=cfg.connect_timeout, read=cfg.request_timeout, write=10, pool=10)) as client:
            resp = await client.post(
                f"http://127.0.0.1:{ls.cfg.port}/api/chat",
                json=native_body,
                headers={"Content-Type": "application/json"},
            )
        ms = (time.time() - t0) * 1000
        if resp.status_code >= 500:
            router.on_failure(ls, f"HTTP {resp.status_code}")
            metrics.record(model, ls.cfg.name, ms, f"HTTP {resp.status_code}")
            alt = router.select(model)
            if alt and alt.cfg.name != ls.cfg.name:
                ls = alt
                resp = await client.post(
                    f"http://127.0.0.1:{ls.cfg.port}/api/chat",
                    json=native_body, headers={"Content-Type": "application/json"},
                )
        else:
            router.on_success(ls)
            metrics.record(model, ls.cfg.name, ms)
    except httpx.HTTPError as e:
        router.on_failure(ls, str(e))
        metrics.record(model, ls.cfg.name, (time.time() - t0) * 1000, str(e))
        alt = router.select(model)
        if alt and alt.cfg.name != ls.cfg.name:
            ls = alt
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(connect=cfg.connect_timeout, read=cfg.request_timeout, write=10, pool=10)) as client:
                    resp = await client.post(f"http://127.0.0.1:{ls.cfg.port}/api/chat", json=native_body, headers={"Content-Type": "application/json"})
            except httpx.HTTPError as e2:
                raise HTTPException(status_code=503, detail=f"all lanes failed: {e2}")
        else:
            raise HTTPException(status_code=503, detail=f"lane failed: {e}")

    try:
        native = resp.json()
    except Exception:
        return JSONResponse(content=resp.text, status_code=resp.status_code,
                            headers={"X-GPU-Routed-Lane": ls.cfg.name, "X-GPU-Routed": str(ls.cfg.port)})
    msg = native.get("message", {})
    content = msg.get("content", "") or ""
    thinking = msg.get("thinking", "") or ""
    if thinking:
        content = f"<thinking>\n{thinking}\n</thinking>\n\n{content}" if content else f"<thinking>\n{thinking}\n</thinking>"

    usage = native.get("prompt_eval_count", 0) + native.get("eval_count", 0)
    openai_resp = {
        "id": f"chatcmpl-{int(t0*1000)}",
        "object": "chat.completion",
        "created": int(t0),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": native.get("prompt_eval_count", 0),
            "completion_tokens": native.get("eval_count", 0),
            "total_tokens": usage,
        },
    }
    return JSONResponse(content=openai_resp, status_code=200,
                        headers={"X-GPU-Routed-Lane": ls.cfg.name, "X-GPU-Routed": str(ls.cfg.port)})


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(request: Request, path: str):
    model = await _model_from_request(request, path)
    # Allow a client to pin a lane via header X-Lane (escape hatch / canary).
    pinned = request.headers.get("x-lane")
    if pinned and pinned in state.lanes:
        ls = state.lanes[pinned]
        if not ls.healthy or state.is_circuit_open(ls):
            raise HTTPException(status_code=503, detail=f"pinned lane {pinned} unavailable")
    else:
        ls = router.select(model)
    if not ls:
        raise HTTPException(status_code=503, detail="no healthy lane available")

    target_url = f"http://127.0.0.1:{ls.cfg.port}/{path}"
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length", "x-lane")}
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=cfg.connect_timeout, read=cfg.request_timeout, write=10, pool=10)) as client:
            resp = await client.request(
                request.method, target_url,
                content=body, params=request.query_params, headers=headers,
            )
        ms = (time.time() - t0) * 1000
        if resp.status_code >= 500:
            router.on_failure(ls, f"HTTP {resp.status_code}")
            metrics.record(model, ls.cfg.name, ms, f"HTTP {resp.status_code}")
            # failover: try one alternate lane for server errors
            alt = router.select(model)
            if alt and alt.cfg.name != ls.cfg.name:
                return await _proxy_to(alt, request, path, model, body, headers)
        else:
            router.on_success(ls)
            metrics.record(model, ls.cfg.name, ms)
        try:
            content = resp.json()
        except Exception:
            content = resp.text
        return JSONResponse(content=content, status_code=resp.status_code, headers={"X-GPU-Routed-Lane": ls.cfg.name, "X-GPU-Routed": str(ls.cfg.port)})
    except httpx.HTTPError as e:
        ms = (time.time() - t0) * 1000
        router.on_failure(ls, str(e))
        metrics.record(model, ls.cfg.name, ms, str(e))
        alt = router.select(model)
        if alt and alt.cfg.name != ls.cfg.name:
            return await _proxy_to(alt, request, path, model, body, headers)
        raise HTTPException(status_code=503, detail=f"all lanes failed; last error: {e}")


async def _proxy_to(ls, request, path, model, body, headers):
    target_url = f"http://127.0.0.1:{ls.cfg.port}/{path}"
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=cfg.connect_timeout, read=cfg.request_timeout, write=10, pool=10)) as client:
            resp = await client.request(request.method, target_url, content=body, params=request.query_params, headers=headers)
        ms = (time.time() - t0) * 1000
        if resp.status_code >= 500:
            router.on_failure(ls, f"HTTP {resp.status_code}")
            metrics.record(model, ls.cfg.name, ms, f"HTTP {resp.status_code}")
        else:
            router.on_success(ls)
            metrics.record(model, ls.cfg.name, ms)
        try:
            content = resp.json()
        except Exception:
            content = resp.text
        return JSONResponse(content=content, status_code=resp.status_code, headers={"X-GPU-Routed-Lane": ls.cfg.name, "X-GPU-Routed": str(ls.cfg.port), "X-GPU-Failover": "1"})
    except httpx.HTTPError as e:
        router.on_failure(ls, str(e))
        metrics.record(model, ls.cfg.name, (time.time() - t0) * 1000, str(e))
        raise HTTPException(status_code=503, detail=f"failover lane failed: {e}")
