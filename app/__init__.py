# Ollama GPU Router — self-optimizing bifurcation + load-splitting package.
# Re-export the FastAPI app so `uvicorn app:app` resolves correctly.
from app import main as _main

app = _main.app
__all__ = ["app"]
