#!/usr/bin/env python3
import subprocess
import sys

venv = "/home/scott/ai-workspace/repos/ollama-router/.venv"
packages = ["fastapi", "uvicorn[standard]", "httpx"]

cmd = [f"{venv}/bin/pip", "install", "--quiet"] + packages
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
sys.exit(result.returncode)