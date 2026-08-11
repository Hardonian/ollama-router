#!/usr/bin/env bash
# gpu-router-status.sh — operator one-shot: print live router/lane/GPU state.
set -Eeuo pipefail
PORT="${1:-11438}"
echo "=== ROUTER HEALTH ==="
curl -s --max-time 3 "http://127.0.0.1:$PORT/health" || echo "router down"
echo; echo "=== LANE STATE ==="
curl -s --max-time 3 "http://127.0.0.1:$PORT/status" \
  | grep -oE '<tr><td>.*</td></tr>' | sed 's/<[^>]*>//g' | sed 's/  */ /g' || echo "status unavailable"
echo; echo "=== PHYSICAL GPU FREE VRAM ==="
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader
echo; echo "=== WATCHDOG LAST REPORT ==="
cat /home/scott/.hermes/state/gpu-router-watchdog.json 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print('isolation:',d['isolation']);print('router:',d['router']);print('lanes:',[(l['port'],l['reachable'],l['models']) for l in d['lanes']])" 2>/dev/null || echo "no watchdog report yet"
