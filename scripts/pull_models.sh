#!/usr/bin/env bash
# scripts/pull_models.sh
# Pull one or more Ollama models into the running ollama container.
# Usage:  ./scripts/pull_models.sh [model1] [model2] ...
# Default model is read from DEFAULT_MODEL env var or config.yaml.

set -euo pipefail

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
MODELS=("${@:-${DEFAULT_MODEL:-qwen2.5:7b}}")

for MODEL in "${MODELS[@]}"; do
  echo "Pulling model: $MODEL"
  curl -sf "${OLLAMA_HOST}/api/pull" \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"${MODEL}\"}" | \
    python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try:
        d = json.loads(line)
        status = d.get('status','')
        total  = d.get('total', 0)
        compl  = d.get('completed', 0)
        if total:
            pct = int(compl / total * 100)
            print(f'\r  {status}: {pct}%', end='', flush=True)
        else:
            print(f'  {status}', flush=True)
    except Exception:
        pass
print()
"
  echo "  Done: $MODEL"
done
