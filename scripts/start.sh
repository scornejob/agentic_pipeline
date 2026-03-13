#!/usr/bin/env bash
# scripts/start.sh
# Convenience wrapper: start the stack, pull the default model, then attach to
# the agent's interactive REPL.

set -euo pipefail

source "$(dirname "$0")/../.env" 2>/dev/null || true

DEFAULT_MODEL="${DEFAULT_MODEL:-qwen2.5:7b}"

echo "==> Starting Ollama + Agent services..."
docker compose up -d ollama agent

echo "==> Waiting for Ollama to be healthy..."
until docker compose exec ollama ollama list &>/dev/null; do
  sleep 2
done

echo "==> Checking model: $DEFAULT_MODEL"
if docker compose exec ollama ollama show "$DEFAULT_MODEL" &>/dev/null; then
  echo "    Model already present, skipping pull."
else
  echo "    Pulling model (first time only)..."
  docker compose exec ollama ollama pull "$DEFAULT_MODEL"
fi

echo "==> Attaching to agent REPL (Ctrl-C to detach)..."
docker compose exec -it agent python -m src.main
