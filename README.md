# Agentic Pipeline

A fully local, Docker-based agentic AI pipeline powered by **Ollama**. No API keys required to get started — but the architecture makes it trivial to swap in any cloud LLM (OpenAI, Anthropic, Gemini, …) when you need to.

## Architecture

```
docker-compose.yml
├── ollama          — local LLM inference server (ollama/ollama)
├── agent           — Python ReAct agent (built from Dockerfile)
└── open-webui      — optional chat UI (profile: ui)

src/
├── main.py                  — CLI entry-point (REPL or single-shot)
├── llm/
│   └── provider.py          — LLMProvider abstraction + factory
│                              (Ollama / OpenAI-compatible / Anthropic)
└── agent/
    ├── pipeline.py          — ReAct loop
    ├── tools.py             — tool registry + built-in tools
    └── memory.py            — rolling message history

config/
└── config.yaml              — model, agent behaviour, tool toggles
```

## Quick start

```bash
# 1. Clone / enter the repo
cd agentic_pipeline

# 2. Copy the env template (API keys are optional)
cp .env.example .env

# 3. Start everything and pull the default model
./scripts/start.sh
```

The script:
1. Starts the `ollama` and `agent` containers
2. Waits until Ollama is healthy
3. Pulls the default model (`qwen2.5:7b` unless overridden)
4. Attaches you to the interactive agent REPL

### Manual start

```bash
# Start services (Ollama + agent)
docker compose up -d ollama agent

# Pull the default model inside the Ollama container
docker compose exec ollama ollama pull qwen2.5:7b

# Attach to the agent REPL
docker compose exec -it agent python -m src.main

# Single-shot task
docker compose exec agent python -m src.main "Summarise the Wikipedia article on Docker"
```

### Optional: Open WebUI

```bash
docker compose --profile ui up -d
# then open http://localhost:3000
```

### Optional: Pull extra models

```bash
DEFAULT_MODEL=llama3.2:3b ./scripts/pull_models.sh
# or pull several at once
./scripts/pull_models.sh mistral:7b phi3:mini
```

## Changing the model

Edit `config/config.yaml`:

```yaml
llm:
  model: llama3.2:3b   # any model available in your Ollama instance
```

Or set `DEFAULT_MODEL` in `.env`.

## Switching to a cloud LLM

1. Set `provider` in `config/config.yaml`:

   ```yaml
   llm:
     provider: openai   # or anthropic
     model: gpt-4o
   ```

2. Add your API key to `.env`:

   ```
   OPENAI_API_KEY=sk-...
   ```

3. Rebuild and restart the agent container:

   ```bash
   docker compose up -d --build agent
   ```

Supported providers: `ollama` · `openai` (also works for Azure, Groq, Together AI) · `anthropic`

## Built-in tools

| Tool | Description |
|------|-------------|
| `calculator` | Safe math expression evaluator |
| `python_eval` | Execute an arbitrary Python snippet, capture stdout |
| `shell` | Run a shell command in the workspace directory |
| `file_read` | Read a file from `/app/workspace` |
| `file_write` | Write a file to `/app/workspace` |
| `web_fetch` | Fetch and extract text from a URL |

Toggle any tool in `config/config.yaml` under `tools:`.

## Adding custom tools

Register a new tool in [src/agent/tools.py](src/agent/tools.py) using the `@tool` decorator:

```python
@tool(
    name="my_tool",
    description="Does something useful.",
    usage='Action Input: {"arg": "value"}',
)
def _my_tool(input_str: str) -> str:
    data = _parse_json_input(input_str)
    # ... your logic ...
    return "result"
```

The agent will discover it automatically on the next run.

## GPU support

Uncomment the `deploy` block in [docker-compose.yml](docker-compose.yml) under the `ollama` service:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

