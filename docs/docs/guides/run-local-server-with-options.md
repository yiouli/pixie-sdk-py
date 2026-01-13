# Run Local Server with Options

Learn how to start and configure the Pixie server for local development and debugging.

## Quick Start

Start the server with default settings:

```bash
pixie
```

The server starts on `http://127.0.0.1:8000` with GraphQL endpoint at `/graphql`.

## Server Options

### Host and Port

```bash
pixie --host 0.0.0.0 --port 8080
```

- `--host 0.0.0.0` - Accept connections from any network interface (remote debugging)
- `--host 127.0.0.1` - Localhost only (default)
- `--port PORT` - Custom port (default is 8000)

### Auto-Reload

```bash
pixie --reload
```

Automatically restarts server when code changes. Use only in development.

### Logging Modes

```bash
# Default mode - minimal logging
pixie

# Verbose mode - INFO level for all modules
pixie --log-mode verbose

# Debug mode - DEBUG level for all modules
pixie --log-mode debug
```

| Mode      | Level                              | What You See                           |
| --------- | ---------------------------------- | -------------------------------------- |
| `default` | INFO for pixie, WARNING for others | Server events, app registration        |
| `verbose` | INFO for all                       | Framework instrumentations, operations |
| `debug`   | DEBUG for all                      | Everything including internal traces   |

### Combining Options

```bash
# Development
pixie --reload --log-mode verbose --port 8080

# Remote debugging
pixie --host 0.0.0.0 --log-mode debug
```

## Application Discovery

The server automatically discovers and registers Python files with `@app` decorated functions in the current directory (recursively). It skips:

- Files starting with `_`
- Virtual environments (`.venv`, `venv`, `site-packages`)
- `__pycache__` directories

## Framework Instrumentation

The server automatically enables instrumentation for supported AI frameworks:

- Pydantic AI
- OpenAI Agents SDK
- Google ADK
- CrewAI
- DSpy

If a framework isn't installed, a warning is logged (in verbose/debug mode only).

## Server Endpoints

- **GraphQL:** `http://127.0.0.1:8000/graphql` (API and interactive GraphiQL interface)
- **Root:** `http://127.0.0.1:8000/` (server info)

## Environment Variables

The server automatically loads `.env` files from the current directory:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
```
