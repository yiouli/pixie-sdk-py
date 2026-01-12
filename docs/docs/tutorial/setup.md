---
sidebar_position: 1
---

# Setup

Learn how to install and configure Pixie SDK for your AI agent projects.

## Installation

### Using pip

```bash
pip install pixie-sdk
```

### Using poetry

```bash
poetry add pixie-sdk
```

### From source

```bash
git clone https://github.com/yiouli/pixie-sdk-py.git
cd pixie-sdk-py
poetry install
```

## Project Setup

### 1. Create Your Project Structure

Organize your project with the following structure:

```
my-agent-project/
├── pyproject.toml
├── .env
├── examples/          # Default directory for agents
│   ├── __init__.py
│   └── my_agent.py
└── README.md
```

### 2. Configure Environment Variables

Create a `.env` file in your project root:

```bash
# LLM Provider API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Optional: Database connection for tools
DATABASE_URL=postgresql://user:pass@localhost/db

# Optional: Langfuse for additional observability
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Create Your First Agent

Create `examples/my_agent.py`:

```python
from pixie import app
from pydantic_ai import Agent

# Create the agent
my_agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a helpful assistant."
)

@app
async def hello(name: str) -> str:
    """Say hello to someone."""
    Agent.instrument_all()
    result = await my_agent.run(f"Say hello to {name}")
    return result.output
```

## Starting the Pixie Server

### Basic Usage

Start the server with default settings:

```bash
pixie
```

This will:

- Scan the current directory and `examples/` for agents
- Start the server on `http://127.0.0.1:8000`
- Enable the GraphiQL interface at `http://127.0.0.1:8000/graphql`
- Serve the web UI at `http://127.0.0.1:8000`

### Command-Line Options

The `pixie` command accepts several optional flags:

#### Change Port

```bash
pixie --port 8080
```

#### Specify App Directories

By default, Pixie searches for agents in the current directory and `examples/`. To specify custom directories:

```bash
pixie --app-dir agents/
```

Multiple directories:

```bash
pixie --app-dir agents/ --app-dir custom/
```

#### Change Host

```bash
pixie --host 0.0.0.0  # Listen on all interfaces
```

#### Enable Auto-Reload (Development)

Automatically restart the server when code changes:

```bash
pixie --reload
```

⚠️ **Note:** Use `--reload` only in development, not in production.

#### Configure Log Level

```bash
pixie --log-level debug    # More verbose
pixie --log-level warning  # Less verbose
```

Options: `debug`, `info`, `warning`, `error`, `critical`

#### Disable GraphiQL

GraphiQL is enabled by default. To disable it:

```bash
pixie --no-graphiql
```

#### Full Example

```bash
pixie \
  --port 8080 \
  --host 0.0.0.0 \
  --app-dir agents/ \
  --app-dir tools/ \
  --reload \
  --log-level debug
```

## Server Configuration

### Environment-Based Configuration

You can also configure the server using environment variables:

```bash
# In .env or export in shell
PIXIE_PORT=8080
PIXIE_HOST=0.0.0.0
PIXIE_APP_DIR=agents/
PIXIE_LOG_LEVEL=info
```

Then simply run:

```bash
pixie
```

### Programmatic Configuration

For advanced use cases, you can start the server programmatically:

```python
# server_config.py
from pixie.server import create_app
import uvicorn

app = create_app(
    app_dirs=["agents/", "custom/"],
    enable_graphiql=True
)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        reload=True
    )
```

Run with:

```bash
python server_config.py
```

## Verifying Your Setup

### 1. Check Server Status

After starting the server, you should see:

```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### 2. Access the Web UI

Open your browser to `http://127.0.0.1:8000` and verify:

- The UI loads correctly
- Your agents appear in the app list

### 3. Test GraphQL Endpoint

Navigate to `http://127.0.0.1:8000/graphql` to access GraphiQL.

Try this query to list available apps:

```graphql
query {
  apps {
    name
    description
    inputSchema
    outputSchema
  }
}
```

### 4. Test an Agent

In the web UI:

1. Select your agent
2. Enter test input
3. Click "Run"
4. Verify the output

## Project Dependencies

### Required Dependencies

Pixie SDK requires:

- `pydantic >= 2.0` - For data validation
- `fastapi` - For the GraphQL server
- `strawberry-graphql` - For GraphQL schema
- `uvicorn` - ASGI server

These are installed automatically with `pixie-sdk`.

### Agent Framework Dependencies

Install the agent framework(s) you plan to use:

```bash
# PydanticAI
pip install pydantic-ai

# LangChain
pip install langchain langchain-openai

# LangGraph
pip install langgraph

# OpenAI Agents SDK
pip install openai-agents
```

### Optional Dependencies

For enhanced observability:

```bash
# Langfuse integration
pip install langfuse

# OpenTelemetry (included by default)
pip install opentelemetry-api opentelemetry-sdk
```

## Directory Structure Best Practices

### Organizing Multiple Agents

```
project/
├── examples/
│   ├── __init__.py
│   ├── chatbots/
│   │   ├── __init__.py
│   │   ├── customer_support.py
│   │   └── personal_assistant.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── search_tools.py
│   │   └── database_tools.py
│   └── workflows/
│       ├── __init__.py
│       └── research_workflow.py
├── .env
└── pyproject.toml
```

Start the server:

```bash
pixie --app-dir examples/chatbots/ --app-dir examples/workflows/
```

### Shared Utilities

Create a utilities module for shared code:

```
project/
├── examples/
│   └── my_agents.py
├── utils/
│   ├── __init__.py
│   ├── prompts.py
│   └── helpers.py
└── pyproject.toml
```

Import in your agents:

```python
from utils.prompts import SYSTEM_PROMPT
from pixie import app

@app
async def my_agent(query: str) -> str:
    # Use shared utilities
    ...
```

## Next Steps

Now that your environment is set up:

1. [Learn the Web UI](./web-ui.md) - Explore the interface
2. [Structured I/O](./structured-io.md) - Use Pydantic models
3. [Interactive Apps](./interactive-app.md) - Build multi-turn agents
4. [Interactive Tools](./interactive-tool.md) - Add user input to tools

## Troubleshooting

### Port Already in Use

```
ERROR:    [Errno 98] Address already in use
```

**Solution:** Use a different port:

```bash
pixie --port 8080
```

### Agents Not Discovered

If your agents don't appear in the UI:

1. Verify the `@app` decorator is used
2. Check that files are in the scanned directories
3. Ensure `__init__.py` exists in the directory
4. Check server logs for errors

### Import Errors

```
ModuleNotFoundError: No module named 'pixie'
```

**Solution:** Ensure pixie-sdk is installed:

```bash
pip install --upgrade pixie-sdk
```

### Environment Variables Not Loaded

**Solution:** Use `python-dotenv` to load `.env`:

```bash
pip install python-dotenv
```

Add to your agent file:

```python
from dotenv import load_dotenv
load_dotenv()
```

Or use the `--env-file` flag:

```bash
pixie --env-file .env
```
