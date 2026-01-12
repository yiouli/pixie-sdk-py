---
sidebar_position: 2
---

# Quickstart

Get started with Pixie SDK in 5 minutes! This guide will walk you through creating and running your first observable AI agent.

## Prerequisites

- Python 3.10 or higher
- pip or poetry for package management
- An OpenAI API key (or other LLM provider)

## Installation

Install Pixie SDK using pip or poetry:

```bash
# Using pip
pip install pixie-sdk

# Using poetry
poetry add pixie-sdk
```

## Your First Pixie Application

Let's create a simple weather assistant agent.

### 1. Create the Agent

Create a new file `weather_agent.py`:

```python
from pixie import app
from pydantic_ai import Agent

# Create the agent
weather_agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a helpful weather assistant."
)

# Decorate with @app
@app
async def weather(query: str) -> str:
    """Get weather information for any location."""
    # Enable instrumentation for observability
    Agent.instrument_all()

    # Run the agent
    result = await weather_agent.run(query)

    # Return the output
    return result.output
```

That's it! Just three simple steps:

1. Create your agent
2. Decorate your function with `@app`
3. Call `Agent.instrument_all()` for tracing

### 2. Set Up Environment

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Start the Pixie Server

Run the Pixie server to make your agent available:

```bash
pixie
```

You should see:

```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

The server automatically discovers all `@app` decorated functions in your project!

### 4. Open the Web UI

Open your browser and navigate to:

```
http://127.0.0.1:8000
```

You'll see the Pixie web interface with your `weather` agent available.

### 5. Test Your Agent

1. Select your `weather` agent from the list
2. Enter a query like "What's the weather in San Francisco?"
3. Click "Run" or press Enter
4. Watch your agent execute in real-time!

## What Just Happened?

With just a few lines of code, you've created an AI agent that:

- ✅ Is automatically instrumented with tracing
- ✅ Exposes a GraphQL API
- ✅ Has a web UI for testing
- ✅ Captures all LLM calls and reasoning
- ✅ Streams responses in real-time

## Explore the Traces

Click on the "Debug" tab in the web UI to see:

- **Execution timeline** - When each step occurred
- **LLM calls** - Prompts, responses, and token usage
- **Performance metrics** - Latency and duration
- **Agent reasoning** - Internal decision-making process

## Next Steps

Now that you have a basic agent running, explore more advanced features:

### Add Structured Input

Use Pydantic models for type-safe input:

```python
from pydantic import BaseModel

class WeatherQuery(BaseModel):
    location: str
    units: str = "celsius"

@app
async def weather(query: WeatherQuery) -> str:
    Agent.instrument_all()
    result = await weather_agent.run(
        f"What's the weather in {query.location}? Use {query.units}."
    )
    return result.output
```

### Build Interactive Agents

Create multi-turn conversations:

```python
from pixie import PixieGenerator, UserInputRequirement

@app
async def chat(_: None) -> PixieGenerator[str, str]:
    Agent.instrument_all()

    yield "Hello! How can I help you today?"

    while True:
        user_input = yield UserInputRequirement(str)

        if user_input.lower() in {"exit", "quit"}:
            yield "Goodbye!"
            break

        result = await agent.run(user_input)
        yield result.output
```

### Add Tools to Your Agent

Give your agent capabilities:

```python
from pydantic_ai import RunContext

@weather_agent.tool
async def get_temperature(ctx: RunContext[None], location: str) -> float:
    """Get the current temperature for a location."""
    # Your implementation here
    return 72.5

@app
async def weather(query: str) -> str:
    Agent.instrument_all()
    result = await weather_agent.run(query)
    return result.output
```

## Learn More

- [Tutorial: Setup](./tutorial/setup.md) - Detailed setup and configuration
- [Tutorial: Web UI](./tutorial/web-ui.md) - Learn all Web UI features
- [Examples Repository](https://github.com/yiouli/pixie-examples) - More examples
- [Concepts: Architecture](./concepts/architecture.md) - How Pixie works

## Troubleshooting

### Agent Not Discovered

Make sure your Python file is in a directory that Pixie scans. By default, Pixie looks in:

- Current directory
- `examples/` directory
- Any directory specified with `--app-dir`

### Import Errors

Ensure pixie-sdk is installed:

```bash
pip install --upgrade pixie-sdk
```

### Server Won't Start

Check that port 8000 is not already in use:

```bash
lsof -i :8000
```

Or specify a different port:

```bash
pixie --port 8080
```

## Need Help?

- Check the [Examples Repository](https://github.com/yiouli/pixie-examples)
- Review the [Tutorial](./tutorial/setup.md) section
- Read the [Concepts](./concepts/architecture.md) documentation
