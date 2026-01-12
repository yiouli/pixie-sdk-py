# Pixie SDK

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://www.python.org/downloads/)

**Observability and Interactive Debugging for AI Agents**

Pixie SDK enables you to build AI applications with built-in observability, pause/resume capabilities, and interactive debugging. Wrap your AI agent functions with the `@app` decorator to automatically expose them through a GraphQL API and gain full visibility into their execution.

## Features

- 🔍 **Automatic Observability**: Full tracing and monitoring of AI agent execution
- ⏸️ **Pause & Resume**: Interactive debugging with breakpoints at LLM calls, tool usage, or custom points
- 🎯 **GraphQL API**: Auto-generated API for all registered applications
- 🔄 **Multi-Turn Interactions**: Support for stateful, conversational applications
- 📊 **OpenTelemetry Integration**: Native support for OTLP traces
- 🛠️ **Framework Support**: Works with Pydantic AI, OpenAI Agents SDK, LangChain, LangGraph, and more

## Installation

```bash
pip install pixie-sdk
```

## Quick Start

### 1. Create a Pixie Application

```python
from pixie import app
from pydantic_ai import Agent

# Create your AI agent
weather_agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a helpful weather assistant."
)

@app
async def weather(query: str) -> str:
    """Get weather information."""
    # Enable instrumentation for observability
    Agent.instrument_all()

    # Run the agent
    result = await weather_agent.run(query)
    return result.output
```

### 2. Start the Pixie Server

```bash
pixie
```

The server will:

- Discover and register all `@app` decorated functions
- Start a GraphQL API at `http://127.0.0.1:8000/graphql`
- Enable the GraphiQL interface for interactive exploration

### 3. Run Your Application

Use the GraphiQL interface or make a GraphQL subscription:

```graphql
subscription {
  run(id: "your.module.weather", inputData: "What's the weather in Tokyo?") {
    runId
    status
    data
    trace {
      otlpTrace {
        resourceSpans {
          scopeSpans {
            spans {
              name
              startTimeUnixNano
            }
          }
        }
      }
    }
  }
}
```

## Application Types

### Single-Turn Application

For applications that execute once and return a result:

```python
from pydantic import BaseModel
from pixie import app

class WeatherQuery(BaseModel):
    location: str
    units: str = "celsius"

@app
async def weather(query: WeatherQuery) -> str:
    """Simple weather query."""
    # Your implementation
    return f"Weather in {query.location}: Sunny"
```

### Multi-Turn Interactive Application

For conversational or stateful applications that need user input:

```python
from pixie import app, PixieGenerator, UserInputRequirement

@app
async def chatbot(_: None) -> PixieGenerator[str, str]:
    """Interactive chatbot with conversation history."""
    yield "Hello! How can I help you?"

    while True:
        # Request user input
        user_msg = yield UserInputRequirement(str)

        if user_msg.lower() in {"exit", "quit"}:
            yield "Goodbye!"
            break

        # Process and respond
        response = await process_message(user_msg)
        yield response
```

## Interactive Debugging

### Pause Execution

Use GraphQL mutations to pause at specific points:

```graphql
mutation {
  pauseRun(runId: "your-run-id", timing: BEFORE, breakpointTypes: [LLM, TOOL])
}
```

### Resume Execution

```graphql
mutation {
  resumeRun(runId: "your-run-id")
}
```

### Send User Input

```graphql
mutation {
  sendInput(runId: "your-run-id", inputData: "User's response")
}
```

## Type Safety

Pixie supports type-safe applications using Pydantic models:

```python
from pydantic import BaseModel
from pixie import app, PixieGenerator

class UserPreferences(BaseModel):
    language: str
    max_results: int

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str

@app
async def search(query: str) -> PixieGenerator[SearchResult, UserPreferences]:
    """Type-safe search with user preferences."""
    yield SearchResult(
        title="Initial Result",
        url="https://example.com",
        snippet="..."
    )

    # Get structured user preferences
    prefs = yield UserInputRequirement(UserPreferences)

    # Continue with preferences
    for result in search_with_preferences(query, prefs):
        yield result
```

## Framework Integrations

Pixie automatically instruments popular AI frameworks:

- **Pydantic AI**: `Agent.instrument_all()`
- **OpenAI Agents SDK**: Auto-instrumented
- **Google ADK**: Auto-instrumented
- **CrewAI**: Auto-instrumented
- **DSpy**: Auto-instrumented
- **LangChain**: Compatible via OpenTelemetry
- **LangGraph**: Compatible via OpenTelemetry

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# Pixie Server
PIXIE_SDK_PORT=8000

# Langfuse (for enhanced observability)
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com

# Your AI Provider Keys
OPENAI_API_KEY=sk-...
```

### Server Options

```python
from pixie.server import start_server

start_server(
    host="0.0.0.0",
    port=8000,
    reload=True  # Enable auto-reload for development
)
```

## GraphQL API

### Queries

- `healthCheck`: Server health status
- `listApps`: List all registered applications with schemas

### Mutations

- `pauseRun(runId, timing, breakpointTypes)`: Pause execution
- `resumeRun(runId)`: Resume paused execution
- `sendInput(runId, inputData)`: Send user input

### Subscriptions

- `run(id, inputData)`: Execute an application and stream updates

## Examples

Check out the [pixie-examples](https://github.com/yiouli/pixie-examples) repository for complete examples:

- **Quickstart**: Simple chatbot and agent examples
- **Pydantic AI**: Bank support, flight booking, SQL generation
- **OpenAI Agents SDK**: Customer service, routing, LLM-as-a-judge
- **LangChain**: Basic agents, SQL agents, personal assistant
- **LangGraph**: RAG systems, multi-agent workflows

## Development

### Install for Development

```bash
# Clone the repository
git clone https://github.com/yiouli/pixie-sdk-py.git
cd pixie-sdk-py

# Install with Poetry
poetry install

# Run tests
poetry run pytest
```

### Project Structure

```
pixie-sdk-py/
├── pixie/              # Core SDK
│   ├── __init__.py     # Public API
│   ├── execution_context.py  # Pause/resume functionality
│   ├── otel_types.py   # OpenTelemetry types
│   ├── registry.py     # Application registry
│   ├── schema.py       # GraphQL schema
│   ├── server.py       # FastAPI server
│   ├── types.py        # Core types
│   └── utils.py        # Utilities
├── langfuse/           # Langfuse SDK (bundled)
└── tests/              # Test suite
```

## Requirements

- Python 3.10 or higher
- FastAPI
- Strawberry GraphQL
- Pydantic
- OpenTelemetry

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

- 📧 Email: yol@gopixie.ai
- 🐛 Issues: [GitHub Issues](https://github.com/yiouli/pixie-sdk-py/issues)

## Acknowledgments

Built with ❤️ using [Langfuse](https://langfuse.com) for observability.
