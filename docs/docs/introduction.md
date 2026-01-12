---
sidebar_position: 1
---

# Introduction

Welcome to **Pixie SDK** - a powerful framework for building observable and controllable AI agents.

## What is Pixie SDK?

Pixie SDK provides observability and control for AI agents by wrapping agent functions with the `@app` decorator. It enables you to:

- 🔍 **Monitor** - Automatic tracing and observability for your AI agents
- ⏸️ **Control** - Pause, resume, and interact with running agents
- 🔌 **Integrate** - Works seamlessly with popular agent frameworks like PydanticAI, LangChain, and LangGraph
- 📊 **Visualize** - Built-in web UI for exploring agent execution and traces
- 🚀 **Deploy** - Easy GraphQL API for integration with any client

## Key Features

### Automatic Instrumentation

Pixie automatically captures detailed traces of your agent's execution, including:

- LLM calls and responses
- Tool usage
- Agent reasoning steps
- Performance metrics

### Interactive Agents

Build multi-turn, interactive agents that can:

- Request input from users mid-execution
- Stream results in real-time
- Handle complex conversational flows

### Framework Support

Pixie SDK works with your favorite AI frameworks:

- **PydanticAI** - Type-safe agents with Pydantic models
- **LangChain** - Composable agent workflows
- **LangGraph** - State machine-based agents
- **OpenAI Agents SDK** - Direct OpenAI integration

### GraphQL API

Every Pixie app is automatically exposed via GraphQL, enabling:

- Easy integration with web and mobile clients
- Real-time subscriptions for streaming
- Standardized API for all agents

## How It Works

Pixie SDK uses a simple decorator pattern to transform your agent functions into observable applications:

```python
from pixie import app
from pydantic_ai import Agent

weather_agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a helpful weather assistant."
)

@app
async def weather(query: str) -> str:
    """Get weather information."""
    Agent.instrument_all()  # Enable tracing
    result = await weather_agent.run(query)
    return result.output
```

That's it! Your agent is now:

- Automatically instrumented
- Exposed via GraphQL API
- Ready to use with the Pixie web UI

## Architecture Overview

Pixie SDK consists of three main components:

1. **SDK** - Python library for decorating and instrumenting agents
2. **Server** - GraphQL server that manages agent execution
3. **Web UI** - React-based interface for interacting with agents

```
┌─────────────┐
│  Your Agent │ ──> Decorated with @app
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Pixie Server│ ──> GraphQL API + Execution Management
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Web UI     │ ──> Visualize & Interact
└─────────────┘
```

## Use Cases

Pixie SDK is perfect for:

- **Development & Debugging** - Understand how your agents behave
- **Production Monitoring** - Track agent performance and errors
- **Interactive Applications** - Build chat interfaces and multi-turn workflows
- **Tool Development** - Test and validate agent tools
- **Research & Education** - Explore agent reasoning patterns

## Next Steps

Ready to get started? Check out the [Quickstart](./quickstart.md) guide to build your first Pixie application in minutes.

Or dive deeper into specific topics:

- [Setup & Installation](./tutorial/setup.md)
- [Web UI Guide](./tutorial/web-ui.md)
- [Concepts & Architecture](./concepts/architecture.md)
- [Examples Repository](https://github.com/yiouli/pixie-examples)
