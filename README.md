# Pixie

[![MIT License](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/discord/1459772566528069715?style=flat-square&logo=Discord&logoColor=white&label=Discord&color=%23434EE4)](https://discord.gg/YMNYu6Z3)

**web UI for AI agents manual testing.**
LangSmith Studio/Google ADK Web for AI agents built with other frameworks. Easy to setup, support multi-turn interactions, Human-in-the-loop, live tracing, and more.

## Why?

Manually testing AI applications is time-consuming and cumbersome.
Especially for early stage development, it doesn’t make sense to build e2e product just to test, or setup automated tests/evals.
So the process ends up being a lot of inputting awkwardly into the command line, and looking through walls of logs in different places.

## Demo

[Demo Video](https://github.com/user-attachments/assets/8c164f1f-9f0f-4a4e-a1f2-ba0c3ca6f58c)

## Features

- **Interactive Testing**: Support for two way interaction with your application, plus the ability to pause/resume/stop.
- **Real-time Observability**: Application traces are streamed in real-time while you debug.
- **Structured Input/Output**: Native support for structured input/output using Pydantic models.
- **Data Privacy**: Communications are only between your browser and your server, your data stays private.
- **Framework Support**: Out-of-the-box support for popular AI development frameworks like Pydantic AI, OpenAI Agents SDK, LangChain, LangGraph, and more.

## Get Started

### 1. Setup

In your project folder, install `pixie-sdk` package:

```bash
pip install pixie-sdk
```

Create _.env_ file in your project folder and add LLM API key(s):

```ini
# .env
OPENAI_API_KEY=...
```

Add AI Development framework depdendencies as needed:

```bash
pip install pydantic-ai-slim[openai]
```

Start the local server for debugging by running:

```bash
pixie
```

### 2. Connect Your Application

Add `@pixie.app` decorator to your main application function:

```python
# my_chatbot.py
from pydantic_ai import Agent

import pixie

# You can implement your application using any major AI development framework
agent = Agent(
    name="Simple chatbot",
    instructions="You are a helpful assistant.",
    model="gpt-4o-mini",
)


@pixie.app
async def my_chatbot():
    """Chatbot application example."""
    yield "How can I help you today?"
    messages = []
    while True:
        user_msg = yield pixie.InputRequired(str)
        response = await agent.run(user_msg, message_history=messages)
        messages = response.all_messages()
        yield response.output

```

You should see in the log of `pixie` server that your app is registered.

### 3. Debug with web UI

Visit the web UI [gopixie.ai](https://gopixie.ai) to start debugging.

## Important Links

- [**Documentation**](https://yiouli.github.io/pixie-sdk-py/) - Complete documentation with tutorials and API reference
- [**Examples**](https://github.com/yiouli/pixie-examples) - Real-world examples and sample applications
- [**Discord**](https://discord.gg/YMNYu6Z3) - Join our community for support and discussions
- [**Web UI Repo**](https://github.com/yiouli/pixie-ui) - Github repository for the Web UI

## Acknowledgments

This project is built on top of many awesome open-source projects:

- [Langfuse](https://github.com/langfuse/langfuse) for instrumentation
- [Pydantic](https://github.com/pydantic/pydantic) for structured data validation
- [FastAPI](https://github.com/fastapi/fastapi) for web API
- [Strawberry](https://github.com/strawberry-graphql/strawberry) for graphql
- [Uvicorn](https://github.com/Kludex/uvicorn) for web server
- [Janus](https://github.com/aio-libs/janus) for sync-async queue
- [docstring-parser](https://github.com/rr-/docstring_parser) for docstring parsing
