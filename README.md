# Pixie SDK

[![MIT License](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/discord/1459772566528069715?style=flat-square&logo=Discord&logoColor=white&label=Discord&color=%23434EE4)](https://discord.gg/YMNYu6Z3)

**Interactive debugging tool for AI applications.**

Debug your AI application interactively, and inspect traces in real-time, all in one single web UI.

<a href="https://youtu.be/FEIvuiPDr9I" target="_blank" rel="noopener">
  <img src="https://github.com/user-attachments/assets/ac2ec55f-b487-4b3f-ae6f-b8743ad296e4" alt="Demo video" width="800" target="_blank" />
</a>

## Features

- **Interactive Debugging**: Support for two way interaction with your application, plus the ability to pause/resume/stop.
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
pip install pydantic-ai-slim
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

- [**Documentation**](https://yiouli.github.io/pixie-sdk-py/)
- [**Examples**](https://github.com/yiouli/pixie-examples)
- [**Discord**](https://discord.gg/YMNYu6Z3)

## Acknowledgments

This project is built on top of many awesome open-source projects:

- [Langfuse](https://github.com/langfuse/langfuse) for instrumentation
- [Pydantic](https://github.com/pydantic/pydantic) for structured data validation
- [FastAPI](https://github.com/fastapi/fastapi) for web API
- [Strawberry](https://github.com/strawberry-graphql/strawberry) for graphql
- [Uvicorn](https://github.com/Kludex/uvicorn) for web server
- [Janus](https://github.com/aio-libs/janus) for sync-async queue
- [docstring-parser](https://github.com/rr-/docstring_parser) for docstring parsing
