# Pixie

[![MIT License](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/discord/1459772566528069715?style=flat-square&logo=Discord&logoColor=white&label=Discord&color=%23434EE4)](https://discord.gg/YMNYu6Z3)

**Generate Evals from debugging LLM Applications**

Evals takes a lot of effort to setup, and the results are not always helpful. What if we can generate evals automatically based on how you debug your LLM applications?

## Demo

[Demo](https://github.com/user-attachments/assets/84472190-cd50-4e9a-9494-e30b43457031)


## Get Started

### 1. Setup

In your project folder, install `pixie-sdk` package:

```bash
pip install pixie-sdk
```

Start the local debug server by running:

```bash
pixie
```

### 2. Connect Your Application

Add `@pixie.session` decorator to any code you'd like to debug, use `pixie.print(...)` to log data to the debugger UI.

```python
# my_chatbot.py
import asyncio
from pydantic_ai import Agent
import pixie.sdk as pixie

# You can implement your application using any major AI development framework
agent = Agent(
    name="Simple chatbot",
    instructions="You are a helpful assistant.",
    model="gpt-4o-mini",
)


@pixie.session
async def my_chatbot():
    """Chatbot application example."""
    await pixie.print("How can I help you today?")
    messages = []
    while True:
        user_msg = await asyncio.to_thread(input)
        await pixie.print(user_msg, from_user=True)
        response = await agent.run(user_msg, message_history=messages)
        messages = response.all_messages()
        await pixie.print(response.output)

```

### 3. Debug with web UI

Visit the web UI [gopixie.ai](https://gopixie.ai) to start debugging.
Run your application as normal while `pixie` debug server is running, and your session would show up in the debugger UI.

## Important Links

- [**Documentation**](https://yiouli.github.io/pixie-sdk-py/) - Complete documentation with tutorials and API reference
- [**Examples**](https://github.com/yiouli/pixie-examples) - Real-world examples and sample applications
- [**Demo**](https://gopixie.ai/?url=https%3A%2F%2Fdemo.yiouli.us%2Fgraphql) - Live Demo with the examples server setup
- [**Discord**](https://discord.gg/YMNYu6Z3) - Join our community for support and discussions

## Acknowledgments

This project is built on top of many awesome open-source projects:

- [Langfuse](https://github.com/langfuse/langfuse) for instrumentation
- [Pydantic](https://github.com/pydantic/pydantic) for structured data validation
- [FastAPI](https://github.com/fastapi/fastapi) for web API
- [Strawberry](https://github.com/strawberry-graphql/strawberry) for graphql
- [Uvicorn](https://github.com/Kludex/uvicorn) for web server
- [Janus](https://github.com/aio-libs/janus) for sync-async queue
- [docstring-parser](https://github.com/rr-/docstring_parser) for docstring parsing
