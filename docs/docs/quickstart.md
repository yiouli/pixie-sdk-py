---
sidebar_position: 2
---

# Quickstart

Get started with Pixie SDK in 5 minutes! This guide will walk you through creating and running your first observable AI agent.

## Prerequisites

- Python 3.10 or higher
- pip or poetry for package management
- An OpenAI API key (or other LLM provider)

## Setup

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

## Connect Your Application

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

## Debug with web UI

Visit the web UI [gopixie.ai](https://gopixie.ai) to start debugging.
