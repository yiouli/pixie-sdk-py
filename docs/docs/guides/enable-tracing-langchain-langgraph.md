# Enable Tracing for LangChain/LangGraph

Because LangChain/LangGraph doesn't support OTel out-of-box, Pixie cannot automatically instrument applications built with them.

To enable instrumentation so you can see traces in the web UI, you need to add callback handlers to your LangChain/LangGraph invoke calls.

> You don't need to install `langfuse` as dependency because `pixie-sdk` is built on top of `langfuse` fork.

```python
from langfuse.langchain import CallbackHandler

# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()


agent = create_agent(
    ...
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is 42 + 58?"}]},
    config={"callbacks": [langfuse_handler]}
)
```
