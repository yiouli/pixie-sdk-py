---
sidebar_position: 2
---

# Instrumentation

Learn how Pixie SDK instruments your AI agents to provide detailed observability through OpenTelemetry and Langfuse integration.

## Overview

Instrumentation is the process of adding observability to your code. Pixie SDK provides automatic instrumentation for AI agents, capturing:

- LLM calls and responses
- Tool executions
- Agent reasoning steps
- Performance metrics
- Costs and token usage

## How Instrumentation Works

### The Magic Line

Adding observability to your agent is as simple as one line:

```python
from pydantic_ai import Agent
from pixie import app

agent = Agent("openai:gpt-4o-mini")

@app
async def my_agent(query: str) -> str:
    Agent.instrument_all()  # ← This enables tracing!
    result = await agent.run(query)
    return result.output
```

### What `Agent.instrument_all()` Does

1. **Initializes OpenTelemetry** - Sets up tracer provider
2. **Registers Langfuse** - Configures Langfuse integration (if API keys present)
3. **Patches Libraries** - Instruments PydanticAI, OpenAI, etc.
4. **Starts Span** - Creates root span for the execution

## OpenTelemetry Concepts

### Traces, Spans, and Events

**Trace**

- Represents a complete execution path
- Contains multiple spans
- Has a unique trace ID

**Span**

- Represents a single operation
- Has start and end time
- Contains attributes and events
- Can have child spans

**Event**

- Point-in-time occurrence within a span
- Contains timestamp and attributes
- Examples: log messages, state changes

### Span Hierarchy Example

```
Trace: weather_agent_execution
│
├─ Span: weather_agent (root)
│  │  Duration: 2.5s
│  │  Status: OK
│  │
│  ├─ Span: Agent.run
│  │  │  Duration: 2.3s
│  │  │
│  │  ├─ Span: LLM Call (gpt-4o-mini)
│  │  │  │  Duration: 1.8s
│  │  │  │  Attributes:
│  │  │  │    - model: gpt-4o-mini
│  │  │  │    - input_tokens: 150
│  │  │  │    - output_tokens: 75
│  │  │  │    - cost: $0.0023
│  │  │  │
│  │  │  ├─ Event: prompt
│  │  │  │    Timestamp: t0
│  │  │  │    Content: "What's the weather..."
│  │  │  │
│  │  │  └─ Event: response
│  │  │       Timestamp: t0 + 1.8s
│  │  │       Content: "The weather is..."
│  │  │
│  │  └─ Span: Tool Call (get_temperature)
│  │     │  Duration: 0.3s
│  │     │  Attributes:
│  │     │    - tool_name: get_temperature
│  │     │    - tool_input: {"location": "SF"}
│  │     │
│  │     ├─ Event: tool_start
│  │     └─ Event: tool_end
│  │
│  └─ Span: Result Formatting
│     Duration: 0.2s
│
└─ End
```

## Captured Information

### LLM Calls

For each LLM API call, Pixie captures:

**Request Data**

- **Model** - Which model was used (e.g., gpt-4o-mini)
- **Prompt** - Complete input prompt
- **System Prompt** - Agent's system instructions
- **Temperature** - Sampling temperature
- **Max Tokens** - Token limit
- **Tools** - Available tools/functions

**Response Data**

- **Output** - Complete LLM response
- **Finish Reason** - Why generation stopped
- **Token Usage**
  - Input tokens
  - Output tokens
  - Total tokens
- **Cost** - Estimated API cost
- **Latency** - Time to first token, total duration

**Example Span Attributes**

```json
{
  "llm.vendor": "openai",
  "llm.request.model": "gpt-4o-mini",
  "llm.request.type": "chat",
  "llm.usage.input_tokens": 150,
  "llm.usage.output_tokens": 75,
  "llm.usage.total_tokens": 225,
  "llm.response.finish_reason": "stop",
  "llm.latency_ms": 1847,
  "llm.cost_usd": 0.00225
}
```

### Tool Calls

For agent tool executions:

**Tool Information**

- **Tool Name** - Function name
- **Tool Description** - Docstring
- **Arguments** - Input parameters with values
- **Return Value** - Tool output

**Execution Details**

- **Start Time** - When tool began
- **End Time** - When tool completed
- **Duration** - Total execution time
- **Status** - Success or error
- **Exception** - Error details if failed

**Example**

```json
{
  "tool.name": "get_weather",
  "tool.arguments": "{\"location\": \"San Francisco\"}",
  "tool.result": "{\"temp\": 72, \"condition\": \"sunny\"}",
  "tool.duration_ms": 234,
  "tool.status": "success"
}
```

### Agent Steps

Pixie tracks agent reasoning:

- **Planning** - Agent deciding which tool to use
- **Execution** - Running the tool
- **Reflection** - Analyzing tool results
- **Iteration** - Multiple reasoning loops
- **Completion** - Final answer generation

### User Interactions

For interactive apps:

- **User Input Requests** - When agent requests input
- **User Responses** - What user provided
- **Input Validation** - Whether input was valid
- **Timing** - How long user took to respond

## Langfuse Integration

### What is Langfuse?

[Langfuse](https://langfuse.com) is an open-source LLM engineering platform that provides:

- **Tracing** - Detailed execution traces
- **Prompt Management** - Version and test prompts
- **Evaluation** - Score and compare outputs
- **Cost Tracking** - Monitor API spending
- **Analytics** - Usage patterns and trends

### Enabling Langfuse

Set environment variables:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com  # or self-hosted
```

Then `Agent.instrument_all()` automatically uses Langfuse!

### Langfuse Features with Pixie

**Automatic Logging**

All LLM calls are logged to Langfuse:

```python
Agent.instrument_all()  # Automatically logs to Langfuse

result = await agent.run(query)  # Logged!
```

**Session Tracking**

Group related executions:

```python
from langfuse import Langfuse

langfuse = Langfuse()

@app
async def chat(_: None) -> PixieGenerator[str, str]:
    Agent.instrument_all()

    session_id = langfuse.create_session()

    while True:
        user_msg = yield UserInputRequirement(str)

        # All calls in this session are grouped
        with langfuse.trace(session_id=session_id):
            result = await agent.run(user_msg)

        yield result.output
```

**Custom Metadata**

Add metadata to traces:

```python
from langfuse.decorators import observe

@observe()
@app
async def my_agent(query: str) -> str:
    Agent.instrument_all()

    # Custom metadata
    langfuse.log({
        "user_id": "user123",
        "environment": "production",
        "version": "1.0.0"
    })

    result = await agent.run(query)
    return result.output
```

**Scores and Feedback**

Rate agent outputs:

```python
from langfuse import Langfuse

langfuse = Langfuse()

# After execution
trace_id = get_current_trace_id()

langfuse.score(
    trace_id=trace_id,
    name="quality",
    value=0.95,
    comment="Excellent response"
)
```

## Viewing Traces

### In Pixie Web UI

Access the Debug Screen to view traces:

1. **Open Debug Screen** - Click "Debug" button
2. **View Trace Tree** - Hierarchical span view
3. **Select Span** - Click to see details
4. **Export Traces** - Download for analysis

**Features:**

- Timeline view
- Span filtering
- Attribute inspection
- Event viewing
- JSON export

### In Langfuse Dashboard

If Langfuse is configured:

1. Go to [cloud.langfuse.com](https://cloud.langfuse.com) (or your host)
2. Navigate to "Traces"
3. Find your agent executions
4. Click to see detailed breakdown

**Features:**

- Cost analysis
- Token usage charts
- Latency distributions
- Prompt versions
- User feedback

### Programmatic Access

Access traces via OpenTelemetry API:

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("my_operation") as span:
    # Your code
    span.set_attribute("custom_attr", "value")
    span.add_event("checkpoint", {"detail": "info"})
```

## Custom Instrumentation

### Adding Custom Spans

Wrap your code in spans for more detail:

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@app
async def my_agent(query: str) -> str:
    Agent.instrument_all()

    # Custom span for preprocessing
    with tracer.start_as_current_span("preprocess") as span:
        processed_query = preprocess(query)
        span.set_attribute("original_length", len(query))
        span.set_attribute("processed_length", len(processed_query))

    # Custom span for agent execution
    with tracer.start_as_current_span("agent_execution") as span:
        result = await agent.run(processed_query)
        span.set_attribute("output_length", len(result.output))

    return result.output
```

### Adding Events

Log events within spans:

```python
with tracer.start_as_current_span("data_processing") as span:
    span.add_event("started_loading")

    data = load_data()
    span.add_event("loaded_data", {
        "row_count": len(data),
        "columns": list(data.columns)
    })

    processed = process_data(data)
    span.add_event("processed_data", {
        "output_rows": len(processed)
    })
```

### Custom Attributes

Add metadata to spans:

```python
with tracer.start_as_current_span("operation") as span:
    # Standard attributes
    span.set_attribute("user_id", user_id)
    span.set_attribute("tenant", tenant_name)
    span.set_attribute("version", "1.2.3")

    # Custom attributes
    span.set_attribute("my.custom.attribute", value)
```

## Performance Impact

### Overhead

Instrumentation adds minimal overhead:

- **Memory** - ~1-5MB per active trace
- **CPU** - < 1% in most cases
- **Latency** - < 50ms added to total execution

### Optimization Tips

**1. Selective Instrumentation**

Only instrument what you need:

```python
# In development - full instrumentation
if DEBUG:
    Agent.instrument_all()
else:
    # In production - selective
    Agent.instrument_llm_only()
```

**2. Sampling**

Sample a percentage of requests:

```python
import random

@app
async def my_agent(query: str) -> str:
    # Only instrument 10% of requests
    if random.random() < 0.1:
        Agent.instrument_all()

    result = await agent.run(query)
    return result.output
```

**3. Async Export**

Export traces asynchronously:

```python
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Use batch processor instead of simple processor
processor = BatchSpanProcessor(exporter)
```

## Best Practices

### 1. Always Call `instrument_all()` Early

```python
@app
async def my_agent(query: str) -> str:
    # ✅ Do this first
    Agent.instrument_all()

    # Then your code
    result = await agent.run(query)
    return result.output
```

### 2. Use Descriptive Span Names

```python
# ✅ Good
with tracer.start_as_current_span("fetch_user_data"):
    pass

# ❌ Bad
with tracer.start_as_current_span("operation"):
    pass
```

### 3. Add Relevant Attributes

```python
# ✅ Good - specific and useful
span.set_attribute("query_length", len(query))
span.set_attribute("model_name", "gpt-4o-mini")
span.set_attribute("user_tier", "premium")

# ❌ Bad - too vague
span.set_attribute("data", some_large_object)
```

### 4. Handle Errors in Spans

```python
with tracer.start_as_current_span("risky_operation") as span:
    try:
        result = perform_operation()
    except Exception as e:
        # Record the exception
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        raise
```

### 5. Keep Trace Data Reasonable

```python
# ✅ Good - reasonable size
span.set_attribute("summary", text[:200])

# ❌ Bad - too large
span.set_attribute("full_document", entire_document)
```

## Troubleshooting

### Traces Not Appearing

**Problem:** No traces in UI or Langfuse

**Solutions:**

1. Verify `Agent.instrument_all()` is called
2. Check OpenTelemetry configuration
3. Ensure Langfuse keys are valid (if using)
4. Check for errors in server logs

### Incomplete Traces

**Problem:** Missing spans or data

**Solutions:**

1. Ensure spans are properly closed (use `with` statements)
2. Check for exceptions that interrupt tracing
3. Verify exporters are working
4. Check span limits aren't being exceeded

### Performance Issues

**Problem:** Instrumentation slowing down app

**Solutions:**

1. Use sampling to reduce overhead
2. Switch to batch span processor
3. Reduce span attributes
4. Disable instrumentation in critical paths

### Langfuse Not Working

**Problem:** Data not appearing in Langfuse

**Solutions:**

1. Verify environment variables are set
2. Check Langfuse host is accessible
3. Verify API keys are valid
4. Check Langfuse logs for errors

## Advanced Topics

### Context Propagation

Propagate context across async boundaries:

```python
from opentelemetry import context

# Capture context
ctx = context.get_current()

async def background_task():
    # Restore context in new task
    context.attach(ctx)
    with tracer.start_as_current_span("background_work"):
        # Work happens in same trace
        pass

asyncio.create_task(background_task())
```

### Custom Exporters

Create custom trace exporters:

```python
from opentelemetry.sdk.trace.export import SpanExporter

class DatabaseExporter(SpanExporter):
    def export(self, spans):
        # Export to database
        for span in spans:
            save_to_db(span)
        return SpanExportResult.SUCCESS

# Register
provider.add_span_processor(
    BatchSpanProcessor(DatabaseExporter())
)
```

### Distributed Tracing

Trace across multiple services:

```python
from opentelemetry.propagate import inject, extract

# Service A - inject context
headers = {}
inject(headers)  # Adds trace context to headers

# Make request with headers
response = requests.post(url, headers=headers)

# Service B - extract context
context = extract(request.headers)
with tracer.start_as_current_span("service_b", context=context):
    # This span is part of the same trace!
    pass
```

## Next Steps

- [Examples](https://github.com/yiouli/pixie-examples) - See instrumentation in action
- [API Reference](../api/overview.md) - API documentation
- [OpenTelemetry Docs](https://opentelemetry.io/docs/) - Learn more about OTel
- [Langfuse Docs](https://langfuse.com/docs) - Langfuse features

## Summary

Pixie SDK's instrumentation provides:

- **Automatic Tracing** - Via `Agent.instrument_all()`
- **OpenTelemetry** - Industry-standard observability
- **Langfuse Integration** - Advanced LLM observability
- **Low Overhead** - Minimal performance impact
- **Flexibility** - Custom spans and attributes
- **Actionable Insights** - Debug and optimize agents

With proper instrumentation, you can understand, debug, and optimize your AI agents effectively.
