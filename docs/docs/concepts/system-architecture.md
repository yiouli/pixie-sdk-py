# System Architecture

Understand the architecture of the Pixie SDK: application registration, GraphQL server, trace emission, and framework integrations.

## Overview

The Pixie SDK consists of several interconnected components:

1. **Application Registry** - Discovers and registers applications
2. **GraphQL Server** - Exposes applications via API and handles execution
3. **Execution Context** - Manages pause/resume and status updates
4. **Trace Emission** - Captures and streams trace data
5. **Framework Instrumentation** - Integrates with AI frameworks

```
┌─────────────────────────────────────────────────────────────┐
│                    Web UI (Browser)                          │
│                   https://gopixie.ai                         │
└────────────────┬────────────────────────────────────────────┘
                 │ WebSocket (GraphQL Subscription)
                 │
┌────────────────▼────────────────────────────────────────────┐
│               Pixie Server (FastAPI)                         │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐   │
│  │   GraphQL    │  │  Discovery  │  │ Instrumentation  │   │
│  │   Endpoint   │  │  & Registry │  │ (OpenTelemetry)  │   │
│  └──────┬───────┘  └──────┬──────┘  └────────┬─────────┘   │
│         │                  │                   │             │
│  ┌──────▼──────────────────▼───────────────────▼─────────┐  │
│  │       Execution Context & Status Queue                │  │
│  └──────┬────────────────────────────────────────────────┘  │
│         │                                                    │
└─────────┼────────────────────────────────────────────────────┘
          │
┌─────────▼────────────────────────────────────────────────────┐
│              Your AI Application                             │
│  ┌────────────┐  ┌─────────────┐  ┌────────────────────┐   │
│  │   Agent    │  │    Tools    │  │  LLM Framework     │   │
│  └────────────┘  └─────────────┘  └────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## Application Registry

### Discovery

When the server starts, it discovers Python files with `@app` decorated functions:

- Scans current directory recursively
- Skips files starting with `_`
- Skips virtual environments (`.venv`, `site-packages`)
- Loads modules and extracts metadata

### Registration

The `@app` decorator registers functions:

```python
def app(func):
    """Decorator to register an application."""
    # Extract type hints, docstrings
    # Create registry item with metadata
    # Store in global registry
    return func
```

Registry stores: function name, type information (input/output schemas), descriptions, stream handler.

## GraphQL Server

### Schema

The server exposes a GraphQL API:

```graphql
type Query {
  applications: [Application!]!          # List all apps
  application(id: String!): Application   # Get app details
}

type Subscription {
  run(id: String!, inputData: JSON): AppRunUpdate!  # Run app, stream updates
}

type Mutation {
  sendInput(runId: String!, inputData: JSON!): Boolean!  # Send user input
  pauseRun(runId: String!, ...): String!                 # Pause execution
  resumeRun(runId: String!): Boolean!                    # Resume execution
}
```

### Execution Flow

1. Client subscribes to `run`
2. Server initializes execution context
3. Application runs in background task
4. Updates flow through status queue
5. Subscription streams updates to client

## Execution Context

Each running application has an execution context that:

- Tracks running applications
- Queues status updates
- Enables pause/resume functionality
- Handles cancellation

## Trace Emission

### OpenTelemetry Integration

Pixie uses OpenTelemetry to capture traces from AI frameworks:

1. **Instrumentation** - Hooks into framework calls
2. **Span Creation** - Creates spans for LLM calls, tool usage, etc.
3. **Context Propagation** - Links spans into traces
4. **Export** - Emits to Langfuse collector

### Langfuse Bridge

Langfuse converts OpenTelemetry spans to Langfuse format and emits to the web UI.

## Framework Instrumentation

Pixie instruments major AI frameworks automatically:

- **Pydantic AI**: `Agent.instrument_all()`
- **LangChain/LangGraph**: Langfuse callback handlers
- **OpenAI Agents SDK**: OpenTelemetry instrumentation
- **Google ADK**: OpenTelemetry instrumentation
- **CrewAI/DSpy**: OpenTelemetry instrumentation

Instrumentation captures: LLM calls (model, tokens, latency), tool usage, agent steps, errors.
def emit_status_update(
status: AppRunStatus | None,
user_input_requirement: InputRequired | None = None,
data: Optional[JsonValue] = None,
breakpt: Optional[BreakpointDetail] = None,
trace: Optional[dict] = None,
) -> None:
"""Emit status update to subscription."""
ctx = \_execution_context.get()
if ctx:
update = AppRunUpdate(
run_id=ctx.run_id,
status=status,
user_input_schema=extract_schema(user_input_requirement),
data=data,
breakpoint=breakpt,
trace=trace,
) # Put in sync queue (works from any thread)
ctx.status_queue.sync_q.put(update)

````

### Pause/Resume

Execution can be paused at breakpoints:

```python
def set_breakpoint(run_id: str, timing: BreakpointTiming, types: Sequence[BreakpointType]):
    """Configure breakpoints for a run."""
    ctx = _active_runs[run_id]
    ctx.breakpoint_config = BreakpointConfig(
        id=str(uuid4()),
        timing=timing,
        breakpoint_types=types,
    )

def wait_for_resume() -> None:
    """Block until resume is called."""
    ctx = _execution_context.get()
    if ctx:
        ctx.resume_event.wait()  # Blocks thread
        ctx.resume_event.clear()

def resume_run(run_id: str) -> bool:
    """Resume paused execution."""
    ctx = _active_runs.get(run_id)
    if ctx:
        ctx.resume_event.set()  # Unblock thread
        return True
    return False
````

## Component 4: Trace Emission

### Trace Flow

Traces are captured via OpenTelemetry and emitted through Langfuse:

```
┌──────────────────┐
│  AI Framework    │  (LangChain, Pydantic AI, etc.)
│  makes LLM call  │
└────────┬─────────┘
         │ OpenTelemetry span created
         │
┌────────▼─────────────────────────────────┐
│  LangfuseSpanProcessor                   │
│  (BatchSpanProcessor)                    │
│  - Captures span events                  │
│  - Checks for breakpoints                │
│  - Emits to execution context            │
└────────┬─────────────────────────────────┘
         │
         │ Emit trace event
         │
┌────────▼─────────────────────────────────┐
│  Execution Context Status Queue          │
└────────┬─────────────────────────────────┘
         │
         │ Stream to subscription
         │
┌────────▼─────────────────────────────────┐
│  GraphQL Subscription                    │
│  (to Web UI)                             │
└──────────────────────────────────────────┘
```

### Span Processor

The `LangfuseSpanProcessor` intercepts spans:

```python
# langfuse/_client/span_processor.py
class LangfuseSpanProcessor(BatchSpanProcessor):
    """Custom span processor for Langfuse integration."""

    def on_start(self, span: Span, parent_context: Context):
        """Called when span starts."""
        super().on_start(span, parent_context)
        self._check_breakpoint(span, is_before=True)

    def on_end(self, span: ReadableSpan):
        """Called when span ends."""
        self._check_breakpoint(span, is_before=False)
        super().on_end(span)

        # Emit trace data to execution context
        self._emit_trace_event(span)

    def _emit_trace_event(self, span: ReadableSpan):
        """Emit trace to execution context queue."""
        trace_data = {
            "span_name": span.name,
            "span_id": format_span_id(span.context.span_id),
            "trace_id": format_trace_id(span.context.trace_id),
            "start_time": span.start_time,

```
