---
sidebar_position: 1
---

# Architecture

Understanding the architecture of Pixie SDK helps you build better AI applications and troubleshoot issues effectively.

## Overview

Pixie SDK consists of three main components that work together to provide observability and control for AI agents:

```
┌──────────────────────────────────────────────────────────┐
│                     Your Application                      │
│  ┌────────────────────────────────────────────────────┐  │
│  │  @app decorated functions (Agent Applications)     │  │
│  │  - PydanticAI, LangChain, LangGraph agents        │  │
│  │  - Tools and workflows                             │  │
│  └─────────────────┬────────────────────────────────── │  │
│                    │                                      │
│                    ├── Automatic Instrumentation         │
│                    │   (OpenTelemetry + Langfuse)         │
│                    │                                      │
└────────────────────┼──────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│                    Pixie Server                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  GraphQL API Layer                                 │  │
│  │  - Query apps                                       │  │
│  │  - Run subscriptions                               │  │
│  │  - Manage execution                                │  │
│  └─────────────────┬────────────────────────────────── │  │
│                    │                                      │
│  ┌────────────────▼────────────────────────────────┐    │
│  │  Execution Engine                                │    │
│  │  - Discover apps via @app decorator             │    │
│  │  - Manage async execution                        │    │
│  │  - Handle user input requests                    │    │
│  │  - Collect traces                                │    │
│  └──────────────────────────────────────────────────┘    │
│                                                           │
└───────────────────┬───────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────┐
│                    Web UI / Clients                      │
│  ┌────────────────────────────────────────────────────┐  │
│  │  React Web Interface                               │  │
│  │  - Select and run apps                            │  │
│  │  - Interactive chat interface                     │  │
│  │  - Debug screen with traces                       │  │
│  │  - Real-time streaming                            │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Custom Clients (via GraphQL)                     │  │
│  │  - Mobile apps                                     │  │
│  │  - CLI tools                                       │  │
│  │  - Third-party integrations                       │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Pixie SDK (Python Library)

The SDK is what you install and import in your Python code.

#### Key Responsibilities

- **App Registration** - Discovers functions decorated with `@app`
- **Type Handling** - Processes input/output schemas from type annotations
- **Instrumentation** - Integrates with OpenTelemetry and Langfuse
- **Context Management** - Manages execution context for interactive apps

#### Core Modules

**`pixie.registry`**

- Discovers and registers `@app` decorated functions
- Maintains a registry of available applications
- Extracts schema information from type hints

**`pixie.schema`**

- Converts Pydantic models to GraphQL schemas
- Handles JSON serialization/deserialization
- Validates input data

**`pixie.execution_context`**

- Manages async generator execution
- Handles `UserInputRequirement` flow
- Coordinates between app and server

**`pixie.types`**

- Defines `PixieGenerator` type
- Defines `UserInputRequirement` class
- Core type definitions

#### The `@app` Decorator

When you decorate a function with `@app`:

```python
@app
async def my_agent(query: str) -> str:
    ...
```

Behind the scenes:

1. Function is registered in the global registry
2. Input/output types are extracted
3. Schema is generated for GraphQL
4. Function is wrapped for instrumentation

### 2. Pixie Server

The server is a FastAPI application that manages agent execution and provides the GraphQL API.

#### Key Responsibilities

- **App Discovery** - Scans directories for `@app` decorated functions
- **GraphQL API** - Exposes apps via standardized API
- **Execution Management** - Runs apps asynchronously
- **Stream Handling** - Manages real-time streaming
- **Trace Collection** - Aggregates OpenTelemetry data

#### Architecture

**FastAPI Application**

```python
from pixie.server import create_app

app = create_app(
    app_dirs=["examples/"],
    enable_graphiql=True
)
```

**GraphQL Schema (Strawberry)**

- `Query.apps` - List available applications
- `Subscription.run` - Execute an application
- `Mutation.sendInput` - Send user input to running app

**Execution Engine**

- Async execution using Python's asyncio
- Generator management for interactive apps
- State tracking (running, paused, completed)
- Error handling and recovery

#### Server Startup Flow

1. **Initialize** - Create FastAPI app
2. **Discover** - Scan specified directories for `@app` functions
3. **Register** - Build GraphQL schema from discovered apps
4. **Listen** - Start server on specified host/port
5. **Serve** - Handle GraphQL queries and UI requests

### 3. Web UI

The web UI is a React application (built with Next.js) that provides a visual interface.

#### Key Features

- **App Selection** - Browse and select available agents
- **Chat Interface** - Interactive conversation UI
- **Debug Screen** - Visualize execution traces
- **Real-Time Updates** - WebSocket-based streaming
- **Trace Explorer** - Hierarchical trace tree view

#### Tech Stack

- **Next.js 14** - React framework
- **GraphQL** (via Apollo Client) - API communication
- **TailwindCSS** - Styling
- **React JSON View** - Trace visualization
- **Zustand** - State management

#### UI Architecture

**Pages**

- `/` - App selection screen
- `/app/[name]` - Chat interface for specific app
- `/debug` - Debug screen with traces

**Components**

- `AppList` - Displays available apps
- `ChatInterface` - Handles message flow
- `DebugScreen` - Shows trace tree
- `SchemaInput` - Auto-generated forms for structured input

## Data Flow

### Simple Request Flow

1. **User submits query** via Web UI
2. **GraphQL subscription** sent to server:
   ```graphql
   subscription {
     run(name: "my_agent", inputData: "query") {
       data
       status
     }
   }
   ```
3. **Server executes app** asynchronously
4. **Agent runs**, making LLM calls
5. **Traces collected** via OpenTelemetry
6. **Result streamed** back to client
7. **UI displays** output and traces

### Interactive Request Flow

1. **User starts app** via Web UI
2. **App yields initial message**
3. **Server streams** message to client
4. **UI displays** message
5. **App requests input** via `UserInputRequirement`
6. **Server notifies** client that input is needed
7. **UI prompts** user for input
8. **User provides input**
9. **Client sends input** via GraphQL mutation
10. **Server forwards** input to app
11. **App continues** execution
12. **Repeat** steps 2-11 as needed
13. **App completes**, final result sent

## Instrumentation Architecture

### OpenTelemetry Integration

Pixie uses OpenTelemetry for distributed tracing:

```
┌─────────────────────────────────────────────────────┐
│  Your Agent Code                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  Agent.instrument_all()                       │  │
│  │  ↓                                             │  │
│  │  Creates OpenTelemetry Tracer                 │  │
│  └───────────────┬───────────────────────────────┘  │
└──────────────────┼──────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  OpenTelemetry SDK                                   │
│  ┌────────────────────────────────────────────────┐  │
│  │  Span Processor                                │  │
│  │  - Captures span data                          │  │
│  │  - Adds attributes                             │  │
│  │  - Tracks timing                               │  │
│  └────────────────┬───────────────────────────────┘  │
└───────────────────┼──────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────┐
│  Exporters                                           │
│  ┌────────────────────┬─────────────────────────┐    │
│  │  Memory Exporter   │  Langfuse Exporter      │    │
│  │  (for UI display)  │  (optional, for cloud)  │    │
│  └────────────────────┴─────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

### Trace Hierarchy

Traces form a hierarchical structure:

```
Root Span: my_agent execution
├── Span: PydanticAI agent run
│   ├── Span: LLM call (OpenAI)
│   │   ├── Event: prompt
│   │   ├── Event: response
│   │   └── Attributes: model, tokens, cost
│   ├── Span: Tool call (get_weather)
│   │   ├── Event: tool input
│   │   ├── Event: tool output
│   │   └── Attributes: tool_name, duration
│   └── Span: Agent reasoning
└── Span: Result serialization
```

### Langfuse Integration

Pixie integrates with Langfuse for additional observability:

- **Automatic logging** - LLM calls, costs, latency
- **Session tracking** - Group related executions
- **User feedback** - Rate agent responses
- **Cost analysis** - Track API spending
- **A/B testing** - Compare agent versions

## Scalability Considerations

### Concurrent Execution

The server handles multiple concurrent app executions:

```python
# Each run gets its own async task
async def run_app(app_name: str, input_data: any):
    task = asyncio.create_task(execute_app(app_name, input_data))
    return task
```

### State Management

- **Stateless Apps** - Each run is independent
- **Interactive Apps** - State maintained per session
- **Context Isolation** - No shared state between runs

### Resource Limits

Consider these limits in production:

- **Memory** - Each active run consumes memory
- **LLM Rate Limits** - Respect API quotas
- **Connection Limits** - WebSocket connections for UI
- **Trace Storage** - In-memory trace storage (for now)

## Security Model

### Current Security Features

- **No Built-in Auth** - Designed for development/internal use
- **CORS** - Configurable for client access
- **Input Validation** - Via Pydantic models
- **Error Isolation** - Exceptions don't crash server

### Production Considerations

For production deployments, consider adding:

- **Authentication** - JWT tokens, OAuth
- **Authorization** - Role-based access control
- **Rate Limiting** - Prevent abuse
- **Encryption** - HTTPS/TLS
- **Audit Logging** - Track all operations
- **Secrets Management** - Secure API key storage

## Deployment Architecture

### Development

```
┌────────────────────┐
│  Local Machine     │
│  ├── Python Agent  │
│  ├── Pixie Server  │
│  └── Web UI        │
│  (http://localhost:8000)
└────────────────────┘
```

### Production (Recommended)

```
┌───────────────────────────────────────────────────┐
│  Cloud Environment                                 │
│                                                    │
│  ┌──────────────────┐    ┌─────────────────────┐ │
│  │  Load Balancer   │───▶│  Pixie Server       │ │
│  │  (HTTPS)         │    │  (Container/VM)     │ │
│  └──────────────────┘    │  - Apps             │ │
│                           │  - GraphQL API      │ │
│                           └─────────────────────┘ │
│                                                    │
│  ┌──────────────────┐                             │
│  │  Static Hosting  │                             │
│  │  (Web UI)        │                             │
│  │  - S3/CDN        │                             │
│  └──────────────────┘                             │
│                                                    │
│  ┌──────────────────┐                             │
│  │  External        │                             │
│  │  - Langfuse      │                             │
│  │  - LLM APIs      │                             │
│  └──────────────────┘                             │
└───────────────────────────────────────────────────┘
```

## Extension Points

### Custom Exporters

Add custom trace exporters:

```python
from opentelemetry.sdk.trace.export import SpanExporter

class CustomExporter(SpanExporter):
    def export(self, spans):
        # Send spans to custom backend
        pass

# Register exporter
from opentelemetry import trace
provider = trace.get_tracer_provider()
provider.add_span_processor(
    BatchSpanProcessor(CustomExporter())
)
```

### Custom GraphQL Schema

Extend the GraphQL schema:

```python
import strawberry
from pixie.server import create_app

@strawberry.type
class CustomQuery:
    @strawberry.field
    def my_custom_field(self) -> str:
        return "custom data"

app = create_app(
    app_dirs=["examples/"],
    additional_queries=[CustomQuery]
)
```

### Custom UI

Build a custom UI using the GraphQL API:

```typescript
import { ApolloClient, gql } from "@apollo/client";

const client = new ApolloClient({
  uri: "http://localhost:8000/graphql",
});

// Query apps
const { data } = await client.query({
  query: gql`
    query {
      apps {
        name
        description
      }
    }
  `,
});

// Run app
const subscription = client.subscribe({
  query: gql`
    subscription {
      run(name: "my_agent", inputData: "query") {
        data
        status
      }
    }
  `,
});
```

## Performance Characteristics

### Latency

- **App discovery** - O(n) where n = number of Python files
- **GraphQL query** - O(1) for app list
- **App execution** - Depends on LLM latency (typically 1-5s)
- **Streaming** - Near real-time (< 100ms overhead)

### Throughput

- **Concurrent runs** - Limited by Python's GIL and asyncio
- **Recommended** - Use multiple server instances for high load
- **Bottleneck** - Usually LLM API rate limits, not Pixie

### Memory

- **Base server** - ~100MB
- **Per active run** - ~10-50MB
- **Traces** - ~1MB per complex trace
- **Recommendation** - Implement trace pruning for long-running servers

## Next Steps

- [Instrumentation](./instrumentation.md) - Deep dive into tracing
- [Examples](https://github.com/yiouli/pixie-examples) - See real implementations
- [API Reference](../api/overview.md) - API documentation

## Summary

Pixie SDK's architecture provides:

- **Separation of Concerns** - SDK, Server, UI are independent
- **Flexibility** - Easy to extend and customize
- **Developer Experience** - Simple `@app` decorator
- **Observability** - Built-in tracing with OpenTelemetry
- **Scalability** - Async execution and streaming
- **Standardization** - GraphQL API for any client

Understanding this architecture helps you build robust, observable AI applications with Pixie SDK.
