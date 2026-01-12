---
sidebar_position: 1
---

# API Reference Overview

Complete API reference for Pixie SDK. This section documents all public classes, functions, and decorators.

## Core Decorators

### `@app`

The main decorator for creating Pixie applications.

```python
from pixie import app

@app
async def my_agent(query: str) -> str:
    """My agent description."""
    ...
```

**Signature:**

```python
def app(func: Callable) -> Callable
```

**Parameters:**

- `func` - An async function or async generator to decorate

**Returns:**

- The decorated function with Pixie registration

**Requirements:**

- Function must be `async def`
- Function must have type annotations for parameters and return type
- For interactive apps, return type must be `PixieGenerator[YieldType, SendType]`

**Example:**

```python
from pixie import app, PixieGenerator, UserInputRequirement

@app
async def simple_app(query: str) -> str:
    """Simple app that returns a string."""
    return f"You asked: {query}"

@app
async def interactive_app(_: None) -> PixieGenerator[str, str]:
    """Interactive app with user input."""
    yield "Hello!"
    response = yield UserInputRequirement(str)
    yield f"You said: {response}"
```

## Type Definitions

### `PixieGenerator`

Generic type for interactive applications that yield values and receive input.

```python
from pixie import PixieGenerator

async def my_app() -> PixieGenerator[YieldType, SendType]:
    ...
```

**Type Parameters:**

- `YieldType` - Type of values yielded to the client
- `SendType` - Type of values received from user input

**Usage:**

```python
# String output, string input
PixieGenerator[str, str]

# Structured output, no input
PixieGenerator[MyModel, None]

# String output, structured input
PixieGenerator[str, UserResponse]
```

### `UserInputRequirement`

Class used to request input from users in interactive applications.

```python
from pixie import UserInputRequirement

user_input = yield UserInputRequirement(str)
```

**Signature:**

```python
class UserInputRequirement(Generic[T]):
    def __init__(self, input_type: Type[T])
```

**Parameters:**

- `input_type` - The type of input expected from the user

**Returns:**

- When yielded, returns a value of type `input_type`

**Examples:**

```python
# Request string
text = yield UserInputRequirement(str)

# Request integer
number = yield UserInputRequirement(int)

# Request structured data
from pydantic import BaseModel

class UserData(BaseModel):
    name: str
    age: int

data = yield UserInputRequirement(UserData)
```

## Instrumentation

### `Agent.instrument_all()`

Enable automatic instrumentation for PydanticAI agents.

```python
from pydantic_ai import Agent

Agent.instrument_all()
```

**Signature:**

```python
@classmethod
def instrument_all(cls) -> None
```

**What it does:**

- Initializes OpenTelemetry tracer
- Registers Langfuse integration (if configured)
- Instruments PydanticAI agent operations
- Captures LLM calls, tool usage, and agent steps

**When to call:**

- At the start of your `@app` function
- Before creating or running agents
- Only needs to be called once per execution

**Example:**

```python
from pixie import app
from pydantic_ai import Agent

agent = Agent("openai:gpt-4o-mini")

@app
async def my_agent(query: str) -> str:
    Agent.instrument_all()  # Enable tracing
    result = await agent.run(query)
    return result.output
```

## Server Functions

### `create_app()`

Create a FastAPI application for running Pixie server.

```python
from pixie.server import create_app

app = create_app(app_dirs=["examples/"])
```

**Signature:**

```python
def create_app(
    app_dirs: list[str] = None,
    enable_graphiql: bool = True,
    additional_queries: list = None
) -> FastAPI
```

**Parameters:**

- `app_dirs` - List of directories to scan for `@app` decorated functions
- `enable_graphiql` - Whether to enable GraphiQL interface
- `additional_queries` - Additional Strawberry GraphQL query types

**Returns:**

- FastAPI application instance

**Example:**

```python
from pixie.server import create_app
import uvicorn

app = create_app(
    app_dirs=["examples/", "agents/"],
    enable_graphiql=True
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Command-Line Interface

Run the Pixie server from the command line.

**Basic usage:**

```bash
pixie
```

**Options:**

```bash
pixie --help                    # Show help
pixie --port 8080              # Change port
pixie --host 0.0.0.0           # Change host
pixie --app-dir agents/        # Specify app directory
pixie --reload                  # Enable auto-reload
pixie --log-level debug        # Set log level
pixie --no-graphiql            # Disable GraphiQL
```

**Environment variables:**

```bash
PIXIE_PORT=8080
PIXIE_HOST=0.0.0.0
PIXIE_APP_DIR=agents/
PIXIE_LOG_LEVEL=info
```

## GraphQL API

### Queries

#### `apps`

Query all available applications.

**Schema:**

```graphql
query {
  apps {
    name
    description
    inputSchema
    outputSchema
    isInteractive
  }
}
```

**Response:**

```json
{
  "data": {
    "apps": [
      {
        "name": "weather",
        "description": "Get weather information",
        "inputSchema": "{\"type\": \"string\"}",
        "outputSchema": "{\"type\": \"string\"}",
        "isInteractive": false
      }
    ]
  }
}
```

### Subscriptions

#### `run`

Execute an application and stream results.

**Schema:**

```graphql
subscription {
  run(name: String!, inputData: String) {
    runId
    status
    data
    requiresInput
    error
  }
}
```

**Parameters:**

- `name` - Application name
- `inputData` - JSON string of input data (optional for apps with no input)

**Response fields:**

- `runId` - Unique run identifier
- `status` - Execution status: "running", "waiting_input", "completed", "error"
- `data` - Output data (when available)
- `requiresInput` - Whether user input is required
- `error` - Error message (if status is "error")

**Example:**

```graphql
subscription {
  run(name: "weather", inputData: "{\"location\": \"Tokyo\"}") {
    runId
    status
    data
  }
}
```

### Mutations

#### `sendInput`

Send user input to a waiting application.

**Schema:**

```graphql
mutation {
  sendInput(runId: String!, input: String!) {
    success
    error
  }
}
```

**Parameters:**

- `runId` - The run ID from the subscription
- `input` - JSON string of user input

**Response:**

- `success` - Whether input was accepted
- `error` - Error message (if not successful)

**Example:**

```graphql
mutation {
  sendInput(runId: "abc-123", input: "{\"message\": \"Hello\"}") {
    success
  }
}
```

## Schema Types

### App Schema Types

Pixie automatically converts Python type annotations to JSON schemas.

#### Primitive Types

| Python Type | JSON Schema Type |
| ----------- | ---------------- |
| `str`       | `"string"`       |
| `int`       | `"integer"`      |
| `float`     | `"number"`       |
| `bool`      | `"boolean"`      |
| `None`      | `"null"`         |

#### Complex Types

| Python Type         | JSON Schema Type    |
| ------------------- | ------------------- |
| `list[T]`           | `"array"`           |
| `dict[K, V]`        | `"object"`          |
| `Optional[T]`       | Union with `"null"` |
| `Union[A, B]`       | `"anyOf"`           |
| `Literal["a", "b"]` | `"enum"`            |

#### Pydantic Models

Pydantic models are converted to JSON Schema objects:

```python
from pydantic import BaseModel, Field

class UserQuery(BaseModel):
    """User query model."""

    query: str = Field(description="The search query")
    max_results: int = Field(default=10, ge=1, le=100)
```

Becomes:

```json
{
  "type": "object",
  "title": "UserQuery",
  "description": "User query model.",
  "properties": {
    "query": {
      "type": "string",
      "description": "The search query"
    },
    "max_results": {
      "type": "integer",
      "default": 10,
      "minimum": 1,
      "maximum": 100
    }
  },
  "required": ["query"]
}
```

## Utility Functions

### `test_app()`

Test helper for running apps programmatically (useful for unit tests).

```python
from pixie.utils import test_app

async def test_my_app():
    runner = test_app(my_agent)
    response = await runner.start("test query")
    assert "expected" in response
```

**Signature:**

```python
def test_app(app_func: Callable) -> AppTestRunner
```

**Returns:**

- `AppTestRunner` instance for testing

**Methods:**

- `start(input_data)` - Start the app with input
- `send(user_input)` - Send user input
- `get_last_message()` - Get last output
- `is_complete()` - Check if app finished
- `wait_completion()` - Wait for app to finish

## OpenTelemetry Types

### Span

Represents a single traced operation.

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)
span = tracer.start_span("operation_name")
```

**Methods:**

- `set_attribute(key, value)` - Add attribute to span
- `add_event(name, attributes)` - Add event to span
- `record_exception(exception)` - Record an exception
- `set_status(status)` - Set span status

### Tracer

Creates and manages spans.

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("my_operation") as span:
    # Your code here
    span.set_attribute("key", "value")
```

## Error Types

### `ValidationError`

Raised when input validation fails (from Pydantic).

```python
from pydantic import ValidationError

try:
    result = await app.run(invalid_input)
except ValidationError as e:
    print(f"Validation error: {e.errors()}")
```

### `ExecutionError`

Raised when app execution fails.

```python
from pixie.exceptions import ExecutionError

try:
    result = await app.run(input)
except ExecutionError as e:
    print(f"Execution failed: {e}")
```

## Best Practices

### Type Annotations

Always use type annotations:

```python
# ✅ Good
@app
async def my_app(query: str) -> str:
    ...

# ❌ Bad - no type hints
@app
async def my_app(query):
    ...
```

### Docstrings

Provide clear docstrings:

```python
@app
async def my_app(query: str) -> str:
    """Process user queries with AI.

    This app uses GPT-4 to answer questions about weather.
    """
    ...
```

### Error Handling

Handle errors gracefully:

```python
@app
async def my_app(query: str) -> str:
    Agent.instrument_all()

    try:
        result = await agent.run(query)
        return result.output
    except Exception as e:
        return f"Error: {str(e)}"
```

## Next Steps

- [Examples Repository](https://github.com/yiouli/pixie-examples) - See APIs in use
- [Tutorial](../tutorial/setup.md) - Step-by-step guides
- [Concepts](../concepts/architecture.md) - Understand the internals

## Additional Resources

- **PydanticAI Docs** - https://ai.pydantic.dev
- **OpenTelemetry Python** - https://opentelemetry.io/docs/languages/python/
- **Langfuse Docs** - https://langfuse.com/docs
- **FastAPI Docs** - https://fastapi.tiangolo.com
- **Strawberry GraphQL** - https://strawberry.rocks

## Contributing

To contribute to Pixie SDK API documentation:

1. Fork the repository
2. Update docstrings in source code
3. Regenerate API docs
4. Submit a pull request

See [Contributing Guide](https://github.com/yiouli/pixie-sdk-py/blob/main/CONTRIBUTING.md) for details.
