# Register Your Application

Learn how to register your AI application with Pixie using the `@app` decorator and understand handler requirements.

## The `@app` Decorator

All Pixie applications must be decorated with `@app`. This decorator enables automatic discovery, observability, and interactivity:

```python
from pixie import app

@app
async def my_application(input_data: str) -> str:
    """My application description."""
    return f"Processed: {input_data}"
```

## Handler Function Requirements

### Async function or generator with 0-1 argument

All handler functions must be either an async function or async generator that takes 0-1 argument.
Type-checking would complain if you try to register an unsupported handler.

```python
# ✅ OK
@app
async def my_app(query: str) -> str:
    ...

# ✅ OK
@app
async def my_app() -> PixieGenerator[str, int]:
    ...

# ❌ Not supported - sync function
@app
def my_app(query: str) -> str:
    ...

# ❌ Not supported, multiple args
@app
async def my_app(a: int, b: int) -> str:  # Missing 'async'
    ...
```

### Supported data types

For the type of the handler's argument, the return value for async function, and the yield and receive types for async generator, only valid JSON types or Pydantic model types are allowed.

Valid JSON types are simple types such as `str`, `int`, `bool` etc, or any composite types with `list` and `dict[str, JsonType]`

### Function vs Generator

Pixie supports two handler patterns depending on your application's needs:

#### 1. Async Function Pattern

For simple, single-turn applications that return a result:

```python
from pixie import app

@app
async def weather_lookup(city: str) -> str:
    """Look up weather for a city."""
    # Do work...
    return f"Weather in {city}: Sunny"
```

**Use when:**

- Single request/response interaction
- No streaming outputs
- No mid-execution user input needed

#### 2. Async Generator Pattern

For multi-turn, interactive, or streaming applications:

```python
from pixie import app, PixieGenerator

@app
async def chat() -> PixieGenerator[str, None]:
    """Streaming chatbot."""
    yield "Hello! Thinking..."
    yield "Here's my response"
```

**Use when:**

- Streaming outputs to users
- Interactive conversations
- Progress updates during execution
- User input needed mid-execution

## Initial Input vs Iteractively Received Input

For a generator handler, you can get input data from user in two ways:

1. The initial input data that will be passed to your handler on initial connection
2. The interactively received input that you get from `yield` statements

You can specify different types for these two types of input.
