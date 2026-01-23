# Interactivity Inside Tools

Learn how to emit outputs and request user input from within agent tools using queues and asyncio patterns.

## The Queue Pattern

Use an `asyncio.Queue` to emit outputs from inside tools:

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pixie.sdk import app, PixieGenerator

@dataclass
class Deps:
    """Dependencies with a queue for tool outputs."""
    queue: asyncio.Queue[str | None]

agent = Agent("openai:gpt-4o-mini", deps_type=Deps)

@agent.tool
async def my_tool(ctx: RunContext[Deps]) -> str:
    """A tool that emits progress updates."""
    # Emit progress to queue
    await ctx.deps.queue.put("Starting work...")

    # Do work
    await asyncio.sleep(1)
    await ctx.deps.queue.put("50% complete...")

    # More work
    await asyncio.sleep(1)
    await ctx.deps.queue.put("Finished!")

    return "Tool completed successfully"

@app
async def app_with_tool_output(_: None) -> PixieGenerator[str, None]:
    """App that shows tool outputs."""
    Agent.instrument_all()

    # Create queue
    q = asyncio.Queue[str | None]()
    deps = Deps(queue=q)

    yield "Starting agent..."

    # Run agent in background
    async def run_agent():
        result = await agent.run("Do the work", deps=deps)
        await q.put(None)  # Signal completion

    task = asyncio.create_task(run_agent())

    # Stream tool outputs
    while True:
        update = await q.get()
        if update is None:
            break
        yield update

    yield "Agent finished!"
```

Tool emits progress updates to queue, application reads from queue and yields to user, enabling real-time progress feedback from within tools.

Emit progress updates from long-running tools:

```python
@agent.tool
async def process_large_dataset(ctx: RunContext[Deps], items: int) -> str:
    """Process a large dataset with progress updates."""

    await ctx.deps.queue.put("🔄 Starting processing...")

    for i in range(items):
        # Process item
        await asyncio.sleep(0.1)

        # Report progress every 10 items
        if (i + 1) % 10 == 0:
            progress = (i + 1) / items * 100
            await ctx.deps.queue.put(f"Progress: {progress:.0f}% ({i+1}/{items})")

    await ctx.deps.queue.put("✅ Processing complete!")

    return f"Processed {items} items successfully"
```

## Pattern: Error Reporting

Emit errors or warnings from tools:

```python
@agent.tool
async def validate_data(ctx: RunContext[Deps], data: dict) -> str:
    """Validate data and report issues."""

    issues = []

    if "email" not in data:
        issue = "⚠️ Missing required field: email"
        await ctx.deps.queue.put(issue)
        issues.append(issue)

    if "age" in data and data["age"] < 0:
        issue = "❌ Invalid age: must be positive"
        await ctx.deps.queue.put(issue)
        issues.append(issue)

    if issues:
        return f"Validation failed with {len(issues)} issues"
    else:
        await ctx.deps.queue.put("✅ Validation passed")
        return "Data is valid"
```

## Pattern: Multi-Step Tool Operations

Break complex operations into steps with feedback:

```python
@agent.tool
async def deploy_application(ctx: RunContext[Deps], app_name: str) -> str:
    """Deploy application with step-by-step updates."""

    steps = [
        ("Building application", 2),
        ("Running tests", 1),
        ("Uploading to server", 3),
        ("Starting services", 1),
        ("Verifying deployment", 1),
    ]

    for step_name, duration in steps:
        await ctx.deps.queue.put(f"📦 {step_name}...")
        await asyncio.sleep(duration)
        await ctx.deps.queue.put(f"✅ {step_name} complete")

    return f"Successfully deployed {app_name}"
```

## Best Practices

### 1. Always Signal Completion

```python
async def run_agent():
    result = await agent.run(prompt, deps=deps)
    await q.put(None)  # ✅ Signal completion
```

### 2. Handle Cancellation

```python
task = asyncio.create_task(run_agent())
try:
    while True:
        update = await q.get()
        if update is None:
            break
        yield update
except asyncio.CancelledError:
    task.cancel()  # ✅ Clean up background task
    raise
```

### 3. Use Descriptive Messages

```python
# ✅ Good - clear what's happening
await ctx.deps.queue.put("Processing item 5 of 100...")

# ❌ Bad - unclear
await ctx.deps.queue.put("Working...")
```

### 4. Separate Concerns

Use different queues for different purposes:

```python
@dataclass
class Deps:
    progress_queue: asyncio.Queue[str]
    error_queue: asyncio.Queue[str]
    input_queue: asyncio.Queue[str]
```

### 5. Reset State Between Runs

```python
@agent.tool
async def stateful_tool(ctx: RunContext[Deps]) -> str:
    if ctx.deps.should_reset:
        ctx.deps.counter = 0
        ctx.deps.should_reset = False
    # ... rest of tool
```

## Common Pitfalls

### Queue Never Completes

**Problem:** App hangs forever

**Solution:** Always send `None` sentinel:

```python
async def run_agent():
    try:
        result = await agent.run(prompt, deps=deps)
    finally:
        await q.put(None)  # ✅ Always signal completion
```

### Race Conditions

**Problem:** Messages arrive out of order

**Solution:** Use single queue or synchronization:

```python
# ✅ Single queue for ordering
@dataclass
class Deps:
    queue: asyncio.Queue[tuple[str, str]]  # (type, message)

await ctx.deps.queue.put(("progress", "50% done"))
await ctx.deps.queue.put(("error", "Warning: ..."))
```

### Memory Leaks

**Problem:** Queue grows unbounded

**Solution:** Set max size and handle full queue:

```python
q = asyncio.Queue(maxsize=100)  # ✅ Limit queue size

try:
    await asyncio.wait_for(q.put(msg), timeout=1.0)
except asyncio.TimeoutError:
    # Handle queue full
    pass
```

## Troubleshooting

### Tool Outputs Not Appearing

**Problem:** Tool puts messages in queue but they don't show up

**Solution:** Ensure you're reading from queue:

```python
# ✅ Correct
while True:
    update = await q.get()
    if update is None:
        break
    yield update
```

### Application Blocks Forever

**Problem:** Application doesn't complete

**Solution:** Verify sentinel is sent and task completes:

```python
async def run_agent():
    result = await agent.run(prompt, deps=deps)
    await q.put(None)  # Must signal completion

# Wait for task
await task  # Or handle properly
```

### User Input Not Reaching Tool

**Problem:** Tool waits forever for input

**Solution:** Verify input queue connection:

```python
# Tool
response = await ctx.deps.input_queue.get()

# Application
user_input = yield InputRequired(str)
await input_q.put(user_input)  # Must put in queue
```

## Next Steps

- [Make Your Application Interactive](./make-your-application-interactive.md) - Basic interactivity
- [Use Structured I/O](./use-structured-io.md) - Type-safe I/O
- [Examples](https://github.com/yiouli/pixie-examples) - More interactive examples
