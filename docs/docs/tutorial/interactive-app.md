---
sidebar_position: 4
---

# Interactive Apps

Learn how to build multi-turn, interactive applications that can request input from users during execution.

## What are Interactive Apps?

Interactive apps are applications that can:

- Send multiple messages over time
- Request input from users mid-execution
- Maintain conversation state
- Stream responses in real-time

They use **async generators** instead of simple async functions.

## Basic Interactive App

### Simple Chat Example

```python
from pixie import app, PixieGenerator, UserInputRequirement
from pydantic_ai import Agent

chatbot = Agent(
    "openai:gpt-4o-mini",
    system_prompt="You are a friendly chatbot."
)

@app
async def chat(_: None) -> PixieGenerator[str, str]:
    """Interactive chatbot that maintains conversation."""
    Agent.instrument_all()

    # Send welcome message
    yield "Hello! How can I help you today?"

    # Request user input
    user_message = yield UserInputRequirement(str)

    # Process and respond
    result = await chatbot.run(user_message)
    yield result.output
```

### Key Components

1. **Return Type**: `PixieGenerator[YieldType, SendType]`

   - `YieldType` - Type of data sent to user (e.g., `str`)
   - `SendType` - Type of data received from user (e.g., `str`)

2. **`yield` statement**: Send data to the user

3. **`UserInputRequirement`**: Request input from the user
   ```python
   user_input = yield UserInputRequirement(str)
   ```

## Multi-Turn Conversations

### Continuous Chat Loop

```python
from pixie import app, PixieGenerator, UserInputRequirement
from pydantic_ai import Agent, ModelMessage, ModelRequest

chatbot = Agent("openai:gpt-4o-mini")

@app
async def chat(_: None) -> PixieGenerator[str, str]:
    """Multi-turn chatbot with conversation history."""
    Agent.instrument_all()

    yield "Hello! I'm here to help. Type 'exit' to quit."

    # Maintain conversation history
    history: list[ModelMessage] = []

    while True:
        # Get user input
        user_msg = yield UserInputRequirement(str)

        # Check for exit
        if user_msg.lower() in {"exit", "quit", "bye"}:
            yield "Goodbye! Have a great day!"
            break

        # Run agent with history
        result = await chatbot.run(
            user_msg,
            message_history=history
        )

        # Update history
        history.append(ModelRequest.user_text_prompt(user_msg))
        history.append(result.response)

        # Send response
        yield result.output
```

### How It Works

1. **Initial yield** - Send welcome message
2. **Loop** - Continuously process user input
3. **History** - Maintain context across turns
4. **Exit condition** - Allow user to end conversation

## Structured Interactive Apps

### Using Pydantic Models

Use structured types for both input and output:

```python
from pydantic import BaseModel
from pixie import app, PixieGenerator, UserInputRequirement

class Question(BaseModel):
    text: str
    difficulty: int

class Answer(BaseModel):
    answer: str
    confidence: float

@app
async def quiz(_: None) -> PixieGenerator[Question, Answer]:
    """Interactive quiz with structured I/O."""
    Agent.instrument_all()

    questions = [
        "What is the capital of France?",
        "What is 2 + 2?",
        "What color is the sky?"
    ]

    score = 0

    for i, q in enumerate(questions):
        # Send question
        question = yield Question(text=q, difficulty=i + 1)

        # Get structured answer
        answer = yield UserInputRequirement(Answer)

        # Check answer (simplified)
        if answer.confidence > 0.8:
            score += 1

    # Send final score
    yield Question(
        text=f"Quiz complete! Score: {score}/{len(questions)}",
        difficulty=0
    )
```

### Benefits

- **Type Safety** - Validated input/output
- **Better UI** - Forms for structured data
- **Self-Documenting** - Clear data contracts

## State Management

### Maintaining State Across Turns

```python
from typing import Dict
from pixie import app, PixieGenerator, UserInputRequirement

@app
async def task_manager(_: None) -> PixieGenerator[str, str]:
    """Manage tasks interactively."""
    Agent.instrument_all()

    # State: dictionary of tasks
    tasks: Dict[int, str] = {}
    next_id = 1

    yield "Task Manager. Commands: add, list, remove, done"

    while True:
        command = yield UserInputRequirement(str)

        if command.lower() == "done":
            yield "Goodbye!"
            break

        elif command.lower().startswith("add "):
            task = command[4:]
            tasks[next_id] = task
            yield f"Added task #{next_id}: {task}"
            next_id += 1

        elif command.lower() == "list":
            if not tasks:
                yield "No tasks."
            else:
                task_list = "\n".join(
                    f"#{id}: {task}" for id, task in tasks.items()
                )
                yield f"Tasks:\n{task_list}"

        elif command.lower().startswith("remove "):
            try:
                task_id = int(command[7:])
                if task_id in tasks:
                    removed = tasks.pop(task_id)
                    yield f"Removed: {removed}"
                else:
                    yield f"Task #{task_id} not found."
            except ValueError:
                yield "Invalid task ID."

        else:
            yield "Unknown command."
```

### State Persistence

For longer-lived state, use external storage:

```python
import json
from pathlib import Path

@app
async def persistent_chat(_: None) -> PixieGenerator[str, str]:
    """Chat with persistent history."""
    Agent.instrument_all()

    # Load history from file
    history_file = Path("chat_history.json")
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
    else:
        history = []

    yield "Welcome back!" if history else "Hello!"

    while True:
        user_msg = yield UserInputRequirement(str)

        if user_msg.lower() == "exit":
            # Save history
            with open(history_file, "w") as f:
                json.dump(history, f)
            yield "Saved. Goodbye!"
            break

        # Process message
        history.append({"role": "user", "content": user_msg})
        response = await process_message(user_msg, history)
        history.append({"role": "assistant", "content": response})

        yield response
```

## Streaming Responses

### Real-Time Streaming

Stream LLM responses as they're generated:

```python
from pydantic_ai import Agent

streaming_agent = Agent("openai:gpt-4o-mini")

@app
async def stream_chat(query: str) -> PixieGenerator[str, None]:
    """Stream responses in real-time."""
    Agent.instrument_all()

    yield "Generating response..."

    # Stream from agent
    async with streaming_agent.run_stream(query) as response:
        async for chunk in response.stream_text():
            # Yield each chunk as it arrives
            yield chunk
```

### Chunked Streaming

Send data in logical chunks:

```python
@app
async def generate_report(topic: str) -> PixieGenerator[str, None]:
    """Generate report in sections."""
    Agent.instrument_all()

    yield "# Report: " + topic
    yield "\n\n## Executive Summary"
    summary = await agent.run(f"Summarize {topic}")
    yield summary.output

    yield "\n\n## Detailed Analysis"
    analysis = await agent.run(f"Analyze {topic}")
    yield analysis.output

    yield "\n\n## Recommendations"
    recommendations = await agent.run(f"Recommendations for {topic}")
    yield recommendations.output

    yield "\n\n---\nReport complete."
```

## Advanced Patterns

### Branching Conversations

Create conversation flows with branches:

```python
@app
async def survey(_: None) -> PixieGenerator[str, str]:
    """Survey with conditional questions."""
    Agent.instrument_all()

    yield "Welcome to the survey!"

    yield "Are you a developer? (yes/no)"
    is_dev = yield UserInputRequirement(str)

    if is_dev.lower() == "yes":
        yield "What programming languages do you use?"
        languages = yield UserInputRequirement(str)
        yield f"Great! {languages} are popular choices."

        yield "Do you use AI tools? (yes/no)"
        uses_ai = yield UserInputRequirement(str)

        if uses_ai.lower() == "yes":
            yield "Which AI tools do you use?"
            tools = yield UserInputRequirement(str)
            yield f"Interesting! {tools} are powerful tools."
    else:
        yield "What field do you work in?"
        field = yield UserInputRequirement(str)
        yield f"{field} is a great field!"

    yield "Thanks for completing the survey!"
```

### Progress Indicators

Show progress during long operations:

```python
@app
async def batch_process(items: list[str]) -> PixieGenerator[str, None]:
    """Process items with progress updates."""
    Agent.instrument_all()

    total = len(items)
    yield f"Processing {total} items..."

    for i, item in enumerate(items):
        # Process item
        result = await process_item(item)

        # Update progress
        progress = (i + 1) / total * 100
        yield f"[{progress:.0f}%] Processed: {item} -> {result}"

    yield "✓ All items processed!"
```

### Error Recovery

Handle errors gracefully in interactive apps:

```python
@app
async def resilient_chat(_: None) -> PixieGenerator[str, str]:
    """Chat with error recovery."""
    Agent.instrument_all()

    yield "Hello! I'm here to help."

    retries = 0
    max_retries = 3

    while True:
        try:
            user_msg = yield UserInputRequirement(str)

            if user_msg.lower() == "exit":
                yield "Goodbye!"
                break

            # Process with potential failure
            result = await chatbot.run(user_msg)
            yield result.output

            # Reset retry counter on success
            retries = 0

        except Exception as e:
            retries += 1

            if retries >= max_retries:
                yield f"Too many errors. Please try again later."
                break

            yield f"Error occurred. Please try again. ({retries}/{max_retries})"
```

## Testing Interactive Apps

### Manual Testing via UI

Use the web UI to test interactivity:

1. Select your interactive app
2. Click "Run"
3. Interact with prompts
4. Verify responses
5. Check debug traces

### Automated Testing

Test interactive apps programmatically:

```python
import pytest
from pixie.utils import test_app

@pytest.mark.asyncio
async def test_chat_app():
    """Test interactive chat app."""

    # Create test runner
    runner = test_app(chat)

    # Start app
    response1 = await runner.start()
    assert "Hello" in response1

    # Send input
    response2 = await runner.send("Hi there")
    assert len(response2) > 0

    # Send exit
    response3 = await runner.send("exit")
    assert "Goodbye" in response3

    # Verify completion
    assert runner.is_complete()
```

## Best Practices

### 1. Always Provide Exit

Give users a way to exit:

```python
if user_input.lower() in {"exit", "quit", "stop"}:
    yield "Exiting..."
    break
```

### 2. Clear Instructions

Tell users what to expect:

```python
yield "Type 'help' for commands, 'exit' to quit."
```

### 3. Validate Input

Check user input before processing:

```python
user_input = yield UserInputRequirement(str)

if not user_input.strip():
    yield "Please provide valid input."
    continue
```

### 4. Maintain Context

Keep conversation context:

```python
history: list[ModelMessage] = []

# Always pass history to agent
result = await agent.run(msg, message_history=history)
```

### 5. Handle Timeouts

Set reasonable timeouts:

```python
import asyncio

try:
    result = await asyncio.wait_for(
        agent.run(query),
        timeout=30.0
    )
except asyncio.TimeoutError:
    yield "Request timed out. Please try again."
```

## Common Patterns

### Confirmation Dialogs

```python
yield "Are you sure you want to delete? (yes/no)"
confirmation = yield UserInputRequirement(str)

if confirmation.lower() == "yes":
    # Proceed with action
    yield "Deleted."
else:
    yield "Cancelled."
```

### Menu Selection

```python
yield "Options:\n1. Create\n2. Read\n3. Update\n4. Delete"
choice = yield UserInputRequirement(str)

if choice == "1":
    # Handle create
    pass
elif choice == "2":
    # Handle read
    pass
# ... etc
```

### Multi-Step Form

```python
yield "Let's create your profile."

yield "What's your name?"
name = yield UserInputRequirement(str)

yield "What's your email?"
email = yield UserInputRequirement(str)

yield "What's your role?"
role = yield UserInputRequirement(str)

yield f"Profile created for {name} ({email}) - {role}"
```

## Next Steps

- [Interactive Tools](./interactive-tool.md) - Add interactivity to agent tools
- [Concepts: Architecture](../concepts/architecture.md) - Understand how it works
- [Examples](https://github.com/yiouli/pixie-examples) - See more examples

## Troubleshooting

### Generator Not Completing

**Problem:** App keeps running after it should stop

**Solution:** Ensure you return from the generator:

```python
while True:
    if should_exit:
        yield "Goodbye!"
        return  # or break
```

### Lost State

**Problem:** State resets between yields

**Solution:** Define state before the loop:

```python
state = {}  # Define outside loop

while True:
    # Use state inside loop
    pass
```

### Input Not Received

**Problem:** `UserInputRequirement` doesn't get input

**Solution:** Verify the type matches:

```python
# Generator signature
async def app() -> PixieGenerator[str, str]:
    # Must match         ^^^^SendType^^^^
    user_input = yield UserInputRequirement(str)
```
