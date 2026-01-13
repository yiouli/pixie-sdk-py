# Make Your Application Interactive

Learn how to build multi-turn, interactive applications that can request user input during execution using `InputRequired`.

## Basic Pattern

Use `yield InputRequired(type)` to pause execution and request user input:

```python
from pixie import app, PixieGenerator, InputRequired

@app
async def my_interactive_app(_: None) -> PixieGenerator[str, str]:
    """Interactive application."""
    yield "What's your name?"
    name = yield InputRequired(str)  # Pauses here, waits for user input
    yield f"Hello, {name}!"
```

## Type Safety with InputRequired

### String Input

```python
@app
async def text_input(_: None) -> PixieGenerator[str, str]:
    yield "Enter some text:"
    text = yield InputRequired(str)  # Returns: str
    yield f"You entered: {text}"
```

### Integer Input

```python
@app
async def number_input(_: None) -> PixieGenerator[str, int]:
    yield "Enter a number:"
    number = yield InputRequired(int)  # Returns: int
    yield f"Your number squared is: {number ** 2}"
```

### Structured Input

```python
from pydantic import BaseModel, Field

class UserChoice(BaseModel):
    option: str = Field(description="Selected option")
    quantity: int = Field(default=1, description="Quantity")

@app
async def structured_input(_: None) -> PixieGenerator[str, UserChoice]:
    yield "Please make your selection:"
    choice = yield InputRequired(UserChoice)  # Returns: UserChoice
    yield f"You selected {choice.option} (qty: {choice.quantity})"
```

## Generator Type Annotations

The `PixieGenerator` type takes two parameters:

```python
PixieGenerator[YieldType, SendType]
```

- **YieldType**: Type of values you `yield` to the user
- **SendType**: Type of values you receive from `InputRequired`

Examples:

```python
PixieGenerator[str, str]           # Yield strings, receive strings
PixieGenerator[str, int]           # Yield strings, receive integers
PixieGenerator[MyModel, str]       # Yield structured data, receive strings
PixieGenerator[str, None]          # Yield strings, no user input
```

        else:
            yield "❌ Unknown command. Use: add, list, remove, exit"

````

### Example 3: Confirmation Dialog

```python
@app
async def delete_confirmation(_: None) -> PixieGenerator[str, str]:
    """Demonstrate confirmation pattern."""

    yield "⚠️ This will delete all data. Are you sure?"
    yield "Type 'yes' to confirm or 'no' to cancel:"

    confirmation = yield InputRequired(str)

    if confirmation.lower().strip() == "yes":
        yield "🗑️ Deleting data..."
        # Perform deletion
        yield "✅ Data deleted successfully"
    else:
        yield "❌ Operation cancelled"
````

### Example 4: Multi-Step Form

```python
from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    email: str
    age: int
    interests: list[str]

@app
async def profile_builder(_: None) -> PixieGenerator[str, str]:
    """Build user profile interactively."""

    yield "Let's create your profile!"

    yield "What's your name?"
    name = yield InputRequired(str)

    yield "What's your email?"
    email = yield InputRequired(str)

    yield "What's your age?"
    age_str = yield InputRequired(str)
    age = int(age_str)

    yield "List your interests (comma-separated):"
    interests_str = yield InputRequired(str)
    interests = [i.strip() for i in interests_str.split(",")]

    profile = UserProfile(
        name=name,
        email=email,
        age=age,
        interests=interests
    )

    yield f"✅ Profile created for {profile.name}!"
    yield f"Email: {profile.email}"
    yield f"Age: {profile.age}"
    yield f"Interests: {', '.join(profile.interests)}"
```

### Example 5: Weather Chatbot with History

```python
from pydantic_ai import Agent, ModelMessage, ModelRequest

weather_agent = Agent("openai:gpt-4o-mini", system_prompt="You provide weather info.")

@app
async def weather_chat(_: None) -> PixieGenerator[str, str]:
    """Weather chatbot with conversation history."""

    Agent.instrument_all()

    yield "🌤️ Weather Assistant"
    yield "Ask me about weather in any city! (Type 'exit' to quit)"

    history: list[ModelMessage] = []

    while True:
        user_msg = yield InputRequired(str)

        if user_msg.lower() in {"exit", "quit"}:
            yield "Stay dry! 👋"
            break

        # Run agent with history
        result = await weather_agent.run(user_msg, message_history=history)

        # Update history
        history.append(ModelRequest.user_text_prompt(user_msg))
        history.append(result.response)

        # Send response
        yield result.output
```

## Best Practices

### 1. Always Provide Exit Path

```python
while True:
    user_input = yield InputRequired(str)

    if user_input.lower() in {"exit", "quit", "stop"}:
        yield "Goodbye!"
        break  # Exit the loop

    # Process input
```

### 2. Give Clear Instructions

```python
# ✅ Good - clear what to do
yield "Enter your age (must be 18 or older):"
age = yield InputRequired(str)

# ❌ Bad - unclear
yield "Age?"
age = yield InputRequired(str)
```

### 3. Validate Input

```python
yield "Enter a number between 1 and 10:"
input_str = yield InputRequired(str)

try:
    number = int(input_str)
    if 1 <= number <= 10:
        yield f"You chose {number}"
    else:
        yield "Number out of range!"
except ValueError:
    yield "That's not a valid number!"
```

### 4. Maintain State

```python
@app
async def stateful_app(_: None) -> PixieGenerator[str, str]:
    """Keep state across interactions."""

    # State defined outside loop
    counter = 0
    messages = []

    while True:
        user_input = yield InputRequired(str)
        counter += 1
        messages.append(user_input)

        yield f"Message #{counter}: {user_input}"
        yield f"Total messages: {len(messages)}"
```

### 5. Handle Errors Gracefully

```python
@app
async def robust_app(_: None) -> PixieGenerator[str, str]:
    """Handle errors in user input."""
```
