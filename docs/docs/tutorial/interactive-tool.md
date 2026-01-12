---
sidebar_position: 5
---

# Interactive Tool Use

Learn how to add user interaction from within your agent's tools, enabling dynamic, human-in-the-loop workflows.

## Why Interactive Tools?

Interactive tools allow your agent to:

- Request human approval before critical actions
- Gather additional information during execution
- Provide real-time updates on long-running operations
- Enable human-in-the-loop workflows

This is powerful for scenarios like:

- Approving financial transactions
- Confirming destructive operations
- Gathering missing information
- Interactive data exploration

## Basic Interactive Tool

### Simple Example

```python
from pixie import app, PixieGenerator, UserInputRequirement
from pydantic_ai import Agent, RunContext

agent = Agent("openai:gpt-4o-mini")

@agent.tool
async def send_email(
    ctx: RunContext[None],
    to: str,
    subject: str,
    body: str
) -> str:
    """Send an email with user confirmation."""

    # Check if we're in an interactive context
    if hasattr(ctx, 'yield_value'):
        # Request confirmation from user
        ctx.yield_value(
            f"Send email to {to}?\nSubject: {subject}\nBody: {body}\n(yes/no)"
        )

        confirmation = ctx.request_input(str)

        if confirmation.lower() != "yes":
            return "Email cancelled by user."

    # Send the email
    result = actually_send_email(to, subject, body)
    return f"Email sent successfully to {to}"

@app
async def email_assistant(request: str) -> PixieGenerator[str, str]:
    """Email assistant with confirmation."""
    Agent.instrument_all()

    # Agent will call send_email tool, which may request input
    result = await agent.run(request)

    yield result.output
```

## How It Works

### The Flow

1. **Agent calls tool** - During agent execution
2. **Tool yields message** - Using `ctx.yield_value()`
3. **Tool requests input** - Using `ctx.request_input()`
4. **User provides input** - Via the UI
5. **Tool continues** - With user's response
6. **Agent continues** - With tool result

### Key Methods

```python
# In a tool function
ctx.yield_value(message)  # Send message to user
user_input = ctx.request_input(InputType)  # Request input
```

## Approval Workflows

### Approve Before Action

```python
from pydantic_ai import Agent, RunContext

admin_agent = Agent("openai:gpt-4o-mini")

@admin_agent.tool
async def delete_user(ctx: RunContext[None], user_id: int) -> str:
    """Delete a user with admin approval."""

    # Get user details
    user = get_user(user_id)

    # Request approval
    ctx.yield_value(
        f"⚠️  Delete user '{user.name}' (ID: {user_id})?\n"
        f"This action cannot be undone.\n"
        f"Type 'DELETE' to confirm:"
    )

    confirmation = ctx.request_input(str)

    if confirmation != "DELETE":
        return f"Deletion cancelled. User '{user.name}' was not deleted."

    # Perform deletion
    delete_user_from_db(user_id)

    return f"User '{user.name}' (ID: {user_id}) has been deleted."

@app
async def admin_panel(command: str) -> PixieGenerator[str, str]:
    """Admin panel with safety checks."""
    Agent.instrument_all()

    result = await admin_agent.run(command)
    yield result.output
```

### Multi-Step Approval

```python
@agent.tool
async def process_refund(
    ctx: RunContext[None],
    order_id: int,
    amount: float,
    reason: str
) -> str:
    """Process refund with multiple approval steps."""

    order = get_order(order_id)

    # Step 1: Verify order details
    ctx.yield_value(
        f"Refund Request:\n"
        f"Order: {order_id}\n"
        f"Amount: ${amount}\n"
        f"Reason: {reason}\n"
        f"Continue? (yes/no)"
    )

    if ctx.request_input(str).lower() != "yes":
        return "Refund cancelled at verification step."

    # Step 2: Check refund amount
    if amount > order.total:
        ctx.yield_value(
            f"Warning: Refund amount (${amount}) exceeds "
            f"order total (${order.total}). Proceed? (yes/no)"
        )

        if ctx.request_input(str).lower() != "yes":
            return "Refund cancelled due to amount mismatch."

    # Step 3: Final confirmation
    ctx.yield_value(
        f"Final confirmation: Process ${amount} refund? (yes/no)"
    )

    if ctx.request_input(str).lower() != "yes":
        return "Refund cancelled at final confirmation."

    # Process the refund
    process_refund_in_system(order_id, amount, reason)

    return f"Refund of ${amount} processed for order {order_id}."
```

## Gathering Additional Information

### Dynamic Information Gathering

```python
@agent.tool
async def book_flight(
    ctx: RunContext[None],
    destination: str,
    date: str
) -> str:
    """Book a flight with additional details."""

    # Check available flights
    flights = search_flights(destination, date)

    if not flights:
        return f"No flights available to {destination} on {date}."

    # Show options and get user choice
    flight_list = "\n".join(
        f"{i+1}. {f.airline} - ${f.price} - {f.departure_time}"
        for i, f in enumerate(flights)
    )

    ctx.yield_value(
        f"Available flights to {destination} on {date}:\n"
        f"{flight_list}\n"
        f"Select flight (1-{len(flights)}):"
    )

    choice = ctx.request_input(int)

    if choice < 1 or choice > len(flights):
        return "Invalid flight selection."

    selected_flight = flights[choice - 1]

    # Get additional preferences
    ctx.yield_value("Select seat preference (window/aisle/middle):")
    seat_pref = ctx.request_input(str)

    ctx.yield_value("Add baggage? (yes/no):")
    add_baggage = ctx.request_input(str).lower() == "yes"

    # Book the flight
    booking = book_flight_in_system(
        selected_flight,
        seat_preference=seat_pref,
        include_baggage=add_baggage
    )

    return (
        f"Flight booked!\n"
        f"Confirmation: {booking.confirmation_code}\n"
        f"Seat: {seat_pref}\n"
        f"Baggage: {'Included' if add_baggage else 'Not included'}"
    )
```

### Handling Missing Information

```python
@agent.tool
async def create_ticket(
    ctx: RunContext[None],
    title: str,
    description: str = ""
) -> str:
    """Create a ticket, requesting missing info."""

    # Check for empty description
    if not description:
        ctx.yield_value(
            f"Creating ticket: '{title}'\n"
            f"No description provided. Add description? (yes/no)"
        )

        if ctx.request_input(str).lower() == "yes":
            ctx.yield_value("Enter description:")
            description = ctx.request_input(str)

    # Get priority
    ctx.yield_value("Set priority (low/medium/high):")
    priority = ctx.request_input(str)

    # Get assignee
    ctx.yield_value("Assign to (leave empty for unassigned):")
    assignee = ctx.request_input(str)

    # Create ticket
    ticket = create_ticket_in_system(
        title=title,
        description=description,
        priority=priority,
        assignee=assignee if assignee else None
    )

    return f"Ticket #{ticket.id} created: {title}"
```

## Progress Updates

### Long-Running Operations

```python
@agent.tool
async def analyze_data(
    ctx: RunContext[None],
    dataset: str
) -> str:
    """Analyze large dataset with progress updates."""

    data = load_dataset(dataset)
    total_steps = 5

    # Step 1: Loading
    ctx.yield_value(f"[1/{total_steps}] Loading dataset...")
    await asyncio.sleep(1)  # Simulate work

    # Step 2: Cleaning
    ctx.yield_value(f"[2/{total_steps}] Cleaning data...")
    cleaned_data = clean_data(data)
    await asyncio.sleep(2)

    # Step 3: Analysis
    ctx.yield_value(f"[3/{total_steps}] Running analysis...")
    results = analyze(cleaned_data)
    await asyncio.sleep(3)

    # Step 4: Generating report
    ctx.yield_value(f"[4/{total_steps}] Generating report...")
    report = generate_report(results)
    await asyncio.sleep(1)

    # Step 5: Complete
    ctx.yield_value(f"[5/{total_steps}] Complete!")

    return report

@app
async def data_analyst(query: str) -> PixieGenerator[str, None]:
    """Data analysis with progress tracking."""
    Agent.instrument_all()

    result = await agent.run(query)
    yield result.output
```

### Interactive Debugging

```python
@agent.tool
async def debug_issue(
    ctx: RunContext[None],
    error_code: str
) -> str:
    """Debug an issue interactively."""

    ctx.yield_value(f"Investigating error: {error_code}")

    # Check logs
    logs = get_error_logs(error_code)
    ctx.yield_value(f"Found {len(logs)} log entries.")

    # Ask if user wants to see logs
    ctx.yield_value("View logs? (yes/no)")
    if ctx.request_input(str).lower() == "yes":
        ctx.yield_value(f"Logs:\n{logs}")

    # Suggest solutions
    solutions = suggest_solutions(error_code)
    ctx.yield_value(
        f"Possible solutions:\n" +
        "\n".join(f"{i+1}. {s}" for i, s in enumerate(solutions)) +
        f"\nTry solution? (1-{len(solutions)} or 'no')"
    )

    choice = ctx.request_input(str)

    if choice.lower() == "no":
        return "Debug session ended without applying solution."

    try:
        solution_idx = int(choice) - 1
        if 0 <= solution_idx < len(solutions):
            apply_solution(error_code, solutions[solution_idx])
            return f"Applied solution: {solutions[solution_idx]}"
    except ValueError:
        pass

    return "Invalid choice. Debug session ended."
```

## Structured Interactive Tools

### Using Pydantic for Tool I/O

```python
from pydantic import BaseModel

class ApprovalRequest(BaseModel):
    action: str
    details: dict
    risk_level: str

class ApprovalResponse(BaseModel):
    approved: bool
    comment: str

@agent.tool
async def execute_action(
    ctx: RunContext[None],
    action: str,
    params: dict
) -> str:
    """Execute action with structured approval."""

    # Create approval request
    risk = assess_risk(action, params)

    request = ApprovalRequest(
        action=action,
        details=params,
        risk_level=risk
    )

    # Yield structured request
    ctx.yield_value(request.model_dump_json(indent=2))
    ctx.yield_value("Approve? Provide response:")

    # Get structured response
    response = ctx.request_input(ApprovalResponse)

    if not response.approved:
        return f"Action cancelled: {response.comment}"

    # Execute action
    result = execute_action_in_system(action, params)

    return f"Action completed: {result}\nComment: {response.comment}"
```

## Best Practices

### 1. Clear Messages

Always provide clear, actionable messages:

```python
# ✅ Good
ctx.yield_value(
    "Delete 5 files?\n"
    "Files: file1.txt, file2.txt, ...\n"
    "Type 'yes' to confirm, 'no' to cancel:"
)

# ❌ Bad
ctx.yield_value("Delete?")
```

### 2. Validate Input

Always validate user input in tools:

```python
response = ctx.request_input(str)

if response.lower() not in {"yes", "no"}:
    return "Invalid response. Operation cancelled."
```

### 3. Provide Context

Give users enough information to make decisions:

```python
ctx.yield_value(
    f"Transaction Details:\n"
    f"Amount: ${amount}\n"
    f"Recipient: {recipient}\n"
    f"Current Balance: ${balance}\n"
    f"New Balance: ${balance - amount}\n"
    f"Approve transaction? (yes/no)"
)
```

### 4. Handle Timeouts

Consider timeouts for user input:

```python
try:
    response = await asyncio.wait_for(
        ctx.request_input(str),
        timeout=30.0
    )
except asyncio.TimeoutError:
    return "Operation timed out. Please try again."
```

### 5. Graceful Fallbacks

Provide fallbacks when interaction isn't available:

```python
if hasattr(ctx, 'yield_value'):
    # Interactive mode
    ctx.yield_value("Confirm action?")
    confirmed = ctx.request_input(str).lower() == "yes"
else:
    # Non-interactive mode - use default
    confirmed = True

if not confirmed:
    return "Action cancelled."
```

## Testing Interactive Tools

### Unit Testing

Test tools with mock context:

```python
import pytest
from unittest.mock import Mock

@pytest.mark.asyncio
async def test_delete_user_tool():
    """Test delete_user tool with mock context."""

    # Create mock context
    ctx = Mock()

    # Mock user input
    ctx.request_input = Mock(return_value="DELETE")
    ctx.yield_value = Mock()

    # Call tool
    result = await delete_user(ctx, user_id=123)

    # Verify behavior
    assert "deleted" in result.lower()
    ctx.yield_value.assert_called_once()
    ctx.request_input.assert_called_once()
```

### Integration Testing

Test the full flow via the app:

```python
@pytest.mark.asyncio
async def test_email_assistant_flow():
    """Test full interactive tool flow."""

    runner = test_app(email_assistant)

    # Start app
    await runner.start("Send an email to john@example.com")

    # Should request confirmation
    message = await runner.get_last_message()
    assert "Send email" in message

    # Provide confirmation
    await runner.send("yes")

    # Should complete
    result = await runner.wait_completion()
    assert "sent successfully" in result.lower()
```

## Advanced Patterns

### Nested Interactions

Tools can have multiple levels of interaction:

```python
@agent.tool
async def wizard(ctx: RunContext[None]) -> str:
    """Multi-step wizard."""

    ctx.yield_value("Step 1: Enter name:")
    name = ctx.request_input(str)

    ctx.yield_value(f"Step 2: Choose role for {name}:")
    role = ctx.request_input(str)

    if role.lower() == "admin":
        ctx.yield_value("Admin access requires approval. Confirm? (yes/no)")
        if ctx.request_input(str).lower() != "yes":
            role = "user"

    ctx.yield_value(f"Step 3: Set permissions for {role}:")
    permissions = ctx.request_input(str)

    return f"Created {role} '{name}' with permissions: {permissions}"
```

### Conditional Interaction

Only interact when necessary:

```python
@agent.tool
async def smart_action(ctx: RunContext[None], risk: float) -> str:
    """Only request approval for risky actions."""

    # Low risk - no confirmation needed
    if risk < 0.3:
        execute_action()
        return "Action completed automatically (low risk)."

    # High risk - require confirmation
    ctx.yield_value(
        f"⚠️  High risk action (risk: {risk:.0%})\n"
        f"Confirm? (yes/no)"
    )

    if ctx.request_input(str).lower() != "yes":
        return "High risk action cancelled."

    execute_action()
    return "High risk action completed with approval."
```

## Common Pitfalls

### Not Checking for Interactive Context

Always check if interaction is available:

```python
# ✅ Correct
if hasattr(ctx, 'yield_value'):
    ctx.yield_value("Message")
    response = ctx.request_input(str)
else:
    response = default_value

# ❌ Wrong - will fail in non-interactive mode
ctx.yield_value("Message")
response = ctx.request_input(str)
```

### Infinite Loops

Avoid infinite interaction loops:

```python
# ✅ Correct - limited retries
max_attempts = 3
for attempt in range(max_attempts):
    ctx.yield_value("Enter valid input:")
    response = ctx.request_input(str)
    if validate(response):
        break
else:
    return "Too many invalid attempts."

# ❌ Wrong - could loop forever
while True:
    response = ctx.request_input(str)
    if validate(response):
        break
```

### Missing Error Handling

Always handle potential errors:

```python
# ✅ Correct
try:
    ctx.yield_value("Enter number:")
    value = ctx.request_input(int)
except ValueError:
    return "Invalid number provided."

# ❌ Wrong - will crash on invalid input
ctx.yield_value("Enter number:")
value = ctx.request_input(int)  # May raise ValueError
```

## Next Steps

- [Concepts: Architecture](../concepts/architecture.md) - How Pixie works
- [Concepts: Instrumentation](../concepts/instrumentation.md) - Tracing details
- [Examples](https://github.com/yiouli/pixie-examples) - More examples

## Summary

Interactive tools enable powerful human-in-the-loop workflows by:

- Requesting user approval for critical actions
- Gathering additional information dynamically
- Providing progress updates
- Creating guided, multi-step processes

Use them to build safer, more flexible AI agents that collaborate with users in real-time.
