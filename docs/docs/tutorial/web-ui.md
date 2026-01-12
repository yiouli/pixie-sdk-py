---
sidebar_position: 2
---

# Web UI

Learn how to use the Pixie web UI to interact with your AI agents and explore execution traces.

## Accessing the Web UI

Once your Pixie server is running, access the web UI at:

```
http://127.0.0.1:8000
```

Or your configured host/port if different.

## Main Interface

The Pixie web UI consists of three main screens:

1. **App Selection** - Choose which agent to run
2. **Chat Interface** - Interact with your agent
3. **Debug Screen** - Explore traces and execution details

## App Selection Screen

### Overview

When you first open the UI, you'll see a list of all available agents (apps) discovered by the server.

### Features

**App Cards** display:

- **Name** - The function name (e.g., `weather`, `chat`)
- **Description** - From the function docstring
- **Input Schema** - Expected input type
- **Output Schema** - Expected output type

**Actions:**

- Click on an app card to select it
- Click "Run" or press Enter to start

### Filtering Apps

If you have many apps, use the search bar at the top to filter by name or description.

## Chat Interface

### Running an App

After selecting an app:

1. **Enter Input** - Type your input in the text box
2. **Submit** - Click "Run" or press Enter
3. **View Output** - See the agent's response in real-time

### Input Types

#### String Input

For apps that accept a string:

```python
@app
async def weather(query: str) -> str:
    ...
```

Simply type your query:

```
What's the weather in Tokyo?
```

#### Structured Input

For apps with Pydantic models:

```python
class WeatherQuery(BaseModel):
    location: str
    units: str = "celsius"

@app
async def weather(query: WeatherQuery) -> str:
    ...
```

The UI will show a **Schema Input** form with fields for each property:

- `location`: [text input]
- `units`: [text input with default]

Fill in the form and submit.

#### No Input Required

For apps that don't need input:

```python
@app
async def daily_report(_: None) -> str:
    ...
```

The UI will show a "Run" button without an input field.

### Real-Time Streaming

For streaming apps (async generators), the UI displays responses as they arrive:

```python
@app
async def generate_story(prompt: str) -> PixieGenerator[str, None]:
    yield "Once upon a time..."
    yield "There was a brave knight..."
    yield "The end."
```

Each yielded value appears immediately in the chat interface.

### Interactive Apps

For apps that request user input mid-execution:

```python
@app
async def interview(_: None) -> PixieGenerator[str, str]:
    yield "What's your name?"
    name = yield UserInputRequirement(str)
    yield f"Nice to meet you, {name}!"
```

**User Flow:**

1. Agent sends first message: "What's your name?"
2. UI displays message and shows input field
3. You type your name and submit
4. Agent receives your input and continues
5. Repeat for additional interactions

The UI automatically detects when the agent needs input and displays the appropriate interface.

## Debug Screen

Click the **Debug** button in the top-right to access advanced tracing and debugging features.

### Execution Timeline

View a chronological timeline of your agent's execution:

- **Start/End Times** - When each step occurred
- **Duration** - How long each step took
- **Status** - Success, error, or in-progress

### Trace Tree

Explore the hierarchical structure of your agent's execution:

```
📦 weather (app)
  └─ 🤖 Agent Run
      ├─ 🔧 Tool Call: get_temperature
      │   └─ ⏱️ 150ms
      ├─ 💬 LLM Call
      │   ├─ 📝 Prompt: "What's the weather..."
      │   ├─ 📤 Response: "The temperature is..."
      │   └─ ⏱️ 1.2s
      └─ ✅ Complete
```

Click on any node to view details.

### Span Details

Select a span in the trace tree to view:

#### Attributes

- **Name** - Span name
- **Kind** - Internal, Client, Server, etc.
- **Status** - OK, Error
- **Duration** - Total time

#### Events

Time-stamped events that occurred during the span:

- LLM prompts and responses
- Tool calls and results
- Errors and exceptions

#### Context

- **Trace ID** - Unique trace identifier
- **Span ID** - Unique span identifier
- **Parent Span ID** - Links to parent

#### LLM-Specific Data

For LLM calls:

- **Model** - Which model was used
- **Prompt** - Full input prompt
- **Response** - Complete LLM output
- **Token Usage** - Input/output tokens
- **Cost** - Estimated API cost (if available)

### Filtering Traces

Use the filter bar to narrow down spans:

- **By Type** - Show only LLM calls, tool calls, etc.
- **By Status** - Show only errors
- **By Duration** - Show only slow spans (> 1s)

### Exporting Traces

Export trace data for further analysis:

1. Click **Export** in the debug screen
2. Choose format: JSON, CSV, or OpenTelemetry
3. Download the file

Use exported data for:

- Performance analysis
- Cost tracking
- Debugging in external tools

## Control Bar

The control bar at the bottom provides quick actions:

### Run Controls

- **Run** - Start the agent (if not running)
- **Pause** - Pause execution (if supported)
- **Stop** - Terminate execution
- **Reset** - Clear chat history and start fresh

### View Toggles

- **Show Traces** - Toggle trace visibility in chat
- **Auto-Scroll** - Automatically scroll to latest message
- **Debug Mode** - Show additional debug information

### Status Indicators

- **🟢 Ready** - Agent is idle and ready
- **🔵 Running** - Agent is executing
- **🟡 Waiting** - Agent is waiting for user input
- **🔴 Error** - Agent encountered an error

## Advanced Features

### Configuring Alternative Server Address

If your Pixie server is running on a different host or port, configure the UI:

#### Option 1: Environment Variable

Before starting the server, set:

```bash
export PIXIE_UI_SERVER_URL=http://your-server:8080
pixie
```

#### Option 2: URL Parameter

Access the UI with a query parameter:

```
http://127.0.0.1:8000?server=http://your-server:8080
```

#### Option 3: Settings Page

1. Click the ⚙️ icon in the top-right
2. Enter the server URL
3. Click "Save"
4. Refresh the page

The UI will now connect to your specified server.

### Dark Mode

Toggle dark mode:

1. Click the 🌙/☀️ icon in the top-right
2. Your preference is saved in browser storage

### Keyboard Shortcuts

Speed up your workflow with keyboard shortcuts:

- **Enter** - Submit input (or Cmd/Ctrl+Enter if multi-line)
- **Escape** - Clear input field
- **Cmd/Ctrl+K** - Open app selector
- **Cmd/Ctrl+D** - Toggle debug screen
- **Cmd/Ctrl+R** - Refresh app list

### Multi-Line Input

For longer inputs:

1. Click the expand icon (⤢) in the input field
2. Enter multi-line text
3. Use Cmd/Ctrl+Enter to submit

### Copy Output

Hover over any agent response to reveal a **Copy** button. Click to copy the text to your clipboard.

### Sharing Runs

To share a specific agent run:

1. Click **Share** in the control bar
2. Copy the generated URL
3. Anyone with the URL can view the run (read-only)

## Troubleshooting

### UI Not Loading

**Problem:** Blank screen or "Connection refused"

**Solutions:**

1. Verify server is running: `curl http://127.0.0.1:8000/health`
2. Check browser console for errors (F12)
3. Ensure no ad blockers are interfering
4. Try a different browser

### Agent Not Appearing

**Problem:** Your agent doesn't show up in the app list

**Solutions:**

1. Check server logs for discovery errors
2. Verify `@app` decorator is used
3. Restart the server
4. Click "Refresh" in the UI

### Traces Not Showing

**Problem:** Debug screen is empty

**Solutions:**

1. Ensure `Agent.instrument_all()` is called
2. Check that OpenTelemetry is properly configured
3. Verify the agent actually executed (check for errors)
4. Refresh the debug screen

### Input Schema Not Rendering

**Problem:** Structured input form doesn't appear

**Solutions:**

1. Verify your Pydantic model is valid
2. Check that all fields have type annotations
3. Look for server errors in logs
4. Try simplifying your model

### Performance Issues

**Problem:** UI is slow or unresponsive

**Solutions:**

1. Limit trace depth in debug screen
2. Clear old runs (click "Clear History")
3. Use filtering to reduce visible data
4. Check server resources (CPU/memory)

## Next Steps

Now that you're familiar with the UI:

- [Structured Input/Output](./structured-io.md) - Learn about Pydantic models
- [Interactive Apps](./interactive-app.md) - Build conversational agents
- [Examples](https://github.com/yiouli/pixie-examples) - See real-world examples

## Tips & Best Practices

### For Development

1. **Use Auto-Reload** - Start server with `--reload` flag
2. **Keep Debug Screen Open** - Monitor traces in real-time
3. **Test Edge Cases** - Try malformed inputs, timeouts, etc.
4. **Export Traces** - Save interesting execution patterns

### For Demos

1. **Prepare Test Inputs** - Have example queries ready
2. **Use Dark Mode** - Often looks better in presentations
3. **Hide Unnecessary Details** - Focus on the output
4. **Share Runs** - Send links instead of screenshots

### For Production

1. **Configure External Server** - Don't rely on localhost
2. **Enable Authentication** - Protect your agents (coming soon)
3. **Monitor Performance** - Use trace data to optimize
4. **Set Up Alerts** - Get notified of errors
