# Use Structured Input/Output

Learn how to use Pydantic models for type-safe, validated, and well-documented input and output in your applications.

## Why Use Structured I/O?

Using Pydantic models instead of plain types provides:

- **Type Safety:** Automatic validation and type checking
- **Rich Documentation:** Field-level descriptions and constraints
- **Better UX:** Structured forms in the web UI
- **Default Values:** Sensible defaults for optional fields
- **Validation:** Built-in validation rules

## Input Models

```python
from pydantic import BaseModel, Field
from pixie import app

class SearchQuery(BaseModel):
    """Search configuration."""

    query: str = Field(description="Text to search for")
    max_results: int = Field(default=10, description="Maximum results")
    case_sensitive: bool = Field(default=False, description="Case-sensitive search")

@app
async def search(config: SearchQuery) -> str:
    """Search with structured parameters."""
    return f"Found {config.max_results} results for '{config.query}'"
```

### Nested Models

```python
class Address(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")

class CustomerInfo(BaseModel):
    name: str = Field(description="Customer full name")
    address: Address = Field(description="Mailing address")

@app
async def register_customer(customer: CustomerInfo) -> str:
    return f"Registered {customer.name} in {customer.address.city}"
```

## Output Models

```python
class AnalysisResult(BaseModel):
    """Text analysis result."""

    sentiment: str = Field(description="Detected sentiment: positive, negative, neutral")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0-1)")
    keywords: list[str] = Field(description="Extracted keywords")

@app
async def analyze_text(text: str) -> AnalysisResult:
    """Analyze text and return structured results."""
    return AnalysisResult(
        sentiment="positive",
        confidence=0.87,
        keywords=["important", "success"]
    )
```

## Field Configuration

### Validation Constraints

```python
class UserInput(BaseModel):
    # String length
    username: str = Field(
        min_length=3,
        max_length=20,
        description="Username (3-20 characters)"
    )

    # Numeric ranges
    age: int = Field(
        ge=18,
        le=120,
        description="Age (must be 18 or older)"
    )
```

### Optional Fields

```python
class OptionalConfig(BaseModel):
    required_field: str = Field(description="This field is required")
    optional_field: str | None = Field(default=None, description="This field is optional")
```

## Complete Examples

### Example 1: Weather Service

```python
class WeatherQuery(BaseModel):
    """Weather query parameters."""

    city: str = Field(description="City name")
    country: str = Field(default="US", description="Country code")
    include_forecast: bool = Field(default=False, description="Include 5-day forecast")

class WeatherData(BaseModel):
    """Weather data point."""

    temperature: float = Field(description="Temperature in Celsius")
    conditions: str = Field(description="Weather conditions")
    humidity: int = Field(description="Humidity percentage")
    wind_speed: float = Field(description="Wind speed")

class WeatherResponse(BaseModel):
    """Complete weather response."""

    location: str = Field(description="Location name")
    current: WeatherData = Field(description="Current weather")
    forecast: list[WeatherData] | None = Field(
        default=None,
        description="5-day forecast (if requested)"
    )

@app
async def get_weather(query: WeatherQuery) -> WeatherResponse:
    """Get current weather and optional forecast for a city.

    This application provides accurate weather information
    including temperature, conditions, humidity, and wind speed.
    Optionally includes a 5-day forecast.
    """
    # Fetch weather data
    current_weather = WeatherData(
        temperature=22.5,
        conditions="Partly Cloudy",
        humidity=65,
        wind_speed=12.3
    )

    return WeatherResponse(
        location=f"{query.city}, {query.country}",
        current=current_weather,
        forecast=None if not query.include_forecast else []
    )
```

### Example 2: Task Manager

```python
class Task(BaseModel):
    """A single task."""

    id: int = Field(description="Unique task ID")
    title: str = Field(min_length=1, max_length=200, description="Task title")
    completed: bool = Field(default=False, description="Completion status")
    priority: int = Field(default=3, ge=1, le=5, description="Priority (1=high, 5=low)")
    tags: list[str] = Field(default=[], description="Task tags")

class TaskOperation(BaseModel):
    """Task management operation."""

    action: str = Field(description="Action: 'add', 'update', 'delete', 'list'")
    task_id: int | None = Field(default=None, description="Task ID (for update/delete)")
    task: Task | None = Field(default=None, description="Task data (for add/update)")

class TaskResult(BaseModel):
    """Result of task operation."""

    success: bool = Field(description="Whether operation succeeded")
    message: str = Field(description="Result message")
    tasks: list[Task] = Field(default=[], description="Current task list")

@app
async def manage_tasks(operation: TaskOperation) -> TaskResult:
    """Manage tasks with add, update, delete, and list operations.

    This application provides complete task management:
    - Add new tasks
    - Update existing tasks
    - Delete tasks
    - List all tasks

    All operations return the updated task list.
    """
    # Process operation
    if operation.action == "list":
        return TaskResult(
            success=True,
            message="Tasks retrieved",
            tasks=[]  # Return actual tasks
        )

    return TaskResult(
        success=True,
        message=f"Task {operation.action} completed",
        tasks=[]
    )
```

## Web UI Integration

### Input Forms

When you use Pydantic models for input, the web UI automatically generates forms:

```python
class FormInput(BaseModel):
    name: str = Field(description="Your name")
    age: int = Field(default=25, ge=18, description="Your age")
    subscribe: bool = Field(default=True, description="Subscribe to newsletter")
```

**Renders in UI as:**

_(Placeholder for screenshot)_

- **name:** Text input field with label "Your name"
- **age:** Number input with default value 25, minimum 18
- **subscribe:** Checkbox, checked by default

### Output Display

Structured output is displayed in formatted JSON:

```python
class OutputData(BaseModel):
    status: str
    result: dict
    count: int
```

**Displays in UI as:**

_(Placeholder for screenshot)_

```json
{
  "status": "success",
  "result": {
    "key": "value"
  },
  "count": 42
}
```

## Generator with Structured Types

You can use Pydantic models with generators too:

```python
from pixie import PixieGenerator

class Progress(BaseModel):
    """Progress update."""
    step: str = Field(description="Current step")
    percent: int = Field(ge=0, le=100, description="Completion percentage")

@app
async def long_task(_: None) -> PixieGenerator[Progress, None]:
    """Long-running task with structured progress updates."""

    yield Progress(step="Initializing", percent=0)

```
