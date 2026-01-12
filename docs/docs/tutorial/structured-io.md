---
sidebar_position: 3
---

# Structured Input/Output

Learn how to use Pydantic models for type-safe, structured data in your Pixie applications.

## Why Structured I/O?

Using structured input and output with Pydantic models provides:

- **Type Safety** - Catch errors before runtime
- **Validation** - Automatic input validation
- **Documentation** - Self-documenting schemas
- **IDE Support** - Autocomplete and type hints
- **Better UI** - Automatic form generation in web UI

## Basic Structured Input

### Simple Example

Instead of accepting plain strings, use Pydantic models:

```python
from pydantic import BaseModel
from pixie import app
from pydantic_ai import Agent

class WeatherQuery(BaseModel):
    location: str
    units: str = "celsius"  # Default value

weather_agent = Agent("openai:gpt-4o-mini")

@app
async def weather(query: WeatherQuery) -> str:
    """Get weather information with structured input."""
    Agent.instrument_all()

    result = await weather_agent.run(
        f"What's the weather in {query.location}? Use {query.units}."
    )

    return result.output
```

### Benefits

1. **Validation** - Automatically validates input
2. **Defaults** - Provides default values
3. **Type Checking** - IDE will warn about type mismatches
4. **UI Generation** - Web UI creates a form automatically

### Using in GraphQL

Query with structured input:

```graphql
subscription {
  run(
    name: "weather"
    inputData: "{\"location\": \"Tokyo\", \"units\": \"celsius\"}"
  ) {
    data
    status
  }
}
```

## Field Validation

### Built-in Validators

Pydantic provides many built-in validators:

```python
from pydantic import BaseModel, Field
from typing import Literal

class SearchQuery(BaseModel):
    query: str = Field(min_length=3, max_length=200)
    max_results: int = Field(default=10, ge=1, le=100)
    language: Literal["en", "es", "fr", "de"] = "en"
    include_metadata: bool = False
```

Constraints:

- `min_length`, `max_length` - String length
- `ge`, `le` - Greater/less than or equal (numbers)
- `gt`, `lt` - Greater/less than (numbers)
- `Literal` - Restrict to specific values

### Custom Validators

Add custom validation logic:

```python
from pydantic import BaseModel, field_validator

class EmailQuery(BaseModel):
    email: str
    subject: str

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()
```

### Complex Types

Use nested models and collections:

```python
from typing import List, Optional
from datetime import datetime

class Task(BaseModel):
    title: str
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    priority: int = Field(default=1, ge=1, le=5)

class ProjectPlan(BaseModel):
    project_name: str
    tasks: List[Task]
    owner: str

@app
async def create_project(plan: ProjectPlan) -> str:
    Agent.instrument_all()
    # Access nested data
    for task in plan.tasks:
        print(f"Task: {task.title}, Priority: {task.priority}")
    # ...
```

## Structured Output

### Returning Structured Data

Return Pydantic models instead of strings:

```python
from pydantic import BaseModel
from typing import List

class WeatherInfo(BaseModel):
    temperature: float
    condition: str
    humidity: int
    forecast: List[str]

@app
async def weather(location: str) -> WeatherInfo:
    """Get structured weather information."""
    Agent.instrument_all()

    result = await weather_agent.run(
        f"Get weather for {location}",
        result_type=WeatherInfo  # PydanticAI structured output
    )

    # Returns a validated WeatherInfo object
    return result.data
```

### Benefits of Structured Output

1. **Type Safety** - Guaranteed structure
2. **Validation** - Output is validated
3. **Serialization** - Automatically converts to JSON
4. **Documentation** - Clear API contract

### Using Structured Output in UI

The web UI will display structured output as formatted JSON:

```json
{
  "temperature": 72.5,
  "condition": "Sunny",
  "humidity": 45,
  "forecast": [
    "Clear skies today",
    "Slight clouds tomorrow",
    "Rain possible Thursday"
  ]
}
```

## Combining Structured Input and Output

### Full Example

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from pixie import app
from pydantic_ai import Agent

# Input model
class ResearchQuery(BaseModel):
    topic: str = Field(min_length=3)
    depth: int = Field(default=3, ge=1, le=5)
    include_sources: bool = True

# Output model
class ResearchResult(BaseModel):
    summary: str
    key_points: List[str]
    sources: Optional[List[str]] = None
    confidence: float = Field(ge=0.0, le=1.0)

research_agent = Agent("openai:gpt-4o-mini")

@app
async def research(query: ResearchQuery) -> ResearchResult:
    """Conduct research with structured I/O."""
    Agent.instrument_all()

    prompt = f"""
    Research topic: {query.topic}
    Depth level: {query.depth}
    Include sources: {query.include_sources}
    """

    result = await research_agent.run(
        prompt,
        result_type=ResearchResult
    )

    return result.data
```

### Usage in Web UI

1. **Input Form** displays fields:

   - Topic: [text input]
   - Depth: [number input, 1-5]
   - Include Sources: [checkbox]

2. **Output** shows formatted result:
   ```json
   {
     "summary": "...",
     "key_points": ["...", "..."],
     "sources": ["...", "..."],
     "confidence": 0.85
   }
   ```

## Advanced Patterns

### Union Types

Accept multiple input types:

```python
from typing import Union
from pydantic import BaseModel

class TextQuery(BaseModel):
    text: str

class ImageQuery(BaseModel):
    image_url: str
    prompt: str

@app
async def analyze(query: Union[TextQuery, ImageQuery]) -> str:
    Agent.instrument_all()

    if isinstance(query, TextQuery):
        # Handle text
        return await analyze_text(query.text)
    else:
        # Handle image
        return await analyze_image(query.image_url, query.prompt)
```

### Optional Fields

Make fields optional with defaults:

```python
from typing import Optional

class SearchConfig(BaseModel):
    query: str
    # All optional fields
    language: Optional[str] = None
    region: Optional[str] = None
    safe_search: bool = True
    max_results: Optional[int] = None
```

### Computed Fields

Add derived properties:

```python
from pydantic import BaseModel, computed_field

class Task(BaseModel):
    title: str
    estimated_hours: float
    hourly_rate: float

    @computed_field
    @property
    def estimated_cost(self) -> float:
        return self.estimated_hours * self.hourly_rate
```

### Model Configuration

Configure serialization and validation:

```python
from pydantic import BaseModel, ConfigDict

class StrictModel(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,  # Strip whitespace from strings
        str_to_lower=True,          # Convert strings to lowercase
        validate_default=True,      # Validate default values
        frozen=True,                # Make immutable
    )

    name: str
    email: str
```

## Working with Generators

### Streaming with Structured Types

For async generators, use structured types for both yield and send:

```python
from pixie import PixieGenerator, UserInputRequirement

class Message(BaseModel):
    role: str
    content: str
    timestamp: float

class UserResponse(BaseModel):
    message: str
    action: str = "continue"

@app
async def chat(_: None) -> PixieGenerator[Message, UserResponse]:
    Agent.instrument_all()

    # Yield structured messages
    yield Message(
        role="assistant",
        content="Hello! How can I help?",
        timestamp=time.time()
    )

    # Request structured input
    response = yield UserInputRequirement(UserResponse)

    if response.action == "exit":
        yield Message(
            role="assistant",
            content="Goodbye!",
            timestamp=time.time()
        )
```

## Validation Error Handling

### Catching Validation Errors

```python
from pydantic import ValidationError

@app
async def safe_weather(query: WeatherQuery) -> str:
    try:
        Agent.instrument_all()
        # Your logic here
        result = await weather_agent.run(f"Weather for {query.location}")
        return result.output
    except ValidationError as e:
        # Handle validation errors
        return f"Invalid input: {e.errors()}"
```

### Custom Error Messages

```python
from pydantic import BaseModel, Field, field_validator

class StrictQuery(BaseModel):
    location: str = Field(
        min_length=2,
        max_length=100,
        description="City or location name"
    )

    @field_validator('location')
    @classmethod
    def validate_location(cls, v: str) -> str:
        if not v.replace(' ', '').isalpha():
            raise ValueError(
                'Location must contain only letters and spaces'
            )
        return v
```

## Schema Documentation

### Adding Descriptions

Document your models for better API documentation:

```python
from pydantic import BaseModel, Field

class DocumentedQuery(BaseModel):
    """Query model for document search.

    This model defines the parameters for searching documents
    in the knowledge base.
    """

    query: str = Field(
        description="The search query string",
        examples=["machine learning", "python tutorials"]
    )

    max_results: int = Field(
        default=10,
        description="Maximum number of results to return",
        ge=1,
        le=100
    )

    filters: Optional[dict] = Field(
        default=None,
        description="Additional filters to apply"
    )
```

### Viewing Schemas in UI

The web UI automatically displays:

- Field names and types
- Descriptions and examples
- Validation constraints
- Default values

## Best Practices

### 1. Use Descriptive Names

```python
# Good
class UserPreferences(BaseModel):
    theme: Literal["light", "dark"]
    notifications_enabled: bool

# Avoid
class Prefs(BaseModel):
    t: str
    n: bool
```

### 2. Provide Defaults

```python
class Config(BaseModel):
    # Always provide sensible defaults
    timeout: int = 30
    retries: int = 3
    debug: bool = False
```

### 3. Document Everything

```python
class WellDocumented(BaseModel):
    """Clear model description."""

    field: str = Field(
        description="Clear field description",
        examples=["example value"]
    )
```

### 4. Validate Early

```python
@field_validator('email')
@classmethod
def check_email(cls, v: str) -> str:
    # Validate as early as possible
    if '@' not in v:
        raise ValueError('Invalid email')
    return v
```

### 5. Keep Models Simple

```python
# Prefer flat structures when possible
class SimpleModel(BaseModel):
    name: str
    age: int

# Over deeply nested ones
class OverlyNested(BaseModel):
    user: dict[str, dict[str, dict[str, str]]]
```

## Next Steps

- [Interactive Apps](./interactive-app.md) - Build multi-turn conversations
- [Interactive Tools](./interactive-tool.md) - Add interactivity to agent tools
- [Examples](https://github.com/yiouli/pixie-examples) - See more examples

## Common Pitfalls

### Mutable Defaults

```python
# ❌ Wrong - mutable default
class Wrong(BaseModel):
    items: List[str] = []

# ✅ Correct - use default_factory
class Correct(BaseModel):
    items: List[str] = Field(default_factory=list)
```

### Circular Imports

```python
# ❌ Can cause issues
from .other_models import RelatedModel

class MyModel(BaseModel):
    related: RelatedModel

# ✅ Use string annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .other_models import RelatedModel

class MyModel(BaseModel):
    related: 'RelatedModel'
```

### Over-Validation

```python
# ❌ Too restrictive
class TooStrict(BaseModel):
    name: str = Field(regex=r'^[A-Z][a-z]+$')  # Only allows "John"

# ✅ Reasonable validation
class Reasonable(BaseModel):
    name: str = Field(min_length=1, max_length=100)
```
