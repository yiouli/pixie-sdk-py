# App Names and Descriptions

Learn how to configure application metadata for better UI documentation.

## Metadata Sources

Pixie extracts metadata from:

1. **Function name** → Application name
2. **Function docstring** → Application description
3. **Parameter type annotations** → Input schema
4. **Pydantic field metadata** → Field descriptions

## Application Name

The function name becomes the application name:

```python
@app
async def weather_forecast(city: str) -> str:
    """Get weather forecast."""
    return f"Forecast for {city}"
```

Use descriptive, snake_case names like `customer_support_chat` or `generate_sql_query`.

## Application Description

The function docstring becomes the description:

```python
@app
async def sentiment_analyzer(text: str) -> str:
    """Analyze the sentiment of input text.

    Determines emotional tone (positive, negative, neutral)
    using advanced NLP.
    """
    ...
```

First line is the short description. Full docstring provides details.

````

## Parameter Descriptions

Use Pydantic models for rich parameter documentation:

```python
from pydantic import BaseModel, Field

class SearchConfig(BaseModel):
    """Database search configuration."""

    query: str = Field(description="Natural language search query")
    max_results: int = Field(default=10, description="Maximum results")
    include_archived: bool = Field(default=False, description="Include archived records")

@app
async def search_database(config: SearchConfig) -> str:
    """Search the database."""
    ...
````

Benefits: field-level descriptions, default values, type validation, rich UI forms.

## Field Documentation

Always add descriptions to Pydantic fields:

```python
class Config(BaseModel):
    temperature: float = Field(
        default=0.7,
        description="Model temperature (0.0-1.0). Higher = more creative"
    )
```

Document constraints in descriptions:

```python
class SearchQuery(BaseModel):
    query: str = Field(
        min_length=3,
        max_length=200,
        description="Search query (3-200 characters)"
    )
```
