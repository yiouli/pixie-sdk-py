# GitHub Copilot Instructions for pixie-sdk-py

## Test-Driven Development (TDD) Requirements

This project follows strict TDD practices to ensure code quality and maintainability.

### 1. Test-First Development

**CRITICAL**: Always write or verify tests BEFORE implementing features or making changes.

**Development Workflow**:
1. **Understand the requirement**: Clarify what needs to be built or changed
2. **Write the test first**: Create test cases that define expected behavior
3. **Run the test**: Verify it fails (red) - this confirms the test is valid
4. **Implement the code**: Write minimal code to make the test pass
5. **Run the test again**: Verify it passes (green)
6. **Refactor if needed**: Improve code while keeping tests green
7. **Run all tests**: Ensure no regressions

### 2. Test Location and Organization

**All tests for pixie module code must be in the `tests/pixie/` directory:**

```
pixie/
  sdk.py
  utils.py
  agents/
    base.py
    openai_agent.py
  storage/
    database.py

tests/
  pixie/                      # All pixie module tests go here
    test_sdk.py
    test_utils.py
    agents/
      test_base.py
      test_openai_agent.py
    storage/
      test_database.py
```

**Test file naming:**
- Test files must start with `test_` prefix
- Mirror the structure of the source code directory
- One test file per source module

### 3. Test Coverage Requirements

- **All new code must have tests**: Functions, classes, methods, utilities
- **Bug fixes must include regression tests**: Add a test that would have caught the bug
- **Refactoring must maintain tests**: Verify all existing tests still pass

**What to test:**
- Function inputs and outputs
- Edge cases and boundary conditions
- Error handling and exceptions
- Integration between components
- Database operations (use fixtures for setup/teardown)
- API endpoints and responses
- Agent behavior and decision making

### 4. Running Tests

Before committing changes, always run:

```bash
pytest                       # Run all tests
pytest tests/pixie/          # Run only pixie tests
pytest tests/pixie/test_sdk.py  # Run specific test file
pytest -k "test_function_name"  # Run specific test
pytest --cov=pixie           # Run with coverage report
```

### 5. Test Quality Guidelines

**Good tests are:**
- **Focused**: Test one thing at a time
- **Independent**: Don't depend on other tests or execution order
- **Readable**: Clear arrange-act-assert structure
- **Fast**: Use mocks/fixtures for external dependencies
- **Maintainable**: Easy to update when requirements change

**Example test structure:**

```python
import pytest
from pixie.sdk import PixieSDK
from pixie.types import AgentConfig

class TestPixieSDK:
    """Tests for PixieSDK class."""

    def test_initialize_with_valid_config(self):
        # Arrange
        config = AgentConfig(name="test-agent")

        # Act
        sdk = PixieSDK(config)

        # Assert
        assert sdk.config.name == "test-agent"
        assert sdk.is_initialized

    def test_raises_error_on_invalid_config(self):
        # Arrange
        invalid_config = None

        # Act & Assert
        with pytest.raises(ValueError, match="Config cannot be None"):
            PixieSDK(invalid_config)

    @pytest.fixture
    def sdk_with_database(self, tmp_path):
        """Fixture providing SDK with temporary database."""
        db_path = tmp_path / "test.db"
        sdk = PixieSDK(database_path=str(db_path))
        yield sdk
        sdk.cleanup()

    def test_store_and_retrieve_data(self, sdk_with_database):
        # Arrange
        test_data = {"key": "value"}

        # Act
        sdk_with_database.store(test_data)
        result = sdk_with_database.retrieve("key")

        # Assert
        assert result == "value"
```

### 6. Fixtures and Test Utilities

Use pytest fixtures for common setup:

```python
# In tests/pixie/conftest.py
import pytest
from pixie.sdk import PixieSDK

@pytest.fixture
def temp_database(tmp_path):
    """Provide temporary database for testing."""
    db_path = tmp_path / "test.db"
    yield str(db_path)

@pytest.fixture
def pixie_sdk(temp_database):
    """Provide configured PixieSDK instance."""
    sdk = PixieSDK(database_path=temp_database)
    yield sdk
    sdk.cleanup()
```

## Type Safety Requirements

This project uses Python type hints with strict type checking via mypy.

### 1. Always Run Type Checking

Before committing, verify there are no type errors:

```bash
mypy pixie/                  # Type check pixie module
mypy tests/pixie/            # Type check tests
mypy .                       # Type check entire project
```

Run this command after making changes to ensure type safety.

### 2. Type Annotation Rules

**CRITICAL**: All function signatures must have type annotations for parameters and return values.

**❌ WRONG** - Missing type annotations:

```python
def process_data(data, config):
    return data.process(config)

def calculate(x, y):  # No return type
    return x + y
```

**✅ CORRECT** - Proper type annotations:

```python
from typing import Dict, List, Optional, Union
from pixie.types import Config, ProcessedData

def process_data(data: Dict[str, str], config: Config) -> ProcessedData:
    return data.process(config)

def calculate(x: int, y: int) -> int:
    return x + y

def fetch_user(user_id: str) -> Optional[User]:
    """Returns User if found, None otherwise."""
    return database.get_user(user_id)
```

### 3. Type Safety for Common Patterns

**Optional values:**

```python
from typing import Optional

def get_config(name: str) -> Optional[Config]:
    """Returns Config or None if not found."""
    return config_map.get(name)

# Usage
config = get_config("my-config")
if config is not None:
    # Type checker knows config is Config here
    config.validate()
```

**Union types:**

```python
from typing import Union

def process_input(value: Union[str, int, List[str]]) -> str:
    if isinstance(value, str):
        return value
    elif isinstance(value, int):
        return str(value)
    else:
        return ", ".join(value)
```

**Generic types:**

```python
from typing import TypeVar, List, Callable

T = TypeVar('T')

def filter_items(items: List[T], predicate: Callable[[T], bool]) -> List[T]:
    """Filter items using predicate function."""
    return [item for item in items if predicate(item)]

# Type-safe usage
numbers = filter_items([1, 2, 3, 4], lambda x: x > 2)
strings = filter_items(["a", "b", "c"], lambda x: x != "b")
```

**Dictionary and data structures:**

```python
from typing import Dict, List, Any, TypedDict

# Use TypedDict for structured dictionaries
class UserData(TypedDict):
    id: str
    name: str
    email: str
    roles: List[str]

def create_user(data: UserData) -> User:
    return User(**data)

# For flexible dictionaries
def process_metadata(metadata: Dict[str, Any]) -> None:
    """Process metadata with string keys and any value type."""
    for key, value in metadata.items():
        handle_metadata(key, value)
```

### 4. Class Type Annotations

All class attributes and methods must be typed:

```python
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Agent:
    """AI Agent configuration and state."""

    name: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    history: List[str] = None  # ❌ WRONG: mutable default

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []

    def add_message(self, message: str) -> None:
        """Add message to history."""
        self.history.append(message)

    def get_context(self) -> str:
        """Get conversation context."""
        return "\n".join(self.history)


# Better approach with field factory
from dataclasses import dataclass, field

@dataclass
class Agent:
    """AI Agent configuration and state."""

    name: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    history: List[str] = field(default_factory=list)  # ✅ CORRECT

    def add_message(self, message: str) -> None:
        """Add message to history."""
        self.history.append(message)
```

### 5. Async Type Annotations

For async functions, use proper return type annotations:

```python
from typing import List, Optional
import asyncio

async def fetch_data(url: str) -> Dict[str, Any]:
    """Fetch data from URL asynchronously."""
    response = await http_client.get(url)
    return response.json()

async def process_batch(items: List[str]) -> List[ProcessedItem]:
    """Process items in parallel."""
    tasks = [process_item(item) for item in items]
    results = await asyncio.gather(*tasks)
    return results
```

### 6. Protocol and Abstract Base Classes

Use Protocol for structural typing and ABC for inheritance:

```python
from typing import Protocol
from abc import ABC, abstractmethod

# Protocol for structural typing (duck typing with types)
class Runnable(Protocol):
    """Protocol for objects that can be run."""

    def run(self) -> None:
        """Run the object."""
        ...

# Abstract base class for inheritance
class Agent(ABC):
    """Abstract base class for agents."""

    @abstractmethod
    def execute(self, task: str) -> str:
        """Execute task and return result."""
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities."""
        pass
```

### 7. Avoid Type Checking Bypass

**❌ NEVER do this:**

```python
data = fetch_data()  # type: ignore
result: Any = process(data)
value = cast(str, unknown_value)  # Avoid cast unless absolutely necessary
```

**✅ CORRECT approach:**

```python
# Use proper type guards
def is_valid_config(obj: Any) -> TypeGuard[Config]:
    return isinstance(obj, dict) and 'name' in obj and 'model' in obj

data = fetch_data()
if is_valid_config(data):
    # Type checker knows data is Config here
    process_config(data)
```

## Code Quality Tools

This project uses several tools to maintain code quality:

### 1. Linting and Formatting

```bash
ruff check .                 # Run linter
ruff format .                # Format code
pylint pixie/                # Additional linting
```

### 2. Pre-commit Checks

Before committing, run:

```bash
pytest                       # All tests must pass
mypy .                       # Zero type errors
ruff check .                 # No linting errors
```

## Summary

**Before every commit:**
1. ✅ Write/update tests in `tests/pixie/` for your changes
2. ✅ Run `pytest` - all tests must pass
3. ✅ Run `mypy .` - zero type errors allowed
4. ✅ Run `ruff check .` - no linting errors
5. ✅ Verify functionality works as expected

**Development cycle:**
1. Write test first in `tests/pixie/` (TDD)
2. Implement feature with proper type annotations
3. Run tests (`pytest`)
4. Run type checking (`mypy .`)
5. Run linting (`ruff check .`)
6. Fix any issues
7. Commit

Following these practices ensures high code quality, type safety, maintainability, and reliability.
