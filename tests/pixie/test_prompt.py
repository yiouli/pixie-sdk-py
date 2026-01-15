"""Comprehensive unit tests for pixie.prompts.prompt module."""

import os
import json
import pytest
import tempfile
from types import NoneType

from pixie.prompts.storage import FilePromptStorage
from pixie.prompts.prompt import (
    Prompt,
    PromptVariables,
    update_prompt_registry,
    UntypedPrompt,
    _prompt_registry,  # Import the registry explicitly
)


class SamplePromptVariables(PromptVariables):
    """Sample subclass of PromptVariables for testing."""

    name: str
    age: int
    city: str = "Unknown"


class AnotherPromptVariables(PromptVariables):
    """Another sample subclass with different fields."""

    greeting: str
    topic: str


class TestPromptInitialization:
    """Tests for Prompt initialization."""

    def test_init_with_string_versions(self):
        """Test that a string version is converted to a dict with 'default' key."""
        prompt = Prompt(versions="Hello, world!")

        assert isinstance(prompt._versions, dict)
        assert "default" in prompt._versions
        assert prompt._versions["default"] == "Hello, world!"

    def test_init_with_dict_versions(self):
        """Test initialization with a dictionary of versions."""
        versions = {"v1": "Version 1", "v2": "Version 2"}
        prompt = Prompt(versions=versions)

        assert prompt._versions == versions
        assert prompt.version_ids == {"v1", "v2"}

    def test_init_with_dict_versions_creates_copy(self):
        """Test that the versions dict is deep copied to prevent external mutations."""
        original_versions = {"v1": "Version 1"}
        prompt = Prompt(versions=original_versions)

        # Modify the original dict
        original_versions["v1"] = "Modified"
        original_versions["v2"] = "New version"

        # Prompt should still have the original value
        assert prompt._versions["v1"] == "Version 1"
        assert "v2" not in prompt._versions

    def test_init_with_explicit_default_version(self):
        """Test setting an explicit default version."""
        versions = {"v1": "Version 1", "v2": "Version 2", "v3": "Version 3"}
        prompt = Prompt(versions=versions, default_version_id="v2")

        assert prompt.default_version_id == "v2"

    def test_init_default_version_is_first_key_when_not_specified(self):
        """Test that default version is the first key when not explicitly set."""
        # Note: dict order is preserved in Python 3.7+
        versions = {"first": "First version", "second": "Second version"}
        prompt = Prompt(versions=versions)

        assert prompt.default_version_id == "first"

    def test_init_with_string_and_default_version(self):
        """Test that default_version_id works with string versions."""
        prompt = Prompt(versions="Hello!", default_version_id="default")

        assert prompt.default_version_id == "default"

    def test_init_with_variable_definitions(self):
        """Test initialization with variable definitions."""
        prompt = Prompt(
            versions="Hello, {name}!",
            variable_definitions=SamplePromptVariables,
        )

        assert prompt._variable_definitions == SamplePromptVariables

    def test_init_with_none_variable_definitions(self):
        """Test initialization with NoneType variable definitions (default)."""
        from types import NoneType

        prompt = Prompt(versions="Hello!")

        assert prompt._variable_definitions == NoneType


class TestPromptProperties:
    """Tests for Prompt properties."""

    def test_version_ids_property(self):
        """Test the version_ids property returns all version keys."""
        versions = {"v1": "Version 1", "v2": "Version 2", "v3": "Version 3"}
        prompt = Prompt(versions=versions)

        version_ids = prompt.version_ids

        assert isinstance(version_ids, set)
        assert version_ids == {"v1", "v2", "v3"}

    def test_version_ids_with_single_version(self):
        """Test version_ids with a single version."""
        prompt = Prompt(versions="Single version")

        assert prompt.version_ids == {"default"}

    def test_default_version_id_property(self):
        """Test the default_version_id property."""
        versions = {"v1": "Version 1", "v2": "Version 2"}
        prompt = Prompt(versions=versions, default_version_id="v2")

        assert prompt.default_version_id == "v2"

    def test_default_version_id_property_with_string_init(self):
        """Test default_version_id when initialized with string."""
        prompt = Prompt(versions="Test prompt")

        assert prompt.default_version_id == "default"


class TestPromptCompileWithoutVariables:
    """Tests for Prompt.compile() without variables (NoneType case)."""

    def test_compile_without_variables_default_version(self):
        """Test compiling a prompt without variables using default version."""
        prompt = Prompt(versions="Hello, world!")

        result = prompt.compile()

        assert result == "Hello, world!"

    def test_compile_without_variables_specific_version(self):
        """Test compiling a prompt without variables using a specific version."""
        versions = {
            "formal": "Good day, esteemed user.",
            "casual": "Hey there!",
            "excited": "Hello!!!",
        }
        prompt = Prompt(versions=versions, default_version_id="casual")

        result_formal = prompt.compile(version_id="formal")
        result_casual = prompt.compile(version_id="casual")
        result_excited = prompt.compile(version_id="excited")

        assert result_formal == "Good day, esteemed user."
        assert result_casual == "Hey there!"
        assert result_excited == "Hello!!!"

    def test_compile_without_variables_uses_default_when_no_version_specified(self):
        """Test that compile uses default version when version_id is None."""
        versions = {"v1": "Version 1", "v2": "Version 2"}
        prompt = Prompt(versions=versions, default_version_id="v2")

        result = prompt.compile()

        assert result == "Version 2"

    def test_compile_plain_text_no_formatting(self):
        """Test compiling plain text without any formatting placeholders."""
        text = "This is a plain text prompt with no variables."
        prompt = Prompt(versions=text)

        result = prompt.compile()

        assert result == text


class TestPromptCompileWithVariables:
    """Tests for Prompt.compile() with variables."""

    def test_compile_with_variables(self):
        """Test compiling a prompt with variable substitution."""
        template = "Hello, {name}! You are {age} years old."
        prompt = Prompt(
            versions=template,
            variable_definitions=SamplePromptVariables,
        )

        variables = SamplePromptVariables(name="Alice", age=30)
        result = prompt.compile(variables)

        assert result == "Hello, Alice! You are 30 years old."

    def test_compile_with_variables_and_default_values(self):
        """Test that default values in Pydantic model work correctly."""
        template = "Hello, {name} from {city}!"
        prompt = Prompt(
            versions=template,
            variable_definitions=SamplePromptVariables,
        )

        variables = SamplePromptVariables(name="Bob", age=25)
        result = prompt.compile(variables)

        assert result == "Hello, Bob from Unknown!"

    def test_compile_with_variables_specific_version(self):
        """Test compiling with variables using a specific version."""
        versions = {
            "greeting": "Hello, {name}!",
            "farewell": "Goodbye, {name}!",
            "question": "How are you, {name}?",
        }
        prompt = Prompt(
            versions=versions,
            variable_definitions=SamplePromptVariables,
            default_version_id="greeting",
        )

        variables = SamplePromptVariables(name="Charlie", age=35)

        greeting = prompt.compile(variables, version_id="greeting")
        farewell = prompt.compile(variables, version_id="farewell")
        question = prompt.compile(variables, version_id="question")

        assert greeting == "Hello, Charlie!"
        assert farewell == "Goodbye, Charlie!"
        assert question == "How are you, Charlie?"

    def test_compile_with_multiple_variables(self):
        """Test compiling with multiple variable substitutions."""
        template = "{greeting}, {topic} is fascinating!"
        prompt = Prompt(
            versions=template,
            variable_definitions=AnotherPromptVariables,
        )

        variables = AnotherPromptVariables(greeting="Hello", topic="Python")
        result = prompt.compile(variables)

        assert result == "Hello, Python is fascinating!"

    def test_compile_with_variables_complex_template(self):
        """Test compiling with a more complex template."""
        template = """
Name: {name}
Age: {age}
City: {city}
Status: Active
"""
        prompt = Prompt(
            versions=template,
            variable_definitions=SamplePromptVariables,
        )

        variables = SamplePromptVariables(name="Diana", age=28, city="Paris")
        result = prompt.compile(variables)

        expected = """
Name: Diana
Age: 28
City: Paris
Status: Active
"""
        assert result == expected

    def test_compile_with_variables_uses_default_version(self):
        """Test that compile with variables uses default version when not specified."""
        versions = {
            "v1": "Version 1: {name}",
            "v2": "Version 2: {name}",
        }
        prompt = Prompt(
            versions=versions,
            variable_definitions=SamplePromptVariables,
            default_version_id="v2",
        )

        variables = SamplePromptVariables(name="Eve", age=40)
        result = prompt.compile(variables)

        assert result == "Version 2: Eve"

    def test_compile_variables_required_when_definitions_exist(self):
        """Test that ValueError is raised when variables are required but not provided."""
        prompt = Prompt(
            versions="Hello, {name}!",
            variable_definitions=SamplePromptVariables,
        )

        with pytest.raises(ValueError):
            prompt.compile()  # type: ignore[call-arg]

    def test_compile_variables_required_with_none_passed(self):
        """Test that ValueError is raised when None is explicitly passed."""
        prompt = Prompt(
            versions="Hello, {name}!",
            variable_definitions=SamplePromptVariables,
        )

        with pytest.raises(ValueError):
            prompt.compile(None)  # type: ignore[arg-type]


class TestPromptEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_string_version(self):
        """Test with an empty string raises ValueError."""
        # Empty string is falsy, so it should fail validation
        with pytest.raises(ValueError, match="No versions provided"):
            Prompt(versions="")

    def test_empty_dict_versions(self):
        """Test with an empty dict raises ValueError (no versions available)."""
        # This creates a prompt with no versions, which fails validation
        with pytest.raises(ValueError, match="No versions provided"):
            Prompt(versions={})

    def test_version_with_special_characters(self):
        """Test version content with special characters."""
        special_text = "Hello! @#$%^&*() {name} [brackets] 'quotes' \"double\""
        prompt = Prompt(
            versions=special_text,
            variable_definitions=SamplePromptVariables,
        )

        variables = SamplePromptVariables(name="Test", age=20)
        result = prompt.compile(variables)

        assert result == "Hello! @#$%^&*() Test [brackets] 'quotes' \"double\""

    def test_version_with_curly_braces_not_variables(self):
        """Test that literal curly braces (doubled) are preserved."""
        template = "This {{is}} not {name} a variable"
        prompt = Prompt(
            versions=template,
            variable_definitions=SamplePromptVariables,
        )

        variables = SamplePromptVariables(name="formatted", age=10)
        result = prompt.compile(variables)

        # Python's .format() handles {{}} as escaped braces
        assert result == "This {is} not formatted a variable"

    def test_unicode_content(self):
        """Test with Unicode characters in version content."""
        unicode_text = "Hello, {name}! 你好 🎉 Привет"
        prompt = Prompt(
            versions=unicode_text,
            variable_definitions=SamplePromptVariables,
        )

        variables = SamplePromptVariables(name="World", age=1)
        result = prompt.compile(variables)

        assert result == "Hello, World! 你好 🎉 Привет"

    def test_multiline_template(self):
        """Test with a multiline template."""
        template = """Line 1: {name}
Line 2: {age}
Line 3: {city}"""
        prompt = Prompt(
            versions=template,
            variable_definitions=SamplePromptVariables,
        )

        variables = SamplePromptVariables(name="Frank", age=50, city="London")
        result = prompt.compile(variables)

        expected = """Line 1: Frank
Line 2: 50
Line 3: London"""
        assert result == expected

    def test_version_id_lookup_key_error_propagates(self):
        """Test that KeyError is raised when version_id doesn't exist."""
        prompt = Prompt(versions={"v1": "Version 1"})

        with pytest.raises(KeyError):
            prompt.compile(version_id="nonexistent")

    def test_missing_variable_in_template_raises_key_error(self):
        """Test that KeyError is raised when template variable is not in model."""
        template = "Hello, {name} and {missing_var}!"
        prompt = Prompt(
            versions=template,
            variable_definitions=SamplePromptVariables,
        )

        variables = SamplePromptVariables(name="Test", age=25)

        with pytest.raises(KeyError):
            prompt.compile(variables)

    def test_extra_variables_in_model_dont_affect_compile(self):
        """Test that extra variables in the model that aren't in template are ignored."""
        template = "Hello, {name}!"
        prompt = Prompt(
            versions=template,
            variable_definitions=SamplePromptVariables,
        )

        # Model has age and city, but template only uses name
        variables = SamplePromptVariables(name="Grace", age=60, city="Tokyo")
        result = prompt.compile(variables)

        assert result == "Hello, Grace!"

    def test_numeric_values_in_template(self):
        """Test that numeric values are properly converted to strings."""
        template = "Count: {age}, Double: {age}"
        prompt = Prompt(
            versions=template,
            variable_definitions=SamplePromptVariables,
        )

        variables = SamplePromptVariables(name="Test", age=42)
        result = prompt.compile(variables)

        assert result == "Count: 42, Double: 42"


class TestPromptTypeAnnotations:
    """Tests related to type annotations and generic behavior."""

    def test_prompt_with_nonetype_generic(self):
        """Test Prompt with NoneType explicitly."""
        from types import NoneType

        prompt: Prompt[NoneType] = Prompt(versions="No variables here")

        result = prompt.compile()

        assert result == "No variables here"

    def test_prompt_with_custom_type_generic(self):
        """Test Prompt with custom PromptVariables type."""
        prompt: Prompt[SamplePromptVariables] = Prompt(
            versions="Name: {name}",
            variable_definitions=SamplePromptVariables,
        )

        variables = SamplePromptVariables(name="Type Test", age=99)
        result = prompt.compile(variables)

        assert result == "Name: Type Test"


class TestPromptIntegration:
    """Integration tests combining multiple features."""

    def test_multiple_versions_with_variables(self):
        """Test using multiple versions with the same variable definitions."""
        versions = {
            "short": "{greeting}!",
            "medium": "{greeting}, let's talk about {topic}.",
            "long": "{greeting}! Today we'll discuss {topic} in detail.",
        }
        prompt = Prompt(
            versions=versions,
            variable_definitions=AnotherPromptVariables,
            default_version_id="medium",
        )

        variables = AnotherPromptVariables(greeting="Hi", topic="AI")

        short = prompt.compile(variables, version_id="short")
        medium = prompt.compile(variables)  # Uses default
        long = prompt.compile(variables, version_id="long")

        assert short == "Hi!"
        assert medium == "Hi, let's talk about AI."
        assert long == "Hi! Today we'll discuss AI in detail."

    def test_switching_versions_maintains_state(self):
        """Test that switching versions doesn't affect internal state."""
        versions = {"v1": "Version 1", "v2": "Version 2"}
        prompt = Prompt(versions=versions, default_version_id="v1")

        result1 = prompt.compile(version_id="v2")
        result2 = prompt.compile()  # Should still use v1 as default

        assert result1 == "Version 2"
        assert result2 == "Version 1"
        assert prompt.default_version_id == "v1"

    def test_same_prompt_compiled_multiple_times(self):
        """Test that compiling the same prompt multiple times produces consistent results."""
        template = "Hello, {name}!"
        prompt = Prompt(
            versions=template,
            variable_definitions=SamplePromptVariables,
        )

        variables = SamplePromptVariables(name="Harry", age=45)

        results = [prompt.compile(variables) for _ in range(5)]

        assert all(r == "Hello, Harry!" for r in results)

    def test_different_variable_instances_same_values(self):
        """Test that different variable instances with same values produce same output."""
        prompt = Prompt(
            versions="{name} is {age}",
            variable_definitions=SamplePromptVariables,
        )

        vars1 = SamplePromptVariables(name="Ivy", age=33)
        vars2 = SamplePromptVariables(name="Ivy", age=33)

        result1 = prompt.compile(vars1)
        result2 = prompt.compile(vars2)

        assert result1 == result2
        assert result1 == "Ivy is 33"


class TestUpdatePromptRegistry:
    """Tests for the update_prompt_registry function."""

    def test_update_prompt_registry_new_prompt(self):
        """Test that update_prompt_registry handles new prompts."""
        new_prompt = UntypedPrompt(versions={"v1": "New version"})

        result = update_prompt_registry(new_prompt)

        assert result.id == new_prompt.id
        assert result.versions["v1"] == "New version"
        assert result.variable_definitions == NoneType
        assert result.is_valid

    def test_update_prompt_registry_existing_prompt(self):
        """Test that update_prompt_registry updates existing prompts."""
        original_prompt = UntypedPrompt(versions={"v1": "Original version"})
        # Manually add to registry with variable definitions
        _prompt_registry[original_prompt.id] = Prompt.from_untyped(
            original_prompt, variable_definitions=SamplePromptVariables
        )

        # Temporarily remove from registry to create updated UntypedPrompt
        del _prompt_registry[original_prompt.id]

        updated_prompt = UntypedPrompt(
            id=original_prompt.id, versions={"v1": "Updated version"}
        )

        # Add back to registry to simulate existing
        _prompt_registry[original_prompt.id] = Prompt.from_untyped(
            original_prompt, variable_definitions=SamplePromptVariables
        )

        result = update_prompt_registry(updated_prompt)

        assert result.id == original_prompt.id
        assert result.versions["v1"] == "Updated version"
        assert (
            result.variable_definitions == SamplePromptVariables
        )  # Should retain original var_def
        assert result.is_valid


@pytest.mark.asyncio
class TestFilePromptStorage:
    """Tests for the FilePromptStorage class."""

    async def test_save_new_prompt(self):
        """Test saving a new prompt to storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FilePromptStorage(directory=temp_dir)

            new_prompt = UntypedPrompt(versions={"v1": "New version"})
            is_new = await storage.save(new_prompt)  # Use await for async call

            assert is_new
            assert os.path.exists(os.path.join(temp_dir, f"{new_prompt.id}.json"))

    async def test_save_existing_prompt(self):
        """Test saving an existing prompt updates its data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FilePromptStorage(directory=temp_dir)

            prompt = UntypedPrompt(versions={"v1": "Initial version"})
            await storage.save(prompt)  # Use await for async call

            # Temporarily remove from registry to create updated UntypedPrompt
            del _prompt_registry[prompt.id]

            # Update the prompt
            updated_prompt = UntypedPrompt(
                id=prompt.id, versions={"v1": "Updated version"}
            )

            is_new = await storage.save(updated_prompt)  # Use await for async call

            assert not is_new

            # Verify the file content
            filepath = os.path.join(temp_dir, f"{prompt.id}.json")
            with open(filepath, "r") as f:
                data = json.load(f)

            assert data["versions"]["v1"] == "Updated version"
