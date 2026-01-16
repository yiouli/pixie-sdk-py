"""Comprehensive unit tests for pixie.prompts.prompt module."""

import os
import json
import pytest
import tempfile
from types import NoneType

from pixie.prompts.storage import _FilePromptStorage
from pixie.prompts.prompt import (
    DEFAULT_VERSION_ID,
    BasePrompt,
    PromptVariables,
    BaseUntypedPrompt,
    OutdatedPrompt,
    _prompt_registry,  # Import the registry explicitly
    _compiled_prompt_registry,
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
        prompt = BasePrompt(versions="Hello, world!")

        assert isinstance(prompt._versions, dict)
        assert DEFAULT_VERSION_ID in prompt._versions
        assert prompt._versions[DEFAULT_VERSION_ID] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_init_with_dict_versions(self):
        """Test initialization with a dictionary of versions."""
        versions = {"v1": "Version 1", "v2": "Version 2"}
        prompt = BasePrompt(versions=versions)

        assert prompt._versions == versions
        assert set(prompt._versions.keys()) == {"v1", "v2"}

    def test_init_with_dict_versions_creates_copy(self):
        """Test that the versions dict is deep copied to prevent external mutations."""
        original_versions = {"v1": "Version 1"}
        prompt = BasePrompt(versions=original_versions)

        # Modify the original dict
        original_versions["v1"] = "Modified"
        original_versions["v2"] = "New version"

        # Prompt should still have the original value
        assert prompt._versions["v1"] == "Version 1"
        assert "v2" not in prompt._versions

    @pytest.mark.asyncio
    async def test_init_with_explicit_default_version(self):
        """Test setting an explicit default version."""
        versions = {"v1": "Version 1", "v2": "Version 2", "v3": "Version 3"}
        prompt = BasePrompt(versions=versions, default_version_id="v2")

        assert await prompt.get_default_version_id() == "v2"

    @pytest.mark.asyncio
    async def test_init_default_version_is_first_key_when_not_specified(self):
        """Test that default version is the first key when not explicitly set."""
        # Note: dict order is preserved in Python 3.7+
        versions = {"first": "First version", "second": "Second version"}
        prompt = BasePrompt(versions=versions)

        assert await prompt.get_default_version_id() == "first"

    @pytest.mark.asyncio
    async def test_init_with_string_and_default_version(self):
        """Test that default_version_id works with string versions."""
        prompt = BasePrompt(versions="Hello!", default_version_id="default")

        assert await prompt.get_default_version_id() == "default"

    def test_init_with_variables_definition(self):
        """Test initialization with variable definitions."""
        prompt = BasePrompt(
            versions="Hello, {name}!",
            variables_definition=SamplePromptVariables,
        )

        assert prompt._variables_definition == SamplePromptVariables

    def test_init_with_none_variables_definition(self):
        """Test initialization with NoneType variable definitions (default)."""

        prompt = BasePrompt(versions="Hello!")

        assert prompt._variables_definition == NoneType


class TestPromptProperties:
    """Tests for Prompt properties."""

    @pytest.mark.asyncio
    async def test_version_ids_property(self):
        """Test the version_ids property returns all version keys."""
        versions = {"v1": "Version 1", "v2": "Version 2", "v3": "Version 3"}
        prompt = BasePrompt(versions=versions)

        version_ids = set(prompt._versions.keys())

        assert isinstance(version_ids, set)
        assert version_ids == {"v1", "v2", "v3"}

    @pytest.mark.asyncio
    async def test_version_ids_with_single_version(self):
        """Test version_ids with a single version."""
        prompt = BasePrompt(versions="Single version")

        assert set(prompt._versions.keys()) == {DEFAULT_VERSION_ID}

    @pytest.mark.asyncio
    async def test_default_version_id_property(self):
        """Test the default_version_id property."""
        versions = {"v1": "Version 1", "v2": "Version 2"}
        prompt = BasePrompt(versions=versions, default_version_id="v2")

        assert await prompt.get_default_version_id() == "v2"

    @pytest.mark.asyncio
    async def test_default_version_id_property_with_string_init(self):
        """Test default_version_id when initialized with string."""
        prompt = BasePrompt(versions="Test prompt")

        assert await prompt.get_default_version_id() == DEFAULT_VERSION_ID


class TestPromptCompileWithoutVariables:
    """Tests for Prompt.compile() without variables (NoneType case)."""

    def test_compile_without_variables_default_version(self):
        """Test compiling a prompt without variables using default version."""
        prompt = BasePrompt(versions="Hello, world!")

        result = prompt.compile()

        assert result == "Hello, world!"

    def test_compile_without_variables_specific_version(self):
        """Test compiling a prompt without variables using a specific version."""
        versions = {
            "formal": "Good day, esteemed user.",
            "casual": "Hey there!",
            "excited": "Hello!!!",
        }
        prompt = BasePrompt(versions=versions, default_version_id="casual")

        result_formal = prompt.compile(version_id="formal")
        result_casual = prompt.compile(version_id="casual")
        result_excited = prompt.compile(version_id="excited")

        assert result_formal == "Good day, esteemed user."
        assert result_casual == "Hey there!"
        assert result_excited == "Hello!!!"

    def test_compile_without_variables_uses_default_when_no_version_specified(self):
        """Test that compile uses default version when version_id is None."""
        versions = {"v1": "Version 1", "v2": "Version 2"}
        prompt = BasePrompt(versions=versions, default_version_id="v2")

        result = prompt.compile()

        assert result == "Version 2"

    def test_compile_plain_text_no_formatting(self):
        """Test compiling plain text without any formatting placeholders."""
        text = "This is a plain text prompt with no variables."
        prompt = BasePrompt(versions=text)

        result = prompt.compile()

        assert result == text


class TestPromptCompileWithVariables:
    """Tests for Prompt.compile() with variables."""

    def test_compile_with_variables(self):
        """Test compiling a prompt with variable substitution."""
        template = "Hello, {name}! You are {age} years old."
        prompt = BasePrompt(
            versions=template,
            variables_definition=SamplePromptVariables,
        )

        variables = SamplePromptVariables(name="Alice", age=30)
        result = prompt.compile(variables)

        assert result == "Hello, Alice! You are 30 years old."

    def test_compile_with_variables_and_default_values(self):
        """Test that default values in Pydantic model work correctly."""
        template = "Hello, {name} from {city}!"
        prompt = BasePrompt(
            versions=template,
            variables_definition=SamplePromptVariables,
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
        prompt = BasePrompt(
            versions=versions,
            variables_definition=SamplePromptVariables,
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
        prompt = BasePrompt(
            versions=template,
            variables_definition=AnotherPromptVariables,
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
        prompt = BasePrompt(
            versions=template,
            variables_definition=SamplePromptVariables,
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
        prompt = BasePrompt(
            versions=versions,
            variables_definition=SamplePromptVariables,
            default_version_id="v2",
        )

        variables = SamplePromptVariables(name="Eve", age=40)
        result = prompt.compile(variables)

        assert result == "Version 2: Eve"

    def test_compile_variables_required_when_definitions_exist(self):
        """Test that ValueError is raised when variables are required but not provided."""
        prompt = BasePrompt(
            versions="Hello, {name}!",
            variables_definition=SamplePromptVariables,
        )

        with pytest.raises(ValueError):
            prompt.compile()  # type: ignore[call-arg]

    def test_compile_variables_required_with_none_passed(self):
        """Test that ValueError is raised when None is explicitly passed."""
        prompt = BasePrompt(
            versions="Hello, {name}!",
            variables_definition=SamplePromptVariables,
        )

        with pytest.raises(ValueError):
            prompt.compile(None)  # type: ignore[arg-type]


class TestPromptEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_string_version(self):
        """Test with an empty string raises ValueError."""
        # Empty string is falsy, so it should fail validation
        with pytest.raises(ValueError, match="No versions provided"):
            BasePrompt(versions="")

    def test_empty_dict_versions(self):
        """Test with an empty dict raises ValueError (no versions available)."""
        # This creates a prompt with no versions, which fails validation
        with pytest.raises(ValueError, match="No versions provided"):
            BasePrompt(versions={})

    def test_version_with_special_characters(self):
        """Test version content with special characters."""
        special_text = "Hello! @#$%^&*() {name} [brackets] 'quotes' \"double\""
        prompt = BasePrompt(
            versions=special_text,
            variables_definition=SamplePromptVariables,
        )

        variables = SamplePromptVariables(name="Test", age=20)
        result = prompt.compile(variables)

        assert result == "Hello! @#$%^&*() Test [brackets] 'quotes' \"double\""

    def test_version_with_curly_braces_not_variables(self):
        """Test that literal curly braces (doubled) are preserved."""
        template = "This {{is}} not {name} a variable"
        prompt = BasePrompt(
            versions=template,
            variables_definition=SamplePromptVariables,
        )

        variables = SamplePromptVariables(name="formatted", age=10)
        result = prompt.compile(variables)

        # Python's .format() handles {{}} as escaped braces
        assert result == "This {is} not formatted a variable"

    def test_unicode_content(self):
        """Test with Unicode characters in version content."""
        unicode_text = "Hello, {name}! 你好 🎉 Привет"
        prompt = BasePrompt(
            versions=unicode_text,
            variables_definition=SamplePromptVariables,
        )

        variables = SamplePromptVariables(name="World", age=1)
        result = prompt.compile(variables)

        assert result == "Hello, World! 你好 🎉 Привет"

    def test_multiline_template(self):
        """Test with a multiline template."""
        template = """Line 1: {name}
Line 2: {age}
Line 3: {city}"""
        prompt = BasePrompt(
            versions=template,
            variables_definition=SamplePromptVariables,
        )

        variables = SamplePromptVariables(name="Frank", age=50, city="London")
        result = prompt.compile(variables)

        expected = """Line 1: Frank
Line 2: 50
Line 3: London"""
        assert result == expected

    def test_version_id_lookup_key_error_propagates(self):
        """Test that KeyError is raised when version_id doesn't exist."""
        prompt = BasePrompt(versions={"v1": "Version 1"})

        with pytest.raises(KeyError):
            prompt.compile(version_id="nonexistent")

    def test_missing_variable_in_template_raises_key_error(self):
        """Test that KeyError is raised when template variable is not in model."""
        template = "Hello, {name} and {missing_var}!"
        prompt = BasePrompt(
            versions=template,
            variables_definition=SamplePromptVariables,
        )

        variables = SamplePromptVariables(name="Test", age=25)

        with pytest.raises(KeyError):
            prompt.compile(variables)

    def test_extra_variables_in_model_dont_affect_compile(self):
        """Test that extra variables in the model that aren't in template are ignored."""
        template = "Hello, {name}!"
        prompt = BasePrompt(
            versions=template,
            variables_definition=SamplePromptVariables,
        )

        # Model has age and city, but template only uses name
        variables = SamplePromptVariables(name="Grace", age=60, city="Tokyo")
        result = prompt.compile(variables)

        assert result == "Hello, Grace!"

    def test_numeric_values_in_template(self):
        """Test that numeric values are properly converted to strings."""
        template = "Count: {age}, Double: {age}"
        prompt = BasePrompt(
            versions=template,
            variables_definition=SamplePromptVariables,
        )

        variables = SamplePromptVariables(name="Test", age=42)
        result = prompt.compile(variables)

        assert result == "Count: 42, Double: 42"


class TestPromptTypeAnnotations:
    """Tests related to type annotations and generic behavior."""

    def test_prompt_with_nonetype_generic(self):
        """Test Prompt with NoneType explicitly."""

        prompt: BasePrompt[NoneType] = BasePrompt(versions="No variables here")

        result = prompt.compile()

        assert result == "No variables here"

    def test_prompt_with_custom_type_generic(self):
        """Test Prompt with custom PromptVariables type."""
        prompt: BasePrompt[SamplePromptVariables] = BasePrompt(
            versions="Name: {name}",
            variables_definition=SamplePromptVariables,
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
        prompt = BasePrompt(
            versions=versions,
            variables_definition=AnotherPromptVariables,
            default_version_id="medium",
        )

        variables = AnotherPromptVariables(greeting="Hi", topic="AI")

        short = prompt.compile(variables, version_id="short")
        medium = prompt.compile(variables)  # Uses default
        long = prompt.compile(variables, version_id="long")

        assert short == "Hi!"
        assert medium == "Hi, let's talk about AI."
        assert long == "Hi! Today we'll discuss AI in detail."

    @pytest.mark.asyncio
    async def test_switching_versions_maintains_state(self):
        """Test that switching versions doesn't affect internal state."""
        versions = {"v1": "Version 1", "v2": "Version 2"}
        prompt = BasePrompt(versions=versions, default_version_id="v1")

        result1 = prompt.compile(version_id="v2")
        result2 = prompt.compile()  # Should still use v1 as default

        assert result1 == "Version 2"
        assert result2 == "Version 1"
        assert await prompt.get_default_version_id() == "v1"

    def test_same_prompt_compiled_multiple_times(self):
        """Test that compiling the same prompt multiple times produces consistent results."""
        template = "Hello, {name}!"
        prompt = BasePrompt(
            versions=template,
            variables_definition=SamplePromptVariables,
        )

        variables = SamplePromptVariables(name="Harry", age=45)

        results = [prompt.compile(variables) for _ in range(5)]

        assert all(r == "Hello, Harry!" for r in results)

    def test_different_variable_instances_same_values(self):
        """Test that different variable instances with same values produce same output."""
        prompt = BasePrompt(
            versions="{name} is {age}",
            variables_definition=SamplePromptVariables,
        )

        vars1 = SamplePromptVariables(name="Ivy", age=33)
        vars2 = SamplePromptVariables(name="Ivy", age=33)

        result1 = prompt.compile(vars1)
        result2 = prompt.compile(vars2)

        assert result1 == result2
        assert result1 == "Ivy is 33"


class TestPromptUpdateAndOutdated:
    """Tests for Prompt.update() and OutdatedPrompt behavior."""

    @pytest.mark.asyncio
    async def test_prompt_update_returns_outdated_prompt(self):
        """Test that Prompt.update() returns an OutdatedPrompt."""
        prompt = BasePrompt(versions={"v1": "Original"})
        current, outdated = await prompt._update(versions={"v1": "Updated"})

        assert isinstance(outdated, OutdatedPrompt)
        assert outdated.id == prompt.id
        assert await outdated.get_versions() == {"v1": "Original"}

        assert current == prompt  # Current prompt remains the same instance
        assert await current.get_versions() == {"v1": "Updated"}

    @pytest.mark.asyncio
    async def test_prompt_update_modifies_prompt_in_place(self):
        """Test that Prompt.update() modifies the prompt object in place."""
        prompt = BasePrompt(versions={"v1": "Original"}, default_version_id="v1")

        await prompt._update(
            versions={"v1": "Updated", "v2": "New version"}, default_version_id="v2"
        )

        versions = await prompt.get_versions()
        assert versions["v1"] == "Updated"
        assert versions["v2"] == "New version"
        assert await prompt.get_default_version_id() == "v2"

    @pytest.mark.asyncio
    async def test_prompt_remains_usable_after_update(self):
        """Test that Prompt remains usable after update."""
        prompt = BasePrompt(versions={"v1": "Original {name}"})

        # Compile before update
        result_before = prompt.compile()

        # Update
        await prompt._update(versions={"v1": "Updated {name}"})

        # Compile after update
        result_after = prompt.compile()

        assert result_before == "Original {name}"
        assert result_after == "Updated {name}"

    def test_outdated_prompt_compile_raises_error(self):
        """Test that OutdatedPrompt.compile() raises ValueError."""
        outdated = OutdatedPrompt(
            versions={"v1": "Test"},
            default_version_id="v1",
            id="test_id",
            variables_definition=NoneType,
        )

        with pytest.raises(ValueError, match="This prompt is outdated"):
            outdated.compile()

    @pytest.mark.asyncio
    async def test_outdated_prompt_update_does_nothing(self):
        """Test that OutdatedPrompt.update() returns self without changes."""
        outdated = OutdatedPrompt(
            versions={"v1": "Original"},
            default_version_id="v1",
            id="test_id",
            variables_definition=NoneType,
        )

        with pytest.raises(ValueError):
            await outdated._update(versions={"v1": "Updated"})

    @pytest.mark.asyncio
    async def test_compiled_prompts_become_outdated_on_update(self):
        """Test that compiled prompts reference OutdatedPrompt after prompt update."""
        prompt = BasePrompt(
            versions={"v1": "Version {name}"},
            variables_definition=SamplePromptVariables,
        )
        variables = SamplePromptVariables(name="Test", age=25)

        # Compile a prompt
        compiled_result = prompt.compile(variables)

        # Find the compiled prompt in registry
        compiled_prompt = None
        for cp in _compiled_prompt_registry.values():
            if cp.value == compiled_result:
                compiled_prompt = cp
                break

        assert compiled_prompt is not None
        assert isinstance(compiled_prompt.prompt, BasePrompt)

        # Update the prompt
        await prompt._update(versions={"v1": "Updated {name}"})

        # Check that the same compiled prompt now references OutdatedPrompt
        updated_compiled = None
        for cp in _compiled_prompt_registry.values():
            if cp.value == compiled_result:
                updated_compiled = cp
                break

        assert updated_compiled is not None
        assert isinstance(updated_compiled.prompt, OutdatedPrompt)
        versions = await updated_compiled.prompt.get_versions()
        assert versions["v1"] == "Version {name}"

    @pytest.mark.asyncio
    async def test_outdated_compiled_prompt_cannot_compile(self):
        """Test that trying to compile an outdated compiled prompt raises error."""
        prompt = BasePrompt(
            versions={"v1": "Version {name}"},
            variables_definition=SamplePromptVariables,
        )
        variables = SamplePromptVariables(name="Test", age=25)

        # Compile a prompt
        compiled_result = prompt.compile(variables)

        # Update the prompt, making compiled prompt outdated
        await prompt._update(versions={"v1": "Updated {name}"})

        # Find the outdated compiled prompt
        outdated_cp = None
        for cp in _compiled_prompt_registry.values():
            if cp.value == compiled_result:
                outdated_cp = cp
                break

        assert outdated_cp is not None
        assert isinstance(outdated_cp.prompt, OutdatedPrompt)

        with pytest.raises(ValueError, match="This prompt is outdated"):
            outdated_cp.prompt.compile(variables)


@pytest.mark.asyncio
class TestBasePromptNewMethods:
    """Tests for newly added methods on BasePrompt: append_version and update_default_version_id."""

    @pytest.fixture(autouse=True)
    def clear_compiled_registry(self):
        """Clear the compiled prompt registry before each test."""
        _compiled_prompt_registry.clear()

    async def test_append_version_adds_new_version(self):
        """Test that append_version adds a new version to the prompt."""
        prompt = BasePrompt(versions={"v1": "Version 1"})

        await prompt.append_version(version_id="v2", content="Version 2")

        versions = await prompt.get_versions()
        assert "v2" in versions
        assert versions["v2"] == "Version 2"
        assert versions["v1"] == "Version 1"  # Original version still exists

    async def test_append_version_sets_as_default_when_requested(self):
        """Test that append_version can set the new version as default."""
        prompt = BasePrompt(versions={"v1": "Version 1"}, default_version_id="v1")

        await prompt.append_version(
            version_id="v2", content="Version 2", set_as_default=True
        )

        assert await prompt.get_default_version_id() == "v2"

    async def test_append_version_keeps_existing_default_when_not_requested(self):
        """Test that append_version keeps existing default when set_as_default=False."""
        prompt = BasePrompt(versions={"v1": "Version 1"}, default_version_id="v1")

        await prompt.append_version(
            version_id="v2", content="Version 2", set_as_default=False
        )

        assert await prompt.get_default_version_id() == "v1"

    async def test_append_version_raises_error_for_existing_version_id(self):
        """Test that append_version raises ValueError for existing version ID."""
        prompt = BasePrompt(versions={"v1": "Version 1", "v2": "Version 2"})

        with pytest.raises(ValueError, match="Version ID 'v1' already exists"):
            await prompt.append_version(version_id="v1", content="Duplicate")

    async def test_append_version_creates_outdated_prompt(self):
        """Test that append_version creates an OutdatedPrompt for compiled prompts."""
        prompt = BasePrompt(
            versions={"v1": "Version {name}"},
            variables_definition=SamplePromptVariables,
        )
        variables = SamplePromptVariables(name="Test", age=25)

        # Compile to create a compiled prompt entry
        compiled_result = prompt.compile(variables)

        # Append new version
        await prompt.append_version(version_id="v2", content="New {name}")

        # Check that compiled prompt now references OutdatedPrompt
        compiled_prompt = None
        for cp in _compiled_prompt_registry.values():
            if cp.value == compiled_result:
                compiled_prompt = cp
                break

        assert compiled_prompt is not None
        assert isinstance(compiled_prompt.prompt, OutdatedPrompt)

    async def test_update_default_version_id_changes_default(self):
        """Test that update_default_version_id changes the default version."""
        prompt = BasePrompt(
            versions={"v1": "Version 1", "v2": "Version 2", "v3": "Version 3"},
            default_version_id="v1",
        )

        await prompt.update_default_version_id("v2")

        assert await prompt.get_default_version_id() == "v2"

    async def test_update_default_version_id_raises_error_for_nonexistent_version(self):
        """Test that update_default_version_id raises ValueError for nonexistent version."""
        prompt = BasePrompt(versions={"v1": "Version 1"}, default_version_id="v1")

        with pytest.raises(ValueError, match="Version ID 'nonexistent' does not exist"):
            await prompt.update_default_version_id("nonexistent")

    async def test_update_default_version_id_noop_when_same_version(self):
        """Test that update_default_version_id does nothing when setting to same version."""
        prompt = BasePrompt(
            versions={"v1": "Version 1", "v2": "Version 2"}, default_version_id="v1"
        )

        # This should not raise an error or change anything
        await prompt.update_default_version_id("v1")

        assert await prompt.get_default_version_id() == "v1"

    async def test_update_default_version_id_creates_outdated_prompt(self):
        """Test that update_default_version_id creates OutdatedPrompt for compiled prompts."""
        prompt = BasePrompt(
            versions={"v1": "Version {name}", "v2": "Other {name}"},
            variables_definition=SamplePromptVariables,
            default_version_id="v1",
        )
        variables = SamplePromptVariables(name="Test", age=25)

        # Compile to create a compiled prompt entry
        compiled_result = prompt.compile(variables)

        # Update default version
        await prompt.update_default_version_id("v2")

        # Check that compiled prompt now references OutdatedPrompt
        compiled_prompt = None
        for cp in _compiled_prompt_registry.values():
            if cp.value == compiled_result:
                compiled_prompt = cp
                break

        assert compiled_prompt is not None
        assert isinstance(compiled_prompt.prompt, OutdatedPrompt)

    async def test_append_version_with_variables_compiles_correctly(self):
        """Test that appended versions compile correctly with variables."""
        prompt = BasePrompt(
            versions={"v1": "Hello {name}"}, variables_definition=SamplePromptVariables
        )

        await prompt.append_version(version_id="v2", content="Hi {name}, you are {age}")

        variables = SamplePromptVariables(name="Alice", age=30)

        result_v1 = prompt.compile(variables, version_id="v1")
        result_v2 = prompt.compile(variables, version_id="v2")

        assert result_v1 == "Hello Alice"
        assert result_v2 == "Hi Alice, you are 30"

    async def test_update_default_version_id_affects_compile_without_version_id(self):
        """Test that updating default version affects compile() without version_id parameter."""
        prompt = BasePrompt(
            versions={"v1": "Version 1", "v2": "Version 2"},
            variables_definition=SamplePromptVariables,
            default_version_id="v1",
        )

        variables = SamplePromptVariables(name="Test", age=25)

        # Initially uses v1
        result_before = prompt.compile(variables)
        assert result_before == "Version 1"

        # Update default to v2
        await prompt.update_default_version_id("v2")

        # Now uses v2 as default
        result_after = prompt.compile(variables)
        assert result_after == "Version 2"


class TestUpdatePromptRegistry:
    """Tests for the update_prompt_registry function."""

    @pytest.mark.asyncio
    async def test_update_prompt_registry_new_prompt_raises_error(self):
        """Test that update_prompt_registry raises KeyError for new prompts."""
        new_prompt = BaseUntypedPrompt(versions={"v1": "New version"})

        with pytest.raises(KeyError):
            await BasePrompt.update_prompt_registry(new_prompt)

    @pytest.mark.asyncio
    async def test_update_prompt_registry_existing_prompt(self):
        """Test that update_prompt_registry updates existing prompts."""
        original_prompt = BaseUntypedPrompt(versions={"v1": "Original version"})
        # Manually add to registry with variable definitions
        _prompt_registry[original_prompt.id] = await BasePrompt.from_untyped(
            original_prompt, variables_definition=SamplePromptVariables
        )

        # Temporarily remove from registry to create updated UntypedPrompt
        del _prompt_registry[original_prompt.id]

        updated_prompt = BaseUntypedPrompt(
            id=original_prompt.id, versions={"v1": "Updated version"}
        )

        # Add back to registry to simulate existing
        _prompt_registry[original_prompt.id] = await BasePrompt.from_untyped(
            original_prompt, variables_definition=SamplePromptVariables
        )

        result = await BasePrompt.update_prompt_registry(updated_prompt)

        assert result.id == original_prompt.id
        versions = await result.get_versions()
        assert versions["v1"] == "Updated version"
        assert (
            result.variables_definition == SamplePromptVariables
        )  # Should retain original var_def


@pytest.mark.asyncio
class TestFilePromptStorage:
    """Tests for the FilePromptStorage class."""

    async def test_save_existing_prompt(self):
        """Test saving an existing prompt updates its data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = _FilePromptStorage(directory=temp_dir)

            # Create the prompt in registry first
            prompt = BaseUntypedPrompt(versions={"v1": "Initial version"})
            await BasePrompt.from_untyped(prompt)

            await storage.save(prompt)  # Use await for async call

            # Update the prompt
            updated_prompt = BaseUntypedPrompt(
                id=prompt.id, versions={"v1": "Updated version"}
            )

            is_new = await storage.save(updated_prompt)  # Use await for async call

            assert not is_new

            # Verify the file content
            filepath = os.path.join(temp_dir, f"{prompt.id}.json")
            with open(filepath, "r") as f:
                data = json.load(f)

            assert data["versions"]["v1"] == "Updated version"


class TestVariablesDefinitionToSchema:
    """Tests for variables_definition_to_schema function."""

    def test_nonetype_returns_empty_schema(self):
        """Test that NoneType returns an empty schema."""
        from pixie.prompts.prompt import variables_definition_to_schema

        result = variables_definition_to_schema(NoneType)
        assert result == {"type": "object", "properties": {}}

    def test_prompt_variables_returns_json_schema(self):
        """Test that PromptVariables subclass returns proper JSON schema."""
        from pixie.prompts.prompt import variables_definition_to_schema

        result = variables_definition_to_schema(SamplePromptVariables)

        assert result["type"] == "object"
        assert "properties" in result
        assert "name" in result["properties"]
        assert "age" in result["properties"]
        assert "city" in result["properties"]

        # Check required fields
        assert "required" in result
        assert "name" in result["required"]
        assert "age" in result["required"]
        # city has default, so not required

    def test_different_variable_classes_have_different_schemas(self):
        """Test that different PromptVariables classes produce different schemas."""
        from pixie.prompts.prompt import variables_definition_to_schema

        schema1 = variables_definition_to_schema(SamplePromptVariables)
        schema2 = variables_definition_to_schema(AnotherPromptVariables)

        assert schema1 != schema2
        assert "name" in schema1["properties"]
        assert "greeting" in schema2["properties"]


@pytest.mark.asyncio
class TestBasePromptFromUntyped:
    """Tests for BasePrompt.from_untyped with schema validation."""

    async def test_from_untyped_with_compatible_schema(self):
        """Test that from_untyped works with compatible schema."""
        untyped = BaseUntypedPrompt(
            versions={"v1": "Hello {name}"},
            default_version_id="v1",
            variables_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "city": {"type": "string"},
                },
                "required": ["name", "age"],
            },
        )

        # SamplePromptVariables has name, age, city (with default)
        # This should be compatible
        typed = await BasePrompt.from_untyped(untyped, SamplePromptVariables)

        assert typed.variables_definition == SamplePromptVariables
        assert await typed.get_versions() == {"v1": "Hello {name}"}

    async def test_from_untyped_with_incompatible_schema_raises_error(self):
        """Test that from_untyped raises ValueError for incompatible schema."""
        # Base schema requires 'greeting' and 'topic'
        untyped = BaseUntypedPrompt(
            versions={"v1": "Hello"},
            default_version_id="v1",
            variables_schema={
                "type": "object",
                "properties": {
                    "greeting": {"type": "string"},
                    "topic": {"type": "string"},
                },
                "required": ["greeting", "topic"],
            },
        )

        # SamplePromptVariables has name, age, city - incompatible
        with pytest.raises(TypeError):
            await BasePrompt.from_untyped(untyped, SamplePromptVariables)

    async def test_from_untyped_with_nonetype(self):
        """Test that from_untyped works with NoneType."""
        untyped = BaseUntypedPrompt(
            versions={"v1": "Hello"},
            default_version_id="v1",
        )

        typed = await BasePrompt.from_untyped(untyped, NoneType)

        assert typed.variables_definition == NoneType
        result = typed.compile()
        assert result == "Hello"

    async def test_from_untyped_preserves_id(self):
        """Test that from_untyped preserves the prompt ID."""
        untyped = BaseUntypedPrompt(
            versions={"v1": "Hello"},
            default_version_id="v1",
            id="test_id_123",
        )

        typed = await BasePrompt.from_untyped(untyped, NoneType)

        assert typed.id == "test_id_123"

    async def test_from_untyped_with_empty_schema_accepts_any_variables(self):
        """Test that empty schema accepts any variables definition."""
        untyped = BaseUntypedPrompt(
            versions={"v1": "Hello {name}"},
            default_version_id="v1",
            variables_schema={"type": "object", "properties": {}},
        )

        # Should accept any PromptVariables subclass
        typed1 = await BasePrompt.from_untyped(untyped, SamplePromptVariables)
        typed2 = await BasePrompt.from_untyped(untyped, AnotherPromptVariables)

        assert typed1.variables_definition == SamplePromptVariables
        assert typed2.variables_definition == AnotherPromptVariables


@pytest.mark.asyncio
class TestBaseUntypedPromptWithSchema:
    """Tests for BaseUntypedPrompt with variables_schema parameter."""

    async def test_init_with_variables_schema(self):
        """Test initialization with explicit variables_schema."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        prompt = BaseUntypedPrompt(
            versions={"v1": "Hello"},
            variables_schema=schema,
        )

        result_schema = await prompt.get_variables_schema()
        assert result_schema == schema

    async def test_init_without_variables_schema_uses_empty(self):
        """Test that missing variables_schema defaults to empty schema."""
        prompt = BaseUntypedPrompt(versions={"v1": "Hello"})

        result_schema = await prompt.get_variables_schema()
        assert result_schema == {"type": "object", "properties": {}}

    async def test_get_variables_schema_returns_copy(self):
        """Test that get_variables_schema returns a deep copy."""
        original_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }

        prompt = BaseUntypedPrompt(
            versions={"v1": "Hello"},
            variables_schema=original_schema,
        )

        result_schema = await prompt.get_variables_schema()
        result_schema["properties"]["name"]["type"] = "integer"

        # Original should be unchanged
        check_schema = await prompt.get_variables_schema()
        assert check_schema["properties"]["name"]["type"] == "string"


@pytest.mark.asyncio
class TestOutdatedPromptGetMethods:
    """Tests for OutdatedPrompt async get methods."""

    async def test_outdated_prompt_get_versions(self):
        """Test that OutdatedPrompt.get_versions works."""
        outdated = OutdatedPrompt(
            versions={"v1": "Test", "v2": "Test2"},
            default_version_id="v1",
            id="test_id",
            variables_definition=NoneType,
        )

        versions = await outdated.get_versions()
        assert versions == {"v1": "Test", "v2": "Test2"}

    async def test_outdated_prompt_get_default_version_id(self):
        """Test that OutdatedPrompt.get_default_version_id works."""
        outdated = OutdatedPrompt(
            versions={"v1": "Test"},
            default_version_id="v1",
            id="test_id",
            variables_definition=NoneType,
        )

        default_id = await outdated.get_default_version_id()
        assert default_id == "v1"

    async def test_outdated_prompt_preserves_variables_definition(self):
        """Test that OutdatedPrompt preserves variables_definition."""
        prompt = BasePrompt(
            versions={"v1": "Hello {name}"},
            variables_definition=SamplePromptVariables,
        )

        outdated = await OutdatedPrompt.from_prompt(prompt)

        assert outdated.variables_definition == SamplePromptVariables


class TestGetCompiledPrompt:
    """Tests for get_compiled_prompt function."""

    def setup_method(self):
        """Clear registries before each test."""
        _compiled_prompt_registry.clear()
        _prompt_registry.clear()

    def teardown_method(self):
        """Clear registries after each test."""
        _compiled_prompt_registry.clear()
        _prompt_registry.clear()

    def test_get_compiled_prompt_empty_registry(self):
        """Test get_compiled_prompt returns None when registry is empty."""
        from pixie.prompts.prompt import get_compiled_prompt

        result = get_compiled_prompt("any text")
        assert result is None

    def test_get_compiled_prompt_direct_match(self):
        """Test get_compiled_prompt finds prompt by direct id match."""
        from pixie.prompts.prompt import get_compiled_prompt

        prompt = BasePrompt(versions="Test prompt")
        compiled_text = prompt.compile()

        # The compiled text should be in the registry with its id as key
        result = get_compiled_prompt(compiled_text)
        assert result is not None
        assert result.value == "Test prompt"
        assert result.prompt == prompt
        assert result.version_id == DEFAULT_VERSION_ID

    def test_get_compiled_prompt_value_match(self):
        """Test get_compiled_prompt finds prompt by value when direct match fails."""
        from pixie.prompts.prompt import get_compiled_prompt

        prompt = BasePrompt(versions="Test prompt")
        prompt.compile()

        # Create a new string with same content but different id
        same_text = "Test prompt"

        result = get_compiled_prompt(same_text)
        assert result is not None
        assert result.value == "Test prompt"
        assert result.prompt == prompt

    def test_get_compiled_prompt_with_variables(self):
        """Test get_compiled_prompt works with variable-substituted prompts."""
        from pixie.prompts.prompt import get_compiled_prompt

        prompt = BasePrompt(
            versions="Hello, {name}!", variables_definition=SamplePromptVariables
        )
        variables = SamplePromptVariables(name="Alice", age=25)
        compiled_text = prompt.compile(variables)

        result = get_compiled_prompt(compiled_text)
        assert result is not None
        assert result.value == "Hello, Alice!"
        assert result.prompt == prompt
        assert result.variables == variables

    def test_get_compiled_prompt_json_string(self):
        """Test get_compiled_prompt can find prompts in JSON strings."""
        from pixie.prompts.prompt import get_compiled_prompt

        prompt = BasePrompt(versions="Test prompt")
        compiled_text = prompt.compile()

        # Embed the compiled text in JSON
        json_text = f'{{"message": "{compiled_text}", "other": "data"}}'

        result = get_compiled_prompt(json_text)
        assert result is not None
        assert result.value == "Test prompt"
        assert result.prompt == prompt

    def test_get_compiled_prompt_json_array(self):
        """Test get_compiled_prompt can find prompts in JSON arrays."""
        from pixie.prompts.prompt import get_compiled_prompt

        prompt = BasePrompt(versions="Test prompt")
        compiled_text = prompt.compile()

        # Embed the compiled text in JSON array
        json_text = f'["start", "{compiled_text}", "end"]'

        result = get_compiled_prompt(json_text)
        assert result is not None
        assert result.value == "Test prompt"
        assert result.prompt == prompt

    def test_get_compiled_prompt_nested_json(self):
        """Test get_compiled_prompt can find prompts in nested JSON structures."""
        from pixie.prompts.prompt import get_compiled_prompt

        prompt = BasePrompt(versions="Test prompt")
        compiled_text = prompt.compile()

        # Embed in nested structure
        json_text = f'{{"data": {{"messages": ["{compiled_text}"]}}}}'

        result = get_compiled_prompt(json_text)
        assert result is not None
        assert result.value == "Test prompt"
        assert result.prompt == prompt

    def test_get_compiled_prompt_no_match(self):
        """Test get_compiled_prompt returns None when no match is found."""
        from pixie.prompts.prompt import get_compiled_prompt

        prompt = BasePrompt(versions="Test prompt")
        prompt.compile()  # Add to registry

        result = get_compiled_prompt("Completely different text")
        assert result is None

    def test_get_compiled_prompt_invalid_json(self):
        """Test get_compiled_prompt handles invalid JSON gracefully."""
        from pixie.prompts.prompt import get_compiled_prompt

        prompt = BasePrompt(versions="Test prompt")
        prompt.compile()  # Add to registry

        # Invalid JSON should not crash
        result = get_compiled_prompt("{invalid json")
        assert result is None

    def test_get_compiled_prompt_empty_string(self):
        """Test get_compiled_prompt with empty string."""
        from pixie.prompts.prompt import get_compiled_prompt

        result = get_compiled_prompt("")
        assert result is None

    def test_get_compiled_prompt_multiple_prompts(self):
        """Test get_compiled_prompt with multiple prompts in registry."""
        from pixie.prompts.prompt import get_compiled_prompt

        prompt1 = BasePrompt(versions="First prompt")
        prompt2 = BasePrompt(versions="Second prompt")

        compiled1 = prompt1.compile()
        compiled2 = prompt2.compile()

        result1 = get_compiled_prompt(compiled1)
        result2 = get_compiled_prompt(compiled2)

        assert result1 is not None
        assert result1.prompt == prompt1
        assert result2 is not None
        assert result2.prompt == prompt2

    @pytest.mark.asyncio
    async def test_get_compiled_prompt_outdated_prompt(self):
        """Test get_compiled_prompt handles outdated prompts correctly."""
        from pixie.prompts.prompt import get_compiled_prompt

        prompt = BasePrompt(versions={"v1": "Original"})
        compiled_text = prompt.compile()

        # Update the prompt to create an outdated version
        await prompt._update(versions={"v1": "Updated"})

        result = get_compiled_prompt(compiled_text)
        assert result is not None
        assert isinstance(result.prompt, OutdatedPrompt)
        assert result.value == "Original"
