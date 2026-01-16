import pytest
from types import NoneType

from pixie.prompts.prompt_management import (
    create_prompt,
    list_prompts,
    get_prompt,
    _registry,
)
from pixie.prompts.prompt import PromptVariables


class DummyVar1(PromptVariables):
    pass


class DummyVar2(PromptVariables):
    pass


class TestPromptManagement:
    def setup_method(self):
        """Clear the registry before each test."""
        _registry.clear()

    def test_create_prompt_without_description(self):
        """Test creating a prompt without description."""
        prompt = create_prompt("test_prompt")

        assert prompt.id == "test_prompt"
        assert prompt.variables_definition == NoneType

        # Check registry
        registered = get_prompt("test_prompt")
        assert registered is not None
        assert registered.description is None
        assert registered.module == "tests.pixie.test_prompt_management"
        assert registered.prompt is prompt

    def test_create_prompt_with_description(self):
        """Test creating a prompt with description."""
        description = "A test prompt description"
        prompt = create_prompt("test_prompt_desc", description=description)

        assert prompt.id == "test_prompt_desc"

        # Check registry
        registered = get_prompt("test_prompt_desc")
        assert registered is not None
        assert registered.description == description
        assert registered.module == "tests.pixie.test_prompt_management"
        assert registered.prompt is prompt

    def test_create_prompt_same_id_same_vars(self):
        """Test creating a prompt with same id and same variables_definition returns existing."""
        prompt1 = create_prompt("duplicate_prompt")
        prompt2 = create_prompt("duplicate_prompt")

        assert prompt1 is prompt2

        # Check registry has only one entry
        registered = get_prompt("duplicate_prompt")
        assert registered is not None
        assert registered.prompt is prompt1

    def test_create_prompt_same_id_different_vars_raises_error(self):
        """Test creating a prompt with same id but different variables_definition raises ValueError."""
        # First create with DummyVar1
        create_prompt("conflict_prompt", variables_definition=DummyVar1)

        # Try to create with DummyVar2
        with pytest.raises(
            ValueError,
            match="Prompt with id 'conflict_prompt' already exists with a different variables definition",
        ):
            create_prompt("conflict_prompt", variables_definition=DummyVar2)

    def test_list_prompts(self):
        """Test listing all prompts."""
        # Initially empty
        assert list_prompts() == []

        # Add some prompts
        create_prompt("prompt1", description="First prompt")
        create_prompt("prompt2", description="Second prompt")

        prompts = list_prompts()
        assert len(prompts) == 2

        # Check contents
        ids = {p.prompt.id for p in prompts}
        assert ids == {"prompt1", "prompt2"}

        descriptions = {p.description for p in prompts}
        assert descriptions == {"First prompt", "Second prompt"}

    def test_get_prompt_existing(self):
        """Test getting an existing prompt."""
        prompt = create_prompt("existing_prompt", description="Exists")
        registered = get_prompt("existing_prompt")

        assert registered is not None
        assert registered.prompt is prompt
        assert registered.description == "Exists"

    def test_get_prompt_nonexistent(self):
        """Test getting a nonexistent prompt returns None."""
        assert get_prompt("nonexistent") is None

    def test_module_registration(self):
        """Test that module is correctly registered as the calling module."""
        create_prompt("module_test")
        registered = get_prompt("module_test")

        # In this test context, it should be the test module
        assert registered is not None
        assert registered.module == "tests.pixie.test_prompt_management"
