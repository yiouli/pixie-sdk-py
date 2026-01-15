"""Comprehensive unit tests for pixie.prompts.storage module."""

import json
import os
import tempfile
import pytest
from types import NoneType
from typing import Dict

from pixie.prompts.prompt import BaseUntypedPrompt, _prompt_registry
from pixie.prompts.storage import _FilePromptStorage


class TestFilePromptStorage:
    """Tests for FilePromptStorage class."""

    @pytest.fixture(autouse=True)
    def clear_prompt_registry(self):
        """Clear the global prompt registry before each test."""
        _prompt_registry.clear()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_prompt_data(self) -> Dict[str, Dict]:
        """Sample prompt data for testing."""
        return {
            "prompt1": {
                "versions": {"v1": "Hello {name}", "v2": "Hi {name}"},
                "defaultVersionId": "v1",
                "variablesSchema": {"type": "object", "properties": {}},
            },
            "prompt2": {
                "versions": {"default": "Goodbye {name}"},
                "defaultVersionId": "default",
                "variablesSchema": {"type": "object", "properties": {}},
            },
        }

    def create_sample_files(self, temp_dir: str, sample_data: Dict[str, Dict]):
        """Create sample JSON files in the temp directory."""
        for prompt_id, data in sample_data.items():
            filepath = os.path.join(temp_dir, f"{prompt_id}.json")
            with open(filepath, "w") as f:
                json.dump(data, f)

    def test_init_creates_directory_if_not_exists(self, temp_dir: str):
        """Test that __init__ creates the directory if it doesn't exist."""
        subdir = os.path.join(temp_dir, "storage")
        assert not os.path.exists(subdir)

        storage = _FilePromptStorage(subdir)
        assert os.path.exists(subdir)
        assert isinstance(storage._prompts, dict)
        assert len(storage._prompts) == 0

    @pytest.mark.asyncio
    async def test_init_loads_existing_files(
        self, temp_dir: str, sample_prompt_data: Dict[str, Dict]
    ):
        """Test that __init__ loads existing JSON files into memory."""
        self.create_sample_files(temp_dir, sample_prompt_data)

        storage = _FilePromptStorage(temp_dir)

        assert len(storage._prompts) == 2
        assert "prompt1" in storage._prompts
        assert "prompt2" in storage._prompts

        prompt1 = storage._prompts["prompt1"]
        assert isinstance(prompt1, BaseUntypedPrompt)
        assert prompt1.id == "prompt1"
        assert await prompt1.get_versions() == sample_prompt_data["prompt1"]["versions"]
        assert (
            await prompt1.get_default_version_id()
            == sample_prompt_data["prompt1"]["defaultVersionId"]
        )

        prompt2 = storage._prompts["prompt2"]
        assert prompt2.id == "prompt2"
        assert await prompt2.get_versions() == sample_prompt_data["prompt2"]["versions"]
        assert (
            await prompt2.get_default_version_id()
            == sample_prompt_data["prompt2"]["defaultVersionId"]
        )

    def test_init_handles_empty_directory(self, temp_dir: str):
        """Test that __init__ handles an empty directory gracefully."""
        storage = _FilePromptStorage(temp_dir)
        assert len(storage._prompts) == 0

    def test_init_skips_non_json_files(self, temp_dir: str):
        """Test that __init__ skips files that don't end with .json."""
        # Create a JSON file and a non-JSON file
        json_path = os.path.join(temp_dir, "prompt1.json")
        with open(json_path, "w") as f:
            json.dump(
                {
                    "versions": {"default": "test"},
                    "defaultVersionId": "default",
                    "variablesSchema": {"type": "object", "properties": {}},
                },
                f,
            )

        txt_path = os.path.join(temp_dir, "readme.txt")
        with open(txt_path, "w") as f:
            f.write("This is not JSON")

        storage = _FilePromptStorage(temp_dir)
        assert len(storage._prompts) == 1
        assert "prompt1" in storage._prompts

    def test_init_handles_invalid_json(self, temp_dir: str):
        """Test that __init__ raises an exception for invalid JSON."""
        invalid_path = os.path.join(temp_dir, "invalid.json")
        with open(invalid_path, "w") as f:
            f.write("invalid json content")

        with pytest.raises(json.JSONDecodeError):
            _FilePromptStorage(temp_dir)

    def test_init_handles_missing_versions(self, temp_dir: str):
        """Test that __init__ raises ValueError for missing versions in JSON."""
        missing_versions_path = os.path.join(temp_dir, "missing.json")
        with open(missing_versions_path, "w") as f:
            json.dump({"defaultVersionId": "default"}, f)

        with pytest.raises(KeyError):
            _FilePromptStorage(temp_dir)

    @pytest.mark.asyncio
    async def test_exists_returns_true_for_existing_prompt(
        self, temp_dir: str, sample_prompt_data: Dict[str, Dict]
    ):
        """Test that exists returns True for existing prompts."""
        self.create_sample_files(temp_dir, sample_prompt_data)
        storage = _FilePromptStorage(temp_dir)

        assert await storage.exists("prompt1") is True
        assert await storage.exists("prompt2") is True

    @pytest.mark.asyncio
    async def test_exists_returns_false_for_non_existing_prompt(self, temp_dir: str):
        """Test that exists returns False for non-existing prompts."""
        storage = _FilePromptStorage(temp_dir)

        assert await storage.exists("nonexistent") is False

    @pytest.mark.asyncio
    async def test_save_creates_new_prompt(self, temp_dir: str):
        """Test that save creates a new prompt and returns True."""
        storage = _FilePromptStorage(temp_dir)

        prompt = BaseUntypedPrompt(
            versions={"v1": "Hello {name}", "v2": "Hi {name}"},
            default_version_id="v1",
            id="new_prompt",
        )

        # Save should work for new prompts now and return True
        result = await storage.save(prompt)
        assert result is True

        # Verify file was created
        filepath = os.path.join(temp_dir, "new_prompt.json")
        assert os.path.exists(filepath)

        # Verify content
        with open(filepath, "r") as f:
            data = json.load(f)
        assert data["versions"] == {"v1": "Hello {name}", "v2": "Hi {name}"}
        assert data["defaultVersionId"] == "v1"
        assert "variablesSchema" in data

    @pytest.mark.asyncio
    async def test_save_updates_existing_prompt(
        self, temp_dir: str, sample_prompt_data: Dict[str, Dict]
    ):
        """Test that save updates an existing prompt and returns False."""
        self.create_sample_files(temp_dir, sample_prompt_data)
        storage = _FilePromptStorage(temp_dir)

        # Register the loaded prompts
        from pixie.prompts.prompt import BasePrompt

        for p in storage._prompts.values():
            await BasePrompt.from_untyped(p)

        # Modify the existing prompt
        storage._prompts["prompt1"]
        updated_versions = {"v1": "Updated {name}", "v3": "New version"}
        updated_prompt = BaseUntypedPrompt(
            versions=updated_versions, default_version_id="v1", id="prompt1"
        )

        result = await storage.save(updated_prompt)
        assert result is False  # Should return False for existing prompt

        # Check in-memory was updated
        assert storage._prompts["prompt1"] is updated_prompt
        assert await storage._prompts["prompt1"].get_versions() == updated_versions

        # Check file was updated
        filepath = os.path.join(temp_dir, "prompt1.json")
        with open(filepath, "r") as f:
            data = json.load(f)
        assert data["versions"] == updated_versions
        assert data["defaultVersionId"] == "v1"

    @pytest.mark.asyncio
    async def test_get_returns_existing_prompt(
        self, temp_dir: str, sample_prompt_data: Dict[str, Dict]
    ):
        """Test that get returns the correct prompt for existing ID."""
        self.create_sample_files(temp_dir, sample_prompt_data)
        storage = _FilePromptStorage(temp_dir)

        prompt = await storage.get("prompt1")
        assert isinstance(prompt, BaseUntypedPrompt)
        assert prompt.id == "prompt1"
        assert await prompt.get_versions() == sample_prompt_data["prompt1"]["versions"]
        assert (
            await prompt.get_default_version_id()
            == sample_prompt_data["prompt1"]["defaultVersionId"]
        )

    @pytest.mark.asyncio
    async def test_get_raises_keyerror_for_non_existing_prompt(self, temp_dir: str):
        """Test that get raises KeyError for non-existing prompt ID."""
        storage = _FilePromptStorage(temp_dir)

        with pytest.raises(KeyError):
            await storage.get("nonexistent")

    @pytest.mark.asyncio
    async def test_save_writes_to_file_before_memory_update(self, temp_dir: str):
        """Test that save creates a new prompt successfully."""
        storage = _FilePromptStorage(temp_dir)

        prompt = BaseUntypedPrompt(
            versions={"default": "Test"}, default_version_id="default", id="test_prompt"
        )

        result = await storage.save(prompt)
        assert result is True

        # Verify it was saved to file
        filepath = os.path.join(temp_dir, "test_prompt.json")
        assert os.path.exists(filepath)

    @pytest.mark.asyncio
    async def test_init_with_default_version_id_none(self, temp_dir: str):
        """Test loading a prompt where defaultVersionId is missing (defaults to first version)."""
        filepath = os.path.join(temp_dir, "prompt.json")
        with open(filepath, "w") as f:
            json.dump(
                {
                    "versions": {"v1": "Version 1"},
                    "defaultVersionId": "v1",
                    "variablesSchema": {"type": "object", "properties": {}},
                },
                f,
            )

        storage = _FilePromptStorage(temp_dir)
        prompt = storage._prompts["prompt"]
        assert (
            await prompt.get_default_version_id() == "v1"
        )  # Defaults to first version

    @pytest.mark.asyncio
    async def test_save_validates_schema_compatibility(self, temp_dir: str):
        """Test that save validates schema compatibility when updating prompts."""
        # Create initial prompt with schema
        initial_prompt = BaseUntypedPrompt(
            versions={"v1": "Hello"},
            default_version_id="v1",
            id="schema_test",
            variables_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )

        storage = _FilePromptStorage(temp_dir)
        await storage.save(initial_prompt)

        # Try to update with incompatible schema (removing required field)
        updated_prompt = BaseUntypedPrompt(
            versions={"v1": "Hello"},
            default_version_id="v1",
            id="schema_test",
            variables_schema={
                "type": "object",
                "properties": {"age": {"type": "integer"}},
                "required": ["age"],
            },
        )

        # Should raise TypeError due to incompatible schema
        with pytest.raises(TypeError):
            await storage.save(updated_prompt)

    @pytest.mark.asyncio
    async def test_save_allows_compatible_schema_extension(self, temp_dir: str):
        """Test that save allows extending schema with compatible changes."""
        # Create initial prompt with broader schema
        initial_prompt = BaseUntypedPrompt(
            versions={"v1": "Hello"},
            default_version_id="v1",
            id="schema_test",
            variables_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
            },
        )

        storage = _FilePromptStorage(temp_dir)
        await storage.save(initial_prompt)

        # Update with narrower but compatible schema (fewer fields)
        # Original schema is a subschema of new schema if new schema is more permissive
        updated_prompt = BaseUntypedPrompt(
            versions={"v1": "Hello"},
            default_version_id="v1",
            id="schema_test",
            variables_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        )

        # Should succeed - removing optional fields makes schema more permissive
        result = await storage.save(updated_prompt)
        assert result is False  # Existing prompt


@pytest.mark.asyncio
class TestStorageBackedPrompt:
    """Tests for StorageBackedPrompt class."""

    @pytest.fixture(autouse=True)
    def clear_prompt_registry(self):
        """Clear the global prompt registry before each test."""
        _prompt_registry.clear()

    @pytest.fixture(autouse=True)
    def reset_storage_instance(self):
        """Reset the global storage instance before each test."""
        import pixie.prompts.storage as storage_module

        storage_module._storage_instance = None
        yield
        storage_module._storage_instance = None

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    async def test_storage_backed_prompt_lazy_loading(self, temp_dir: str):
        """Test that StorageBackedPrompt loads from storage on first access."""
        from pixie.prompts.storage import (
            initialize_prompt_storage,
            StorageBackedPrompt,
        )

        # Create prompt file directly
        import json
        import os

        prompt_file = os.path.join(temp_dir, "test_prompt.json")
        with open(prompt_file, "w") as f:
            json.dump(
                {
                    "versions": {"v1": "Hello {name}"},
                    "defaultVersionId": "v1",
                    "variablesSchema": {"type": "object", "properties": {}},
                },
                f,
            )

        # Initialize storage - it will load existing files
        initialize_prompt_storage(temp_dir)

        # Create StorageBackedPrompt - should not load yet
        backed_prompt = StorageBackedPrompt(id="test_prompt")
        assert backed_prompt._prompt is None

        # Access versions - should trigger loading
        versions = await backed_prompt.get_versions()
        assert versions == {"v1": "Hello {name}"}
        assert backed_prompt._prompt is not None

    async def test_storage_backed_prompt_compile(self, temp_dir: str):
        """Test that StorageBackedPrompt.compile works correctly."""
        from pixie.prompts.storage import (
            initialize_prompt_storage,
            StorageBackedPrompt,
        )
        from pixie.prompts.prompt import PromptVariables

        class TestVars(PromptVariables):
            name: str

        # Create prompt file directly
        import json
        import os

        prompt_file = os.path.join(temp_dir, "test_prompt.json")
        with open(prompt_file, "w") as f:
            json.dump(
                {
                    "versions": {"v1": "Hello {name}!"},
                    "defaultVersionId": "v1",
                    "variablesSchema": {"type": "object", "properties": {}},
                },
                f,
            )

        # Initialize storage
        initialize_prompt_storage(temp_dir)

        # Create StorageBackedPrompt with variable definition
        backed_prompt = StorageBackedPrompt(
            id="test_prompt", variables_definition=TestVars
        )

        # Compile
        variables = TestVars(name="World")
        result = await backed_prompt.compile(variables)
        assert result == "Hello World!"

    async def test_storage_backed_prompt_raises_without_init(self):
        """Test that StorageBackedPrompt raises error if storage not initialized."""
        from pixie.prompts.storage import StorageBackedPrompt
        import pixie.prompts.storage as storage_module

        # Ensure storage is not initialized
        storage_module._storage_instance = None

        backed_prompt = StorageBackedPrompt(id="test_prompt")

        with pytest.raises(
            RuntimeError, match="Prompt storage has not been initialized"
        ):
            await backed_prompt.get_versions()

    async def test_create_prompt_helper(self, temp_dir: str):
        """Test the create_prompt helper function."""
        from pixie.prompts.storage import initialize_prompt_storage, create_prompt

        # Create prompt file directly
        import json
        import os

        prompt_file = os.path.join(temp_dir, "helper_test.json")
        with open(prompt_file, "w") as f:
            json.dump(
                {
                    "versions": {"v1": "Test"},
                    "defaultVersionId": "v1",
                    "variablesSchema": {"type": "object", "properties": {}},
                },
                f,
            )

        # Initialize storage
        initialize_prompt_storage(temp_dir)

        # Create prompt using helper
        prompt = create_prompt(id="helper_test")
        assert prompt.id == "helper_test"

        versions = await prompt.get_versions()
        assert versions == {"v1": "Test"}

    async def test_storage_backed_prompt_schema_compatibility_check_passes(
        self, temp_dir: str
    ):
        """Test that schema compatibility check passes when definition is subschema of storage."""
        from pixie.prompts.storage import (
            initialize_prompt_storage,
            StorageBackedPrompt,
        )
        from pixie.prompts.prompt import PromptVariables

        class TestVars(PromptVariables):
            name: str

        # Create prompt file with empty schema (accepts everything)
        import json
        import os

        prompt_file = os.path.join(temp_dir, "schema_test.json")
        with open(prompt_file, "w") as f:
            json.dump(
                {
                    "versions": {"v1": "Hello {name}!"},
                    "defaultVersionId": "v1",
                    "variablesSchema": {
                        "type": "object",
                        "properties": {},
                    },  # Empty schema
                },
                f,
            )

        # Initialize storage
        initialize_prompt_storage(temp_dir)

        # Create StorageBackedPrompt with restrictive definition
        backed_prompt = StorageBackedPrompt(
            id="schema_test", variables_definition=TestVars
        )

        # Should not raise, since TestVars schema is subschema of empty
        versions = await backed_prompt.get_versions()
        assert versions == {"v1": "Hello {name}!"}

    async def test_storage_backed_prompt_schema_compatibility_check_fails(
        self, temp_dir: str
    ):
        """Test that schema compatibility check fails when definition is not subschema of storage."""
        from pixie.prompts.storage import (
            initialize_prompt_storage,
            StorageBackedPrompt,
        )

        # Create prompt file with restrictive schema (requires name)
        import json
        import os

        prompt_file = os.path.join(temp_dir, "schema_fail_test.json")
        with open(prompt_file, "w") as f:
            json.dump(
                {
                    "versions": {"v1": "Hello {name}!"},
                    "defaultVersionId": "v1",
                    "variablesSchema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                },
                f,
            )

        # Initialize storage
        initialize_prompt_storage(temp_dir)

        # Create StorageBackedPrompt with NoneType (empty schema)
        backed_prompt = StorageBackedPrompt(
            id="schema_fail_test", variables_definition=NoneType
        )

        # Should raise TypeError because empty schema is not subschema of required schema
        with pytest.raises(
            TypeError,
            match="The provided variables_definition is not compatible with the prompt's variables schema",
        ):
            await backed_prompt.get_versions()


class TestInitializePromptStorage:
    """Tests for initialize_prompt_storage function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture(autouse=True)
    def reset_storage_instance(self):
        """Reset the global storage instance before each test."""
        import pixie.prompts.storage as storage_module

        storage_module._storage_instance = None
        yield
        storage_module._storage_instance = None

    def test_initialize_prompt_storage_once(self, temp_dir: str):
        """Test that initialize_prompt_storage can only be called once."""
        from pixie.prompts.storage import initialize_prompt_storage

        initialize_prompt_storage(temp_dir)

        # Should raise error on second call
        with pytest.raises(
            RuntimeError, match="Prompt storage has already been initialized"
        ):
            initialize_prompt_storage(temp_dir)

    def test_initialize_creates_storage(self, temp_dir: str):
        """Test that initialize_prompt_storage creates a FilePromptStorage instance."""
        from pixie.prompts.storage import initialize_prompt_storage
        import pixie.prompts.storage as storage_module

        initialize_prompt_storage(temp_dir)

        assert storage_module._storage_instance is not None
        assert isinstance(storage_module._storage_instance, _FilePromptStorage)
