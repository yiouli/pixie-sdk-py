"""Comprehensive unit tests for pixie.prompts.storage module."""

import json
import os
import tempfile
import pytest
from typing import Dict

from pixie.prompts.prompt import UntypedPrompt, _prompt_registry
from pixie.prompts.storage import FilePromptStorage


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
            },
            "prompt2": {
                "versions": {"default": "Goodbye {name}"},
                "defaultVersionId": "default",
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

        storage = FilePromptStorage(subdir)
        assert os.path.exists(subdir)
        assert isinstance(storage._prompts, dict)
        assert len(storage._prompts) == 0

    def test_init_loads_existing_files(
        self, temp_dir: str, sample_prompt_data: Dict[str, Dict]
    ):
        """Test that __init__ loads existing JSON files into memory."""
        self.create_sample_files(temp_dir, sample_prompt_data)

        storage = FilePromptStorage(temp_dir)

        assert len(storage._prompts) == 2
        assert "prompt1" in storage._prompts
        assert "prompt2" in storage._prompts

        prompt1 = storage._prompts["prompt1"]
        assert isinstance(prompt1, UntypedPrompt)
        assert prompt1.id == "prompt1"
        assert prompt1.versions == sample_prompt_data["prompt1"]["versions"]
        assert (
            prompt1.default_version_id
            == sample_prompt_data["prompt1"]["defaultVersionId"]
        )

        prompt2 = storage._prompts["prompt2"]
        assert prompt2.id == "prompt2"
        assert prompt2.versions == sample_prompt_data["prompt2"]["versions"]
        assert (
            prompt2.default_version_id
            == sample_prompt_data["prompt2"]["defaultVersionId"]
        )

    def test_init_handles_empty_directory(self, temp_dir: str):
        """Test that __init__ handles an empty directory gracefully."""
        storage = FilePromptStorage(temp_dir)
        assert len(storage._prompts) == 0

    def test_init_skips_non_json_files(self, temp_dir: str):
        """Test that __init__ skips files that don't end with .json."""
        # Create a JSON file and a non-JSON file
        json_path = os.path.join(temp_dir, "prompt1.json")
        with open(json_path, "w") as f:
            json.dump(
                {"versions": {"default": "test"}, "defaultVersionId": "default"}, f
            )

        txt_path = os.path.join(temp_dir, "readme.txt")
        with open(txt_path, "w") as f:
            f.write("This is not JSON")

        storage = FilePromptStorage(temp_dir)
        assert len(storage._prompts) == 1
        assert "prompt1" in storage._prompts

    def test_init_handles_invalid_json(self, temp_dir: str):
        """Test that __init__ raises an exception for invalid JSON."""
        invalid_path = os.path.join(temp_dir, "invalid.json")
        with open(invalid_path, "w") as f:
            f.write("invalid json content")

        with pytest.raises(json.JSONDecodeError):
            FilePromptStorage(temp_dir)

    def test_init_handles_missing_versions(self, temp_dir: str):
        """Test that __init__ raises ValueError for missing versions in JSON."""
        missing_versions_path = os.path.join(temp_dir, "missing.json")
        with open(missing_versions_path, "w") as f:
            json.dump({"defaultVersionId": "default"}, f)

        with pytest.raises(KeyError):
            FilePromptStorage(temp_dir)

    @pytest.mark.asyncio
    async def test_exists_returns_true_for_existing_prompt(
        self, temp_dir: str, sample_prompt_data: Dict[str, Dict]
    ):
        """Test that exists returns True for existing prompts."""
        self.create_sample_files(temp_dir, sample_prompt_data)
        storage = FilePromptStorage(temp_dir)

        assert await storage.exists("prompt1") is True
        assert await storage.exists("prompt2") is True

    @pytest.mark.asyncio
    async def test_exists_returns_false_for_non_existing_prompt(self, temp_dir: str):
        """Test that exists returns False for non-existing prompts."""
        storage = FilePromptStorage(temp_dir)

        assert await storage.exists("nonexistent") is False

    @pytest.mark.asyncio
    async def test_save_creates_new_prompt(self, temp_dir: str):
        """Test that save creates a new prompt and returns True."""
        storage = FilePromptStorage(temp_dir)

        prompt = UntypedPrompt(
            versions={"v1": "Hello {name}", "v2": "Hi {name}"},
            default_version_id="v1",
            id="new_prompt",
        )

        result = await storage.save(prompt)
        assert result is True

        # Check in-memory
        assert "new_prompt" in storage._prompts
        assert storage._prompts["new_prompt"] is prompt

        # Check file was written
        filepath = os.path.join(temp_dir, "new_prompt.json")
        assert os.path.exists(filepath)
        with open(filepath, "r") as f:
            data = json.load(f)
        assert data["versions"] == prompt.versions
        assert data["defaultVersionId"] == prompt.default_version_id

    @pytest.mark.asyncio
    async def test_save_updates_existing_prompt(
        self, temp_dir: str, sample_prompt_data: Dict[str, Dict]
    ):
        """Test that save updates an existing prompt and returns False."""
        self.create_sample_files(temp_dir, sample_prompt_data)
        storage = FilePromptStorage(temp_dir)

        # Modify the existing prompt
        storage._prompts["prompt1"]
        updated_versions = {"v1": "Updated {name}", "v3": "New version"}
        # Remove from registry to allow creating new prompt with same id
        _prompt_registry.pop("prompt1", None)
        updated_prompt = UntypedPrompt(
            versions=updated_versions, default_version_id="v1", id="prompt1"
        )

        result = await storage.save(updated_prompt)
        assert result is False

        # Check in-memory was updated
        assert storage._prompts["prompt1"] is updated_prompt
        assert storage._prompts["prompt1"].versions == updated_versions

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
        storage = FilePromptStorage(temp_dir)

        prompt = await storage.get("prompt1")
        assert isinstance(prompt, UntypedPrompt)
        assert prompt.id == "prompt1"
        assert prompt.versions == sample_prompt_data["prompt1"]["versions"]
        assert (
            prompt.default_version_id
            == sample_prompt_data["prompt1"]["defaultVersionId"]
        )

    @pytest.mark.asyncio
    async def test_get_raises_keyerror_for_non_existing_prompt(self, temp_dir: str):
        """Test that get raises KeyError for non-existing prompt ID."""
        storage = FilePromptStorage(temp_dir)

        with pytest.raises(KeyError):
            await storage.get("nonexistent")

    @pytest.mark.asyncio
    async def test_save_writes_to_file_before_memory_update(self, temp_dir: str):
        """Test that save writes to file before updating memory (for crash safety)."""
        storage = FilePromptStorage(temp_dir)

        prompt = UntypedPrompt(
            versions={"default": "Test"}, default_version_id="default", id="test_prompt"
        )

        # Mock file write to fail after writing
        original_open = open
        write_count = 0

        def mock_open(*args, **kwargs):
            nonlocal write_count
            if args and "test_prompt.json" in args[0] and "w" in args[1]:
                write_count += 1
                # Simulate writing to file
                result = original_open(*args, **kwargs)
                result.write(
                    '{"versions": {"default": "Test"}, "defaultVersionId": "default"}'
                )
                result.close()
                # Then raise an exception to simulate failure after write
                if write_count == 1:
                    raise OSError("Simulated write failure")
            return original_open(*args, **kwargs)

        import builtins

        builtins.open = mock_open

        try:
            with pytest.raises(OSError):
                await storage.save(prompt)
        finally:
            builtins.open = original_open

        # Even though memory update failed, file should exist
        filepath = os.path.join(temp_dir, "test_prompt.json")
        assert os.path.exists(filepath)
        # But memory should not be updated
        assert "test_prompt" not in storage._prompts

    def test_init_with_default_version_id_none(self, temp_dir: str):
        """Test loading a prompt where defaultVersionId is None (defaults to first version)."""
        filepath = os.path.join(temp_dir, "prompt.json")
        with open(filepath, "w") as f:
            json.dump({"versions": {"v1": "Version 1"}}, f)  # No defaultVersionId

        storage = FilePromptStorage(temp_dir)
        prompt = storage._prompts["prompt"]
        assert prompt.default_version_id == "v1"  # Defaults to first version
