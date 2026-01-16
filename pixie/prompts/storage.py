import asyncio
import json
import logging
import os
from types import NoneType
from typing import Any, Dict, Self, TypedDict
from typing_extensions import Protocol
import nest_asyncio

from jsonsubschema import isSubschema

from .prompt import (
    BasePrompt,
    BaseUntypedPrompt,
    Prompt,
    TPromptVar,
    variables_definition_to_schema,
)

# This is critical for the sync version of StorageBackedPrompt.compile to work
nest_asyncio.apply()

logger = logging.getLogger(__name__)


class PromptStorage(Protocol):

    async def exists(self, prompt_id: str) -> bool: ...

    async def save(self, prompt: BaseUntypedPrompt) -> None: ...

    async def get(self, prompt_id: str) -> BaseUntypedPrompt: ...


class _BasePromptJson(TypedDict):
    versions: Dict[str, str]
    defaultVersionId: str
    variablesSchema: Dict[str, Any]


class _FilePromptStorage(PromptStorage):

    def __init__(self, directory: str) -> None:
        self._directory = directory
        self._prompts: Dict[str, BaseUntypedPrompt] = {}
        """prompts that are in storage"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                prompt_id = filename[:-5]  # remove .json
                filepath = os.path.join(directory, filename)
                with open(filepath, "r") as f:
                    data: _BasePromptJson = json.load(f)
                versions = data["versions"]
                default_version_id = data["defaultVersionId"]
                variables_schema = data["variablesSchema"]
                prompt = BaseUntypedPrompt(
                    id=prompt_id,
                    versions=versions,
                    default_version_id=default_version_id,
                    variables_schema=variables_schema,
                )
                self._prompts[prompt_id] = prompt

    async def exists(self, prompt_id: str) -> bool:
        return prompt_id in self._prompts

    async def save(self, prompt: BaseUntypedPrompt) -> bool:
        prompt_id = prompt.id
        original = self._prompts.get(prompt_id)
        new_schema = await prompt.get_variables_schema()
        if original:
            original_schema = await original.get_variables_schema()
            if not isSubschema(original_schema, new_schema):
                raise TypeError(
                    "Original schema must be a subschema of the new schema."
                )
        data: _BasePromptJson = {
            "versions": await prompt.get_versions(),
            "defaultVersionId": await prompt.get_default_version_id(),
            "variablesSchema": await prompt.get_variables_schema(),
        }
        filepath = os.path.join(self._directory, f"{prompt_id}.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        try:
            await BasePrompt.update_prompt_registry(prompt)
        except KeyError:
            # Prompt not in type prompt registry yet, meaning there's no usage in code
            # thus this untyped prompt would just be stored but not used in code
            pass
        self._prompts[prompt_id] = prompt
        return original is None

    async def get(self, prompt_id: str) -> BaseUntypedPrompt:
        return self._prompts[prompt_id]


_storage_instance: PromptStorage | None = None


# TODO allow other storage types later
def initialize_prompt_storage(directory: str) -> None:
    global _storage_instance
    if _storage_instance is not None:
        raise RuntimeError("Prompt storage has already been initialized.")
    _storage_instance = _FilePromptStorage(directory)
    logger.info(f"Initialized prompt storage at directory: {directory}")


class StorageBackedPrompt(Prompt[TPromptVar]):

    def __init__(
        self,
        id: str,
        *,
        variables_definition: type[TPromptVar] = NoneType,
    ) -> None:
        self._id = id
        self._variables_definition = variables_definition
        self._prompt: BasePrompt[TPromptVar] | None = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def variables_definition(self) -> type[TPromptVar]:
        return self._variables_definition

    async def get_variables_schema(self) -> dict[str, Any]:
        return variables_definition_to_schema(self._variables_definition)

    async def _get_prompt(self) -> BasePrompt[TPromptVar]:
        if _storage_instance is None:
            raise RuntimeError("Prompt storage has not been initialized.")
        if self._prompt is None:
            untyped_prompt = await _storage_instance.get(self.id)
            self._prompt = await BasePrompt.from_untyped(
                untyped_prompt,
                variables_definition=self.variables_definition,
            )
            schema_from_storage = await untyped_prompt.get_variables_schema()
            schema_from_definition = await self.get_variables_schema()
            if not isSubschema(schema_from_definition, schema_from_storage):
                raise TypeError(
                    "Schema from definition is not a subschema of the schema from storage."
                )
        return self._prompt

    async def actualize(self) -> Self:
        await self._get_prompt()
        return self

    async def exists_in_storage(self) -> bool:
        if _storage_instance is None:
            raise RuntimeError("Prompt storage has not been initialized.")
        try:
            await self.actualize()
            return True
        except KeyError:
            return False

    async def get_versions(self) -> dict[str, str]:
        prompt = await self._get_prompt()
        return await prompt.get_versions()

    async def get_default_version_id(self) -> str:
        prompt = await self._get_prompt()
        return await prompt.get_default_version_id()

    async def acompile(
        self,
        variables: TPromptVar = None,
        *,
        version_id: str | None = None,
    ) -> str:
        prompt = await self._get_prompt()
        return prompt.compile(variables=variables, version_id=version_id)

    def compile(
        self,
        variables: TPromptVar = None,
        *,
        version_id: str | None = None,
    ) -> str:
        return asyncio.run(
            self.acompile(
                variables,
                version_id=version_id,
            )
        )

    async def append_version(
        self,
        version_id: str,
        content: str,
        set_as_default: bool = False,
    ) -> BasePrompt[TPromptVar]:
        if _storage_instance is None:
            raise RuntimeError("Prompt storage has not been initialized.")
        prompt = await self._get_prompt()
        await prompt.append_version(
            version_id=version_id,
            content=content,
            set_as_default=set_as_default,
        )
        await _storage_instance.save(prompt)
        return prompt

    async def update_default_version_id(
        self,
        version_id: str,
    ) -> BasePrompt[TPromptVar]:
        if _storage_instance is None:
            raise RuntimeError("Prompt storage has not been initialized.")
        prompt = await self._get_prompt()
        await prompt.update_default_version_id(version_id)
        await _storage_instance.save(prompt)
        return prompt
