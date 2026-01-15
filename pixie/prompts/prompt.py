from copy import deepcopy
from dataclasses import dataclass
from types import NoneType
from typing import Any, Generic, Protocol, Self, TypeVar, cast, overload
from uuid import uuid4

from jsonsubschema import isSubschema
from pydantic import BaseModel


class PromptVariables(BaseModel):
    # TODO add validation to prevent fields using reserved names
    pass


TPromptVar = TypeVar("TPromptVar", bound=PromptVariables | None)


_prompt_registry: dict[str, "BasePrompt"] = {}
"""Registry of all actualized prompts.

Purpose of the registry is to ensure there's single actualized prompt instance per prompt ID globally,
so that every compiled prompt can track back to one single instance of the prompt it was compiled from.
"""


def get_prompt_by_id(prompt_id: str) -> "BasePrompt":
    return _prompt_registry[prompt_id]


async def update_prompt_registry(untyped_prompt: "BaseUntypedPrompt") -> "BasePrompt":
    existing = get_prompt_by_id(untyped_prompt.id)
    outdated_prompt = await OutdatedPrompt.from_prompt(existing)
    _mark_compiled_prompts_outdated(existing.id, outdated_prompt)
    await existing.update(
        versions=await untyped_prompt.get_versions(),
        default_version_id=await untyped_prompt.get_default_version_id(),
    )
    return existing


@dataclass(frozen=True)
class _CompiledPrompt:
    value: str
    prompt: "BasePrompt | OutdatedPrompt"
    version_id: str
    variables: PromptVariables | None


_compiled_prompt_registry: dict[int, _CompiledPrompt] = {}
"""Registry of all compiled prompts.

This is to keep track of every result string returned by BasePrompt.compile().
key is the id() of the compiled string."""


def _mark_compiled_prompts_outdated(
    prompt_id: str, outdated_prompt: "OutdatedPrompt"
) -> None:
    for key in list(_compiled_prompt_registry.keys()):
        compiled_prompt = _compiled_prompt_registry[key]
        if compiled_prompt.prompt.id == prompt_id:
            _compiled_prompt_registry[key] = _CompiledPrompt(
                value=compiled_prompt.value,
                version_id=compiled_prompt.version_id,
                variables=compiled_prompt.variables,
                prompt=outdated_prompt,
            )


DEFAULT_VERSION_ID = "v0"


def _to_versions_dict(versions: str | dict[str, str]) -> dict[str, str]:
    if isinstance(versions, str):
        return {DEFAULT_VERSION_ID: versions}
    return deepcopy(versions)


class _UnTypedPrompt(Protocol):
    @property
    def id(self) -> str: ...

    async def get_default_version_id(self) -> str: ...

    async def get_versions(self) -> dict[str, str]: ...

    async def get_variables_schema(self) -> dict[str, Any]: ...


EMPTY_VARIABLES_SCHEMA = {"type": "object", "properties": {}}


class BaseUntypedPrompt(_UnTypedPrompt):

    def __init__(
        self,
        *,
        versions: str | dict[str, str],
        default_version_id: str | None = None,
        id: str | None = None,
        variables_schema: dict[str, Any] | None = None,
    ) -> None:
        if not id:
            id = uuid4().hex
            while id in _prompt_registry:
                id = uuid4().hex

        self._id = id
        if not versions:
            raise ValueError("No versions provided for the prompt.")
        self._versions: dict[str, str]
        self._versions = _to_versions_dict(versions)
        self._default_version = default_version_id or next(iter(self._versions))
        self._variables_schema = variables_schema or EMPTY_VARIABLES_SCHEMA

    @property
    def id(self) -> str:
        return self._id

    async def get_default_version_id(self) -> str:
        return self._default_version

    async def get_versions(self) -> dict[str, str]:
        return deepcopy(self._versions)

    async def get_variables_schema(self) -> dict[str, Any]:
        return deepcopy(self._variables_schema)


class Prompt(_UnTypedPrompt, Generic[TPromptVar]):
    @property
    def variables_definition(self) -> type[TPromptVar]: ...

    @overload
    def compile(
        self: "BasePrompt[NoneType]", *, version_id: str | None = None
    ) -> str: ...

    @overload
    def compile(
        self, variables: TPromptVar, *, version_id: str | None = None
    ) -> str: ...

    def compile(
        self,
        variables: TPromptVar | None = None,
        *,
        version_id: str | None = None,
    ) -> str: ...


def variables_definition_to_schema(definition: type[TPromptVar]) -> dict[str, Any]:
    if definition is NoneType:
        return EMPTY_VARIABLES_SCHEMA

    return cast(type[PromptVariables], definition).model_json_schema()


class BasePrompt(BaseUntypedPrompt, Generic[TPromptVar]):
    @classmethod
    async def from_untyped(
        cls,
        untyped_prompt: "BaseUntypedPrompt",
        variables_definition: type[TPromptVar] = NoneType,
    ) -> "BasePrompt[TPromptVar]":
        base_schema = await untyped_prompt.get_variables_schema()
        typed_schema = variables_definition_to_schema(variables_definition)
        if not isSubschema(typed_schema, base_schema):
            raise TypeError(
                "The provided variables_definition is not compatible with the prompt's variables schema."
            )
        return cls(
            variables_definition=variables_definition,
            versions=await untyped_prompt.get_versions(),
            default_version_id=await untyped_prompt.get_default_version_id(),
            id=untyped_prompt.id,
        )

    def __init__(
        self,
        *,
        versions: str | dict[str, str],
        default_version_id: str | None = None,
        variables_definition: type[TPromptVar] = NoneType,
        id: str | None = None,
    ) -> None:
        super().__init__(
            versions=versions,
            default_version_id=default_version_id,
            id=id,
            variables_schema=variables_definition_to_schema(variables_definition),
        )
        self._variables_definition = variables_definition
        _prompt_registry[self.id] = self

    @property
    def variables_definition(self) -> type[TPromptVar]:
        return self._variables_definition

    @overload
    def compile(
        self: "BasePrompt[NoneType]", *, version_id: str | None = None
    ) -> str: ...

    @overload
    def compile(
        self, variables: TPromptVar, *, version_id: str | None = None
    ) -> str: ...

    def compile(
        self,
        variables: TPromptVar | None = None,
        *,
        version_id: str | None = None,
    ) -> str:
        version_id = version_id or self._default_version
        template = self._versions[version_id]
        if self._variables_definition is not NoneType:
            if variables is None:
                raise ValueError(
                    f"Variables[{self._variables_definition}] are required for this prompt."
                )
            ret = template.format(**variables.model_dump(mode="json"))
        else:
            ret = template
        _compiled_prompt_registry[id(ret)] = _CompiledPrompt(
            value=ret,
            version_id=version_id,
            prompt=self,
            variables=variables,
        )
        return ret

    async def update(
        self,
        *,
        versions: str | dict[str, str] | None = None,
        default_version_id: str | None = None,
    ) -> "tuple[Self, OutdatedPrompt[TPromptVar]]":
        outdated_prompt = await OutdatedPrompt.from_prompt(self)
        if versions is not None:
            self._versions = _to_versions_dict(versions)
        if default_version_id is not None:
            self._default_version = default_version_id
        _mark_compiled_prompts_outdated(self.id, outdated_prompt)
        return self, outdated_prompt


class OutdatedPrompt(BasePrompt[TPromptVar]):

    def __init__(
        self,
        *,
        versions: str | dict[str, str],
        default_version_id: str,
        variables_definition: type[TPromptVar],
        id: str,
    ) -> NoneType:
        self._id = id
        self._versions = _to_versions_dict(versions)
        self._default_version = default_version_id
        self._variables_definition = variables_definition

    @classmethod
    async def from_prompt(
        cls, prompt: BasePrompt[TPromptVar]
    ) -> "OutdatedPrompt[TPromptVar]":
        return cls(
            variables_definition=prompt.variables_definition,
            versions=await prompt.get_versions(),
            default_version_id=await prompt.get_default_version_id(),
            id=prompt.id,
        )

    def update(
        self,
        *,
        versions: str | dict[str, str] | None = None,
        default_version_id: str | None = None,
    ) -> "OutdatedPrompt[TPromptVar]":
        raise ValueError("Cannot update an outdated prompt.")

    async def get_default_version_id(self) -> str:
        return self._default_version

    async def get_versions(self) -> dict[str, str]:
        return deepcopy(self._versions)

    def compile(
        self,
        _variables: TPromptVar | None = None,
        *,
        _version_id: str | None = None,
    ) -> str:
        raise ValueError("This prompt is outdated and can no longer be used.")
