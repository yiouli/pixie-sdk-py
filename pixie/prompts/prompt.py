from copy import deepcopy
from dataclasses import dataclass
import json
from types import NoneType
from typing import Any, Generic, Protocol, Self, TypeVar, cast, overload
from uuid import uuid4

import jinja2
from jinja2 import StrictUndefined
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


def _find_matching_prompt(obj):
    if isinstance(obj, str):
        for compiled in _compiled_prompt_registry.values():
            if compiled.value == obj:
                return compiled
        return None
    elif isinstance(obj, dict):
        for value in obj.values():
            result = _find_matching_prompt(value)
            if result:
                return result
        return None
    elif isinstance(obj, list):
        for item in obj:
            result = _find_matching_prompt(item)
            if result:
                return result
        return None
    else:
        return None


def get_compiled_prompt(text: str) -> _CompiledPrompt | None:
    """Find the compiled prompt metadata for a given compiled prompt string."""
    if not _compiled_prompt_registry:
        return None
    direct_match = _compiled_prompt_registry.get(id(text))
    if direct_match:
        return direct_match
    for compiled in _compiled_prompt_registry.values():
        if compiled.value == text:
            return compiled
    try:
        obj = json.loads(text)
        return _find_matching_prompt(obj)

    except json.JSONDecodeError:
        return None


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

    def get_default_version_id(self) -> str: ...

    def get_versions(self) -> dict[str, str]: ...

    def get_variables_schema(self) -> dict[str, Any]: ...


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

    def get_default_version_id(self) -> str:
        return self._default_version

    def get_versions(self) -> dict[str, str]:
        return deepcopy(self._versions)

    def get_variables_schema(self) -> dict[str, Any]:
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
    def from_untyped(
        cls,
        untyped_prompt: "BaseUntypedPrompt",
        variables_definition: type[TPromptVar] = NoneType,
    ) -> "BasePrompt[TPromptVar]":
        base_schema = untyped_prompt.get_variables_schema()
        typed_schema = variables_definition_to_schema(variables_definition)
        if not isSubschema(typed_schema, base_schema):
            raise TypeError(
                "The provided variables_definition is not compatible with the prompt's variables schema."
            )
        return cls(
            variables_definition=variables_definition,
            versions=untyped_prompt.get_versions(),
            default_version_id=untyped_prompt.get_default_version_id(),
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
        template_txt = self._versions[version_id]
        if self._variables_definition is not NoneType:
            if variables is None:
                raise ValueError(
                    f"Variables[{self._variables_definition}] are required for this prompt."
                )
            template = jinja2.Template(template_txt, undefined=StrictUndefined)
            ret = template.render(**variables.model_dump(mode="json"))
        else:
            ret = template_txt
        _compiled_prompt_registry[id(ret)] = _CompiledPrompt(
            value=ret,
            version_id=version_id,
            prompt=self,
            variables=variables,
        )
        return ret

    def _update(
        self,
        *,
        versions: str | dict[str, str] | None = None,
        default_version_id: str | None = None,
    ) -> "tuple[Self, OutdatedPrompt[TPromptVar]]":
        outdated_prompt = OutdatedPrompt.from_prompt(self)
        if versions is not None:
            self._versions = _to_versions_dict(versions)
        if default_version_id is not None:
            self._default_version = default_version_id
        _mark_compiled_prompts_outdated(self.id, outdated_prompt)
        return self, outdated_prompt

    @staticmethod
    def update_prompt_registry(
        untyped_prompt: "BaseUntypedPrompt",
    ) -> "BasePrompt":
        """IMPORTANT: should only be called from storage on storage load!

        Update the matching entry in type prompt registry in-place.
        DO NOT call other than from initial storage load, to keep immutability of prompts in code.
        """
        existing = get_prompt_by_id(untyped_prompt.id)
        outdated_prompt = OutdatedPrompt.from_prompt(existing)
        _mark_compiled_prompts_outdated(existing.id, outdated_prompt)
        existing._update(
            versions=untyped_prompt.get_versions(),
            default_version_id=untyped_prompt.get_default_version_id(),
        )
        return existing

    def append_version(
        self,
        *,
        version_id: str,
        content: str,
        set_as_default: bool = False,
    ) -> None:
        if version_id in self._versions:
            raise ValueError(f"Version ID '{version_id}' already exists.")
        self._update(
            versions={version_id: content, **self._versions},
            default_version_id=version_id if set_as_default else None,
        )

    def update_default_version_id(
        self,
        default_version_id: str,
    ) -> None:
        if default_version_id not in self._versions:
            raise ValueError(f"Version ID '{default_version_id}' does not exist.")
        if self._default_version == default_version_id:
            return
        self._update(
            default_version_id=default_version_id,
        )


class OutdatedPrompt(BasePrompt[TPromptVar]):

    def __init__(
        self,
        *,
        versions: str | dict[str, str],
        default_version_id: str,
        variables_definition: type[TPromptVar],
        id: str,
    ) -> None:
        self._id = id
        self._versions = _to_versions_dict(versions)
        self._default_version = default_version_id
        self._variables_definition = variables_definition

    @classmethod
    def from_prompt(
        cls, prompt: BasePrompt[TPromptVar]
    ) -> "OutdatedPrompt[TPromptVar]":
        return cls(
            variables_definition=prompt.variables_definition,
            versions=prompt.get_versions(),
            default_version_id=prompt.get_default_version_id(),
            id=prompt.id,
        )

    def _update(
        self,
        *,
        versions: str | dict[str, str] | None = None,
        default_version_id: str | None = None,
    ) -> "OutdatedPrompt[TPromptVar]":
        raise ValueError("Cannot update an outdated prompt.")

    def get_default_version_id(self) -> str:
        return self._default_version

    def get_versions(self) -> dict[str, str]:
        return deepcopy(self._versions)

    def compile(
        self,
        _variables: TPromptVar | None = None,
        *,
        _version_id: str | None = None,
    ) -> str:
        raise ValueError("This prompt is outdated and can no longer be used.")
