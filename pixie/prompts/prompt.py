from copy import deepcopy
from dataclasses import dataclass
from types import NoneType
from typing import Generic, TypeVar, overload
from uuid import uuid4

from pydantic import BaseModel


class PromptVariables(BaseModel):
    # TODO add validation to prevent fields using reserved names
    pass


T = TypeVar("T", bound=PromptVariables | None)


_prompt_registry: dict[str, "Prompt"] = {}


def get_prompt_by_id(prompt_id: str) -> "Prompt":
    return _prompt_registry[prompt_id]


def update_prompt_registry(untyped_prompt: "UntypedPrompt") -> "Prompt":
    existing = get_prompt_by_id(untyped_prompt.id)
    outdated_prompt = OutdatedPrompt.from_prompt(existing)
    _mark_compiled_prompts_outdated(existing.id, outdated_prompt)
    existing.update(
        versions=untyped_prompt.versions,
        default_version_id=untyped_prompt.default_version_id,
    )
    return existing


@dataclass(frozen=True)
class _CompiledPrompt:
    id: str
    value: str
    prompt: "Prompt | OutdatedPrompt"
    version_id: str
    variables: PromptVariables | None


_compiled_prompt_registry: dict[str, _CompiledPrompt] = {}


def _mark_compiled_prompts_outdated(
    prompt_id: str, outdated_prompt: "OutdatedPrompt"
) -> None:
    for key in list(_compiled_prompt_registry.keys()):
        compiled_prompt = _compiled_prompt_registry[key]
        if compiled_prompt.prompt.id == prompt_id:
            _compiled_prompt_registry[key] = _CompiledPrompt(
                id=compiled_prompt.id,
                value=compiled_prompt.value,
                version_id=compiled_prompt.version_id,
                variables=compiled_prompt.variables,
                prompt=outdated_prompt,
            )


def _to_versions_dict(versions: str | dict[str, str]) -> dict[str, str]:
    if isinstance(versions, str):
        return {"default": versions}
    return deepcopy(versions)


class UntypedPrompt:

    def __init__(
        self,
        *,
        versions: str | dict[str, str],
        default_version_id: str | None = None,
        id: str | None = None,
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

    @property
    def id(self) -> str:
        return self._id

    @property
    def version_ids(self) -> set[str]:
        return set(self._versions.keys())

    @property
    def default_version_id(self) -> str:
        return self._default_version

    @property
    def versions(self) -> dict[str, str]:
        return deepcopy(self._versions)


class Prompt(UntypedPrompt, Generic[T]):
    @classmethod
    def from_untyped(
        cls, untyped_prompt: "UntypedPrompt", variable_definitions: type[T] = NoneType
    ) -> "Prompt[T]":
        return cls(
            variable_definitions=variable_definitions,
            versions=untyped_prompt.versions,
            default_version_id=untyped_prompt.default_version_id,
            id=untyped_prompt.id,
        )

    def __init__(
        self,
        *,
        versions: str | dict[str, str],
        default_version_id: str | None = None,
        variable_definitions: type[T] = NoneType,
        id: str | None = None,
    ) -> None:
        super().__init__(
            versions=versions,
            default_version_id=default_version_id,
            id=id,
        )
        self._variable_definitions = variable_definitions
        _prompt_registry[self.id] = self

    @property
    def variable_definitions(self) -> type[T]:
        return self._variable_definitions

    @overload
    def compile(self: "Prompt[NoneType]", *, version_id: str | None = None) -> str: ...

    @overload
    def compile(self, variables: T, *, version_id: str | None = None) -> str: ...

    def compile(
        self,
        variables: T | None = None,
        *,
        version_id: str | None = None,
    ) -> str:
        version_id = version_id or self._default_version
        template = self._versions[version_id]
        if self._variable_definitions is not NoneType:
            if variables is None:
                raise ValueError(
                    f"Variables[{self._variable_definitions}] are required for this prompt."
                )
            ret = template.format(**variables.model_dump(mode="json"))
        else:
            ret = template
        compiled_prompt_id = uuid4().hex
        _compiled_prompt_registry[compiled_prompt_id] = _CompiledPrompt(
            id=compiled_prompt_id,
            value=ret,
            version_id=version_id,
            prompt=self,
            variables=variables,
        )
        return ret

    def update(
        self,
        *,
        versions: str | dict[str, str] | None = None,
        default_version_id: str | None = None,
    ) -> "OutdatedPrompt[T]":
        outdated_prompt = OutdatedPrompt.from_prompt(self)
        if versions is not None:
            self._versions = _to_versions_dict(versions)
        if default_version_id is not None:
            self._default_version = default_version_id
        _mark_compiled_prompts_outdated(self.id, outdated_prompt)
        return outdated_prompt


class OutdatedPrompt(Prompt[T]):

    def __init__(
        self,
        *,
        versions: str | dict[str, str],
        default_version_id: str | NoneType = None,
        variable_definitions: type[T] = NoneType,
        id: str | NoneType = None,
    ) -> NoneType:
        self._id = id
        self._versions = _to_versions_dict(versions)
        self._default_version = default_version_id
        self._variable_definitions = variable_definitions

    @classmethod
    def from_prompt(cls, prompt: Prompt[T]) -> "OutdatedPrompt[T]":
        return cls(
            variable_definitions=prompt.variable_definitions,
            versions=prompt.versions,
            default_version_id=prompt.default_version_id,
            id=prompt.id,
        )

    def update(
        self,
        *,
        versions: str | dict[str, str] | None = None,
        default_version_id: str | None = None,
    ) -> "OutdatedPrompt[T]":
        return self

    def compile(
        self,
        _variables: T | None = None,
        *,
        _version_id: str | None = None,
    ) -> str:
        raise ValueError("This prompt is outdated and can no longer be used.")
