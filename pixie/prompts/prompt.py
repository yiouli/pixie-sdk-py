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
    try:
        existing = get_prompt_by_id(untyped_prompt.id)
        var_def = existing.variable_definitions
        existing.invalidate()
    except KeyError:
        # If not in registry, it's a new prompt, use NoneType
        var_def = NoneType
    ret = _prompt_registry[untyped_prompt.id] = Prompt.from_untyped(
        untyped_prompt,
        variable_definitions=var_def,
    )
    return ret


@dataclass(frozen=True)
class _CompiledPrompt:
    id: str
    value: str
    prompt: "Prompt"
    version_id: str
    variables: PromptVariables | None


_compiled_prompt_registry: dict[str, _CompiledPrompt] = {}


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
        if id in _prompt_registry:
            raise ValueError(f"Prompt with ID '{id}' already exists.")

        self._id = id
        if not versions:
            raise ValueError("No versions provided for the prompt.")
        if isinstance(versions, str):
            self._versions = {"default": versions}
        else:
            self._versions = deepcopy(versions)
        self._default_version = default_version_id or next(iter(self._versions))
        self.is_valid = True

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
        self._raise_if_invalid()
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
        self._raise_if_invalid()
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

    def _raise_if_invalid(self) -> None:
        if not self.is_valid:
            raise ValueError("This prompt is no longer valid.")

    def invalidate(self) -> None:
        self.is_valid = False
        _prompt_registry.pop(self._id, None)
        for compiled_prompt_id, compiled_prompt in list(
            _compiled_prompt_registry.items()
        ):
            if compiled_prompt.prompt.id == self._id:
                _compiled_prompt_registry.pop(compiled_prompt_id, None)
