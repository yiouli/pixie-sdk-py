from copy import deepcopy
from types import NoneType
from typing import Generic, TypeVar, overload

from pydantic import BaseModel


class PromptVariables(BaseModel):
    # TODO add validation to prevent fields using reserved names
    pass


T = TypeVar("T", bound=PromptVariables | NoneType)


class Prompt(Generic[T]):
    def __init__(
        self,
        *,
        versions: str | dict[str, str],
        variableDefinitions: type[T] = NoneType,
        default_version_id: str | None = None,
    ) -> None:
        if not versions:
            raise ValueError("No versions provided for the prompt.")
        if isinstance(versions, str):
            self._versions = {"default": versions}
        else:
            self._versions = deepcopy(versions)
        self._variable_definitions = variableDefinitions
        self._default_version = default_version_id or next(iter(self._versions))

    @property
    def version_ids(self) -> set[str]:
        return set(self._versions.keys())

    @property
    def default_version_id(self) -> str:
        return self._default_version

    @overload
    def compile(self: "Prompt[NoneType]", *, version_id: str | None = None) -> str: ...

    @overload
    def compile(self, variables: T, *, version_id: str | None = None) -> str: ...

    def compile(
        self, variables: T | None = None, *, version_id: str | None = None
    ) -> str:
        version_id = version_id or self._default_version
        template = self._versions[version_id]
        if self._variable_definitions is not NoneType:
            if variables is None:
                raise ValueError("Variables are required for this prompt.")
            return template.format(**variables.model_dump())
        return template
