import logging
from types import NoneType

from pixie.prompts.prompt import TPromptVar
from pixie.prompts.storage import StorageBackedPrompt


logger = logging.getLogger(__name__)


_registry: dict[str, StorageBackedPrompt] = {}
"""Registry for StorageBackedPrompts created by `create_prompt`.

StorageBackedPrompt is different from BasePrompt because it can be imcomplete
(when record is not yet fetched from storage, or record doesn't exist at all).
Thus this registry could contain more entries than _prompt_registry in prompt.py."""


def list_prompts() -> list[StorageBackedPrompt]:
    """List all StorageBackedPrompts created via `create_prompt`."""
    return list(_registry.values())


def get_prompt(id: str) -> StorageBackedPrompt | None:
    """Get a StorageBackedPrompt by id, if it was created via `create_prompt`."""
    return _registry.get(id)


def create_prompt(
    id: str,
    variables_definition: type[TPromptVar] = NoneType,
) -> StorageBackedPrompt[TPromptVar]:
    if id in _registry:
        ret = _registry[id]
        if ret.variables_definition != variables_definition:
            raise ValueError(
                f"Prompt with id '{id}' already exists with a different variables definition."
            )
        return ret
    ret = StorageBackedPrompt(id=id, variables_definition=variables_definition)
    _registry[id] = ret
    logger.info(f"✅ Registered prompt: {id}")
    return ret
