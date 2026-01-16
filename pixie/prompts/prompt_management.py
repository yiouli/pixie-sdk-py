from dataclasses import dataclass
import inspect
import logging
from types import NoneType

from pixie.prompts.prompt import TPromptVar
from pixie.prompts.storage import StorageBackedPrompt


logger = logging.getLogger(__name__)


@dataclass
class StorageBackedPromptWithRegistration:
    prompt: StorageBackedPrompt
    description: str | None
    module: str


_registry: dict[str, StorageBackedPromptWithRegistration] = {}
"""Registry for StorageBackedPrompts created by `create_prompt`.

StorageBackedPrompt is different from BasePrompt because it can be imcomplete
(when record is not yet fetched from storage, or record doesn't exist at all).
Thus this registry could contain more entries than _prompt_registry in prompt.py."""


def list_prompts() -> list[StorageBackedPromptWithRegistration]:
    """List all StorageBackedPrompts created via `create_prompt`."""
    return list(_registry.values())


def get_prompt(id: str) -> StorageBackedPromptWithRegistration | None:
    """Get a StorageBackedPrompt by id, if it was created via `create_prompt`."""
    return _registry.get(id)


def _get_calling_module_name():
    """Find the name of the module that called this function."""
    # Get the current frame and the frame above it (the caller's frame)
    try:
        # inspect.stack()[2] gets the caller's frame record
        # frame[0] or frame.frame is the actual frame object
        caller_frame_record = inspect.stack()[2]
        caller_frame = caller_frame_record.frame
    except IndexError:
        # Handle cases where the stack might be shallower, though unlikely for normal calls
        return "__main__"

    # Get the module object from the frame object
    module = inspect.getmodule(caller_frame)

    if module is not None:
        return module.__name__
    else:
        # If getmodule returns None (e.g., if called from the interactive prompt or __main__),
        # try to get the name from the frame's globals
        return caller_frame.f_globals.get("__name__", "__main__")


def create_prompt(
    id: str,
    variables_definition: type[TPromptVar] = NoneType,
    *,
    description: str | None = None,
) -> StorageBackedPrompt[TPromptVar]:
    if id in _registry:
        ret = _registry[id].prompt
        if ret.variables_definition != variables_definition:
            raise ValueError(
                f"Prompt with id '{id}' already exists with a different variables definition."
            )
        return ret
    ret = StorageBackedPrompt(id=id, variables_definition=variables_definition)
    calling_module = _get_calling_module_name()
    _registry[id] = StorageBackedPromptWithRegistration(
        prompt=ret,
        description=description,
        module=calling_module,
    )
    logger.info(f"✅ Registered prompt: {id} ({calling_module})")
    return ret
