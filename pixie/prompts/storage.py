from typing_extensions import Protocol
from .prompt import UntypedPrompt, update_prompt_registry, get_prompt_by_id
import json
import os
from typing import Dict


class PromptStorage(Protocol):

    async def exists(self, prompt_id: str) -> bool: ...

    async def save(self, prompt: UntypedPrompt) -> None: ...

    async def get(self, prompt_id: str) -> UntypedPrompt: ...


class FilePromptStorage:

    def __init__(self, directory: str) -> None:
        self._directory = directory
        self._prompts: Dict[str, UntypedPrompt] = {}
        if not os.path.exists(directory):
            os.makedirs(directory)
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                prompt_id = filename[:-5]  # remove .json
                filepath = os.path.join(directory, filename)
                with open(filepath, "r") as f:
                    data = json.load(f)
                versions = data["versions"]
                default_version_id = data.get("defaultVersionId")
                prompt = UntypedPrompt(
                    id=prompt_id,
                    versions=versions,
                    default_version_id=default_version_id,
                )
                self._prompts[prompt_id] = prompt

    async def exists(self, prompt_id: str) -> bool:
        return prompt_id in self._prompts

    async def save(self, prompt: UntypedPrompt) -> None:
        prompt_id = prompt.id
        try:
            get_prompt_by_id(prompt_id)
        except KeyError:
            raise KeyError(f"Prompt with ID '{prompt_id}' does not exist.")
        data = {
            "versions": prompt.versions,
            "defaultVersionId": prompt.default_version_id,
        }
        filepath = os.path.join(self._directory, f"{prompt_id}.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        update_prompt_registry(prompt)
        self._prompts[prompt_id] = prompt

    async def get(self, prompt_id: str) -> UntypedPrompt:
        return self._prompts[prompt_id]
