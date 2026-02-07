"""Evaluator optimization module for prompt-specific LLM call evaluation.

This module provides functionality to:
1. Fetch labeled LLM call records for a specific prompt
2. Transform records to the format used by rate_prompt_llm_call
3. Run BootstrapFewShot optimization on the evaluator
4. Store and load optimized DSPy programs
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import dspy

from pixie.agents.rating_agent import (
    PromptLlmCallEvalInput,
    PromptLlmCallEvalSignature,
    RatingResult,
)
from pixie.prompts.prompt_management import get_prompt
from pixie.storage.operations import get_llm_call_records
from pixie.storage.types import LlmCallRecord, RecordFilters

logger = logging.getLogger(__name__)

# Directory for storing optimized evaluators
EVALUATORS_BASE_DIR = ".pixie/evaluators"


def _to_openai_tool_format(tools: list[Any] | None) -> list[dict[str, Any]] | None:
    """Convert flat ToolDefinition format to OpenAI API format.

    This mirrors the frontend's toOpenAIToolFormat function.

    The internal flat format is:
    { name: str, description?: str, parameters?: dict }

    The OpenAI API format is:
    { type: "function", function: { name, description?, parameters? } }

    Args:
        tools: List of flat ToolDefinition objects (or dicts)

    Returns:
        List of OpenAI format tool definitions, or None if empty
    """
    if not tools:
        return None

    result = []
    for tool in tools:
        # Handle both ToolDefinition objects and dicts
        if hasattr(tool, "model_dump"):
            tool_dict = tool.model_dump()
        elif isinstance(tool, dict):
            tool_dict = tool
        else:
            continue

        openai_tool: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": tool_dict.get("name", ""),
            },
        }

        if tool_dict.get("description"):
            openai_tool["function"]["description"] = tool_dict["description"]
        if tool_dict.get("parameters"):
            openai_tool["function"]["parameters"] = tool_dict["parameters"]

        result.append(openai_tool)

    return result if result else None


def _record_to_eval_input(
    record: LlmCallRecord,
    prompt_description: str,
) -> PromptLlmCallEvalInput:
    """Transform an LLM call record to PromptLlmCallEvalInput format.

    This mirrors the transformation done in:
    1. Client (recordToTestCase): gets tools from record.tools
    2. Client (useBatchCallLlm): converts tools to OpenAI format via toOpenAIToolFormat
    3. Server (batch_call_llm): receives tools in OpenAI format, passes to evaluator

    The key transformations:
    - Input messages: from llm_input, with 'tools' removed from each message,
      and tool_calls reformatted to OpenAI format
    - Output: from llm_output (the LLM's response)
    - Tools: from record.tools, converted to OpenAI format {type, function: {name, ...}}
    - Output type: from record.output_type

    Args:
        record: The LLM call record from storage
        prompt_description: Description of the prompt template

    Returns:
        PromptLlmCallEvalInput suitable for the rating agent
    """
    # Extract input messages from llm_input
    # llm_input structure is typically a dict with 'messages' key or a list of messages
    raw_messages: list[Any] = []
    if record.llm_input is not None:
        if isinstance(record.llm_input, dict) and "messages" in record.llm_input:
            messages_value = record.llm_input["messages"]
            if isinstance(messages_value, list):
                raw_messages = messages_value
        elif isinstance(record.llm_input, list):
            raw_messages = list(record.llm_input)

    # Process messages like the frontend does:
    # 1. Remove "tools" from each message (delete ret.tools in recordToTestCase)
    # 2. Reformat tool_calls to OpenAI format (chatmlMessageToOpenAiFormat)
    processed_messages: list[Any] = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            processed_messages.append(msg)
            continue

        processed_msg = dict(msg)

        # Remove 'tools' from message (like frontend: delete ret.tools)
        if "tools" in processed_msg:
            del processed_msg["tools"]

        # Reformat tool_calls to OpenAI format (like chatmlMessageToOpenAiFormat)
        # ChatML format: {name, arguments, id, type}
        # OpenAI format: {function: {name, arguments}, id, type}
        if "tool_calls" in processed_msg and processed_msg["tool_calls"]:
            reformatted_tool_calls = []
            for call in processed_msg["tool_calls"]:
                if isinstance(call, dict):
                    reformatted_tool_calls.append(
                        {
                            "function": {
                                "name": call.get("name", ""),
                                "arguments": call.get("arguments", ""),
                            },
                            "id": call.get("id", ""),
                            "type": call.get("type", "function"),
                        }
                    )
            processed_msg["tool_calls"] = reformatted_tool_calls

        processed_messages.append(processed_msg)

    # Convert record.tools to OpenAI format (like frontend's toOpenAIToolFormat)
    tools_openai_format = _to_openai_tool_format(record.tools)

    return PromptLlmCallEvalInput(
        prompt_description=prompt_description,
        input_messages=processed_messages,
        output=record.llm_output,
        tools=tools_openai_format,
        output_type=record.output_type,
    )


def _record_to_example(
    record: LlmCallRecord, prompt_description: str
) -> dspy.Example | None:
    """Transform an LLM call record to a DSPy Example for training.

    Args:
        record: The LLM call record from storage
        prompt_description: Description of the prompt template

    Returns:
        dspy.Example with all fields including the expected rating
    """
    eval_input = _record_to_eval_input(record, prompt_description)

    rating = record.rating
    if rating is not None:

        return dspy.Example(
            prompt_description=eval_input.prompt_description,
            input_messages=eval_input.input_messages,
            output=eval_input.output,
            tools=eval_input.tools,
            output_type=eval_input.output_type,
            rating=rating.value,
            rating_notes=rating.notes,
        ).with_inputs(
            "prompt_description",
            "input_messages",
            "output",
            "tools",
            "output_type",
        )
    else:
        logger.warning(
            "Skipping record without rating (id: %s)",
            record.rating,
            record.span_id,
        )
        return None


def _get_evaluator_dir(prompt_id: str) -> Path:
    """Get the directory path for storing evaluator files for a prompt.

    Args:
        prompt_id: The unique identifier of the prompt

    Returns:
        Path to the evaluator directory
    """
    return Path(EVALUATORS_BASE_DIR) / prompt_id


def _get_evaluator_filename() -> str:
    """Generate a filename for the evaluator based on current timestamp.

    Returns:
        Filename in format <timestamp>.json
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}.json"


async def fetch_training_data(
    prompt_id: str,
    limit: int = 100,
) -> tuple[list[dspy.Example], str]:
    """Fetch labeled LLM call records for training the evaluator.

    Only fetches records that are:
    - Associated with the specified prompt_id
    - Rated as 'good' or 'bad' (excludes 'undecided')
    - Rated by 'user' or 'system' (excludes 'ai')

    Args:
        prompt_id: The unique identifier of the prompt
        limit: Maximum number of records to fetch

    Returns:
        Tuple of (list of dspy.Example, prompt_description)

    Raises:
        ValueError: If prompt_id is not found or has no description
    """
    # Get prompt description from registry
    prompt_registration = get_prompt(prompt_id)
    if prompt_registration is None:
        raise ValueError(f"Prompt with id '{prompt_id}' not found in registry")

    prompt_description = prompt_registration.description
    if not prompt_description:
        raise ValueError(f"Prompt '{prompt_id}' has no description")

    # Fetch records with filters
    filters = RecordFilters(
        prompt_id=prompt_id,
        rating_values=["good", "bad"],  # Only good/bad, exclude undecided
        rated_by_values=["user", "system"],  # Only user/system, exclude ai
        limit=limit,
    )

    records = await get_llm_call_records(filters)

    # Transform records to examples
    examples = [_record_to_example(r, prompt_description) for r in records]
    examples = [e for e in examples if e is not None]

    logger.info(
        "Fetched %d training examples for prompt '%s'", len(examples), prompt_id
    )

    return examples, prompt_description


def evaluator_metric(
    example: dspy.Example, prediction: dspy.Prediction, trace: Any = None
) -> bool:
    """Metric function for evaluating the DSPy evaluator.

    Compares the predicted rating with the expected rating from the example.

    Args:
        example: The training example with expected rating
        prediction: The model's prediction containing the rating
        trace: Optional trace information (unused)

    Returns:
        True if the prediction matches the expected rating
    """
    expected_rating = example.rating
    predicted_rating = getattr(prediction, "rating", None)

    return predicted_rating == expected_rating


async def optimize_evaluator(
    prompt_id: str,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4,
    max_rounds: int = 5,
    train_limit: int = 100,
) -> Path:
    """Optimize the evaluator for a specific prompt using BootstrapFewShot.

    Fetches labeled LLM call records, runs optimization, and stores the result.

    Args:
        prompt_id: The unique identifier of the prompt
        max_bootstrapped_demos: Maximum number of bootstrapped demonstrations
        max_labeled_demos: Maximum number of labeled demonstrations
        max_rounds: Maximum optimization rounds
        train_limit: Maximum number of training examples to fetch

    Returns:
        Path to the saved optimized evaluator file

    Raises:
        ValueError: If prompt_id is not found or has insufficient training data
    """
    logger.info("Starting evaluator optimization for prompt '%s'", prompt_id)

    # Fetch training data
    examples, prompt_description = await fetch_training_data(
        prompt_id, limit=train_limit
    )

    if len(examples) < 2:
        raise ValueError(
            f"Insufficient training data for prompt '{prompt_id}': "
            f"need at least 2 examples, got {len(examples)}"
        )

    # Split into train/val if we have enough data
    # Use 80% for training, 20% for validation
    split_idx = max(1, int(len(examples) * 0.8))
    trainset = examples[:split_idx]
    valset = examples[split_idx:] if split_idx < len(examples) else []

    logger.info(
        "Split data: %d training examples, %d validation examples",
        len(trainset),
        len(valset),
    )

    # Create the base DSPy program
    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
        dspy_program = dspy.ChainOfThought(PromptLlmCallEvalSignature)

        # Create optimizer
        optimizer = dspy.BootstrapFewShot(
            metric=evaluator_metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            max_rounds=max_rounds,
        )

        # Run optimization
        logger.info("Running BootstrapFewShot optimization...")
        compiled_program = optimizer.compile(dspy_program, trainset=trainset)

        # Validate if we have a validation set
        if valset:
            correct = 0
            for example in valset:
                try:
                    pred = compiled_program(
                        prompt_description=example.prompt_description,
                        input_messages=example.input_messages,
                        output=example.output,
                        tools=example.tools,
                        output_type=example.output_type,
                    )
                    if evaluator_metric(example, pred):
                        correct += 1
                except Exception as e:
                    logger.warning("Validation prediction failed: %s", str(e))

            accuracy = correct / len(valset) if valset else 0
            logger.info(
                "Validation accuracy: %.2f%% (%d/%d)",
                accuracy * 100,
                correct,
                len(valset),
            )

    # Save the optimized program
    evaluator_dir = _get_evaluator_dir(prompt_id)
    evaluator_dir.mkdir(parents=True, exist_ok=True)

    filename = _get_evaluator_filename()
    save_path = evaluator_dir / filename

    compiled_program.save(str(save_path), save_program=False)

    logger.info("Saved optimized evaluator to '%s'", save_path)

    return save_path


def get_latest_optimized_evaluator_path(prompt_id: str) -> Path | None:
    """Get the path to the latest optimized evaluator for a prompt.

    Args:
        prompt_id: The unique identifier of the prompt

    Returns:
        Path to the latest evaluator file, or None if no evaluator exists
    """
    evaluator_dir = _get_evaluator_dir(prompt_id)

    if not evaluator_dir.exists():
        return None

    # Find all JSON files and sort by name (which includes timestamp)
    json_files = sorted(evaluator_dir.glob("*.json"), reverse=True)

    if not json_files:
        return None

    return json_files[0]


def load_optimized_evaluator(prompt_id: str) -> dspy.Module | None:
    """Load the latest optimized evaluator for a prompt.

    Args:
        prompt_id: The unique identifier of the prompt

    Returns:
        The loaded DSPy module, or None if no optimized evaluator exists
    """
    evaluator_path = get_latest_optimized_evaluator_path(prompt_id)

    if evaluator_path is None:
        logger.debug("No optimized evaluator found for prompt '%s'", prompt_id)
        return None

    logger.info("Loading optimized evaluator from '%s'", evaluator_path)

    # Create a new program and load the saved state
    program = dspy.ChainOfThought(PromptLlmCallEvalSignature)
    program.load(str(evaluator_path))

    return program


async def rate_prompt_llm_call_with_optimized(
    rating_input: PromptLlmCallEvalInput,
    prompt_id: str | None = None,
) -> RatingResult:
    """Rate an LLM call using an optimized evaluator if available.

    If an optimized evaluator exists for the prompt, uses it.
    Otherwise, falls back to the default evaluator.

    Args:
        rating_input: The input for rating evaluation
        prompt_id: Optional prompt ID to look up optimized evaluator

    Returns:
        RatingResult with thoughts and rating
    """
    optimized_program = None

    if prompt_id:
        optimized_program = load_optimized_evaluator(prompt_id)

    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
        if optimized_program:
            logger.debug("Using optimized evaluator for prompt '%s'", prompt_id)
            res = await optimized_program.acall(
                prompt_description=rating_input.prompt_description,
                input_messages=rating_input.input_messages,
                output=rating_input.output,
                tools=rating_input.tools,
                output_type=rating_input.output_type,
            )
        else:
            # Fall back to default evaluator
            agent = dspy.ChainOfThought(PromptLlmCallEvalSignature)
            res = await agent.acall(
                prompt_description=rating_input.prompt_description,
                input_messages=rating_input.input_messages,
                output=rating_input.output,
                tools=rating_input.tools,
                output_type=rating_input.output_type,
            )

        return RatingResult(thoughts=res.reasoning, rating=res.rating)


def list_optimized_evaluators(prompt_id: str) -> list[dict[str, Any]]:
    """List all optimized evaluator versions for a prompt.

    Args:
        prompt_id: The unique identifier of the prompt

    Returns:
        List of dicts with 'path' and 'timestamp' for each evaluator
    """
    evaluator_dir = _get_evaluator_dir(prompt_id)

    if not evaluator_dir.exists():
        return []

    evaluators = []
    for json_file in sorted(evaluator_dir.glob("*.json"), reverse=True):
        # Parse timestamp from filename (format: YYYYMMDD_HHMMSS.json)
        timestamp_str = json_file.stem
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except ValueError:
            timestamp = None

        evaluators.append(
            {
                "path": str(json_file),
                "filename": json_file.name,
                "timestamp": timestamp.isoformat() if timestamp else None,
            }
        )

    return evaluators


def delete_optimized_evaluator(prompt_id: str, filename: str) -> bool:
    """Delete a specific optimized evaluator version.

    Args:
        prompt_id: The unique identifier of the prompt
        filename: The filename of the evaluator to delete

    Returns:
        True if deleted successfully, False if file not found
    """
    evaluator_dir = _get_evaluator_dir(prompt_id)
    evaluator_path = evaluator_dir / filename

    if not evaluator_path.exists():
        return False

    evaluator_path.unlink()
    logger.info("Deleted evaluator '%s' for prompt '%s'", filename, prompt_id)

    # Remove directory if empty
    try:
        evaluator_dir.rmdir()
    except OSError:
        pass  # Directory not empty or other error

    return True
