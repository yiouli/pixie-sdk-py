import dspy
from pydantic import BaseModel, JsonValue

from pixie.storage.types import Message, Rating


class FindBadResponseInput(BaseModel):
    """Input for finding the problematic response in a conversation."""

    ai_description: str
    conversation: list[Message]
    reasoning_for_negative_rating: str


class LlmCallRatingAgentInput(dspy.Signature):
    app_description: str = dspy.InputField()
    interaction_logs_before_llm_call: list[Message] = dspy.InputField()
    llm_input: JsonValue = dspy.InputField()
    llm_output: JsonValue = dspy.InputField()
    llm_configuration: JsonValue = dspy.InputField()
    internal_logs_after_llm_call: list[JsonValue] = dspy.InputField()
    interaction_logs_after_llm_call: list[Message] = dspy.InputField()


class LlmCallRatingAgentSignature(LlmCallRatingAgentInput):
    """Rate the quality of a specific LLM call within an application execution."""

    rating: Rating = dspy.OutputField()


class RatingResult(BaseModel):
    thoughts: str
    rating: Rating


async def rate_llm_call(rating_input: LlmCallRatingAgentInput) -> RatingResult:
    """DSPy chain-of-thought agent to rate the quality of a specific LLM call within an application execution."""
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
    agent = dspy.ChainOfThought(LlmCallRatingAgentSignature)
    res = await agent.acall(**rating_input.model_dump())
    return RatingResult(thoughts=res.reasoning, rating=res.rating)


class AppRunRatingAgentInput(dspy.Signature):
    app_description: str = dspy.InputField()
    interaction_logs: list[Message] = dspy.InputField()


class AppRunRatingAgentSignature(AppRunRatingAgentInput):
    """Rate the overall quality of an application execution."""

    rating: Rating = dspy.OutputField()


async def rate_app_run(rating_input: AppRunRatingAgentInput) -> RatingResult:
    """DSPy chain-of-thought agent to rate the overall quality of an application execution."""
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
    agent = dspy.ChainOfThought(AppRunRatingAgentSignature)
    res = await agent.acall(**rating_input.model_dump())
    return RatingResult(thoughts=res.reasoning, rating=res.rating)


class FindBadResponseAgentInput(dspy.Signature):
    ai_description: str = dspy.InputField()
    conversation: list[Message] = dspy.InputField()
    reasoning_for_negative_rating: str = dspy.InputField()


class FindBadResponseSignature(FindBadResponseAgentInput):
    """Identify the index of the main assistant message in the conversation,
    that leads to the negative user rating for the conversation with AI."""

    bad_ai_response_index: int = dspy.OutputField()


class FindBadResponseAgentSignature(FindBadResponseAgentInput):
    """Identify the main assistant response in the conversation,
    that leads to the negative user rating for the conversation with AI."""

    bad_response: Message = dspy.OutputField()


class FixIndexSignature(dspy.Signature):
    """Given the reasoning of which assistant message is bad in the conversation,
    and a list of wrong indices, pick the correct index."""

    conversation: list[Message] = dspy.InputField()
    wrong_indices: list[int] = dspy.InputField()
    reasons_for_wrong_indices: list[str] = dspy.InputField()
    bad_ai_message_index: int = dspy.OutputField()


class FindBadResponseAgent(dspy.Module):
    def __init__(self):
        self.find_bad_response = dspy.ChainOfThought(FindBadResponseSignature)
        self.fix_index = dspy.ChainOfThought(FixIndexSignature)

    def forward(
        self, input: FindBadResponseAgentInput
    ) -> FindBadResponseAgentSignature:
        res = self.find_bad_response(
            app_description=input.ai_description,
            conversation=input.conversation,
            reasoning_for_negative_rating=input.reasoning_for_negative_rating,
        )
        target_index = res.bad_ai_response_index

        # If the index is out of bounds, try to fix it
        wrong_indices = []
        reasons_for_wrong_indices = []

        while True:
            if target_index < 0 or target_index >= len(input.conversation):
                wrong_indices.append(target_index)
                reasons_for_wrong_indices.append(
                    f"Index {target_index} is out of bounds for the conversation."
                )

            elif target_index not in wrong_indices:
                message = input.conversation[target_index]
                if message.role != "assistant":
                    wrong_indices.append(target_index)
                    content_preview = (
                        message.content[:25]
                        if hasattr(message, "content")
                        else str(message)[:25]
                    )
                    reasons_for_wrong_indices.append(
                        f"Message at index {target_index} is not from AI: [{message.role}]'{content_preview}...'"
                    )
            elif target_index in wrong_indices:
                wrong_indices.append(target_index)
                reasons_for_wrong_indices.append(
                    f"Index {target_index} has already been identified as wrong."
                )
            else:
                return FindBadResponseAgentSignature(
                    app_description=input.ai_description,
                    conversation=input.conversation,
                    reasoning_for_negative_rating=input.reasoning_for_negative_rating,
                    bad_response=input.conversation[target_index],
                )

            target_index = self.fix_index(
                conversation=input.conversation,
                wrong_indices=wrong_indices,
                reasons_for_wrong_indices=reasons_for_wrong_indices,
            ).bad_ai_message_index


async def find_bad_response(input: FindBadResponseInput) -> Message:
    """DSPy chain-of-thought agent to identify the main assistant response in the conversation
    that leads a negative rating."""
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
    agent = FindBadResponseAgent()
    res = await agent.acall(**input.model_dump())
    return res.bad_response
