import dspy
from pydantic import BaseModel, JsonValue

from pixie.storage.types import Message, Rating


class AppRunRatingAgentInput(dspy.Signature):
    app_description: str = dspy.InputField()
    interaction_logs: list[Message] = dspy.InputField()


class LlmCallRatingAgentInput(dspy.Signature):
    app_description: str = dspy.InputField()
    interaction_logs_before_llm_call: list[Message] = dspy.InputField()
    llm_input: JsonValue = dspy.InputField()
    llm_output: JsonValue = dspy.InputField()
    llm_configuration: JsonValue = dspy.InputField()
    internal_logs_after_llm_call: list[JsonValue] = dspy.InputField()
    interaction_logs_after_llm_call: list[Message] = dspy.InputField()


class AppRunRatingAgentSignature(AppRunRatingAgentInput):
    """Rate the overall quality of an application execution."""

    rating: Rating = dspy.OutputField()


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


async def rate_app_run(rating_input: AppRunRatingAgentInput) -> RatingResult:
    """DSPy chain-of-thought agent to rate the overall quality of an application execution."""
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
    agent = dspy.ChainOfThought(AppRunRatingAgentSignature)
    res = await agent.acall(**rating_input.model_dump())
    return RatingResult(thoughts=res.reasoning, rating=res.rating)
