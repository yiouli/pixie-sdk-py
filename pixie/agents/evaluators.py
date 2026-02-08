from typing import Any, Literal
import dspy
from pydantic import BaseModel

from pixie.storage.types import Message, Rating


class FindBadResponseInput(BaseModel):
    """Input for finding the problematic response in a conversation."""

    ai_description: str
    conversation: list[Message]
    reasoning_for_negative_rating: str


class LlmCallRatingAgentInput(dspy.Signature):
    app_description: str = dspy.InputField()
    interaction_logs_before_llm_call: list[Message] = dspy.InputField()
    llm_input: list = dspy.InputField()
    llm_output: list | dict = dspy.InputField()
    llm_configuration: dict = dspy.InputField()
    internal_logs_after_llm_call: list[dict] = dspy.InputField()
    interaction_logs_after_llm_call: list[Message] = dspy.InputField()


class LlmCallRatingAgentSignature(LlmCallRatingAgentInput):
    """Rate the quality of a specific LLM call within an application execution."""

    rating: Rating = dspy.OutputField()


class RatingResult(BaseModel):
    thoughts: str
    rating: Rating


async def rate_llm_call(rating_input: LlmCallRatingAgentInput) -> RatingResult:
    """DSPy chain-of-thought agent to rate the quality of a specific LLM call within an application execution."""
    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
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
    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
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

    async def aforward(
        self,
        ai_description: str,
        conversation: list[Message],
        reasoning_for_negative_rating: str,
    ) -> FindBadResponseSignature:
        res = await self.find_bad_response.acall(
            ai_description=ai_description,
            conversation=conversation,
            reasoning_for_negative_rating=reasoning_for_negative_rating,
        )
        target_index = res.bad_ai_response_index

        # If the index is out of bounds, try to fix it
        wrong_indices = []
        reasons_for_wrong_indices = []

        tries = 0

        while tries < 3:
            if target_index < 0 or target_index >= len(conversation):
                wrong_indices.append(target_index)
                reasons_for_wrong_indices.append(
                    f"Index {target_index} is out of bounds for the conversation."
                )

            elif target_index in wrong_indices:
                wrong_indices.append(target_index)
                reasons_for_wrong_indices.append(
                    f"Index {target_index} has already been identified as wrong."
                )
            else:
                message = conversation[target_index]
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
                else:
                    return FindBadResponseSignature(
                        ai_description=ai_description,
                        conversation=conversation,
                        reasoning_for_negative_rating=reasoning_for_negative_rating,
                        bad_ai_response_index=target_index,
                    )
            tries += 1

            target_index = (
                await self.fix_index.acall(
                    conversation=conversation,
                    wrong_indices=wrong_indices,
                    reasons_for_wrong_indices=reasons_for_wrong_indices,
                )
            ).bad_ai_message_index

        raise ValueError(
            "Failed to identify the bad AI response after multiple attempts."
        )


async def find_bad_response(input: FindBadResponseInput) -> int:
    """DSPy chain-of-thought agent to identify the main assistant response in the conversation
    that leads a negative rating."""
    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
        agent = FindBadResponseAgent()
        res = await agent.acall(
            ai_description=input.ai_description,
            conversation=input.conversation,
            reasoning_for_negative_rating=input.reasoning_for_negative_rating,
        )
        return res.bad_ai_response_index


class LlmCallSpan(BaseModel):
    span_type: Literal["llm_call"]
    llm_input: list
    llm_output: list | dict


class FindBadLlmCallInput(BaseModel):
    """Input for finding the problematic LLM call span in a trace."""

    ai_description: str
    conversation: list[Message]
    trace: list[LlmCallSpan | dict]
    reasoning_for_negative_rating: str


class FindBadLlmCallAgentInput(dspy.Signature):
    ai_description: str = dspy.InputField()
    conversation: list[Message] = dspy.InputField(
        desc="Conversation leading up to the bad response. The last item is the response being labeled bad."
    )
    trace: list[LlmCallSpan | dict] = dspy.InputField(
        desc="Server trace spans for the last message. "
        "Some are LLM call spans (with span_type='llm_call'), some are not."
    )
    reasoning_for_negative_rating: str = dspy.InputField(
        desc="Notes explaining why the last message in conversation received a bad rating."
    )


class FindBadLlmCallSignature(FindBadLlmCallAgentInput):
    """Identify the index of the LLM call span in the trace that is most likely
    responsible for the problematic assistant response."""

    bad_llm_call_index: int = dspy.OutputField(
        desc="The index of the problematic LLM call span in the trace."
    )


class FixLlmCallIndexSignature(dspy.Signature):
    """Given the reasoning of which LLM call is problematic in the trace,
    and a list of wrong indices, pick the correct index."""

    trace: list[LlmCallSpan | dict] = dspy.InputField()
    wrong_indices: list[int] = dspy.InputField()
    reasons_for_wrong_indices: list[str] = dspy.InputField()
    bad_llm_call_index: int = dspy.OutputField()


class FindBadLlmCallAgent(dspy.Module):
    def __init__(self):
        self.find_bad_llm_call = dspy.ChainOfThought(FindBadLlmCallSignature)
        self.fix_index = dspy.ChainOfThought(FixLlmCallIndexSignature)

    async def aforward(
        self,
        ai_description: str,
        conversation: list[Message],
        trace: list[LlmCallSpan | dict],
        reasoning_for_negative_rating: str,
    ) -> FindBadLlmCallSignature:
        res = await self.find_bad_llm_call.acall(
            ai_description=ai_description,
            conversation=conversation,
            trace=trace,
            reasoning_for_negative_rating=reasoning_for_negative_rating,
        )
        target_index = res.bad_llm_call_index

        # If the index is out of bounds or not an LLM call span, try to fix it
        wrong_indices = []
        reasons_for_wrong_indices = []

        tries = 0

        while tries < 3:
            if target_index < 0 or target_index >= len(trace):
                wrong_indices.append(target_index)
                reasons_for_wrong_indices.append(
                    f"Index {target_index} is out of bounds for the trace."
                )
            elif target_index in wrong_indices:
                wrong_indices.append(target_index)
                reasons_for_wrong_indices.append(
                    f"Index {target_index} has already been identified as wrong."
                )
            else:
                span = trace[target_index]
                # Check if it's an LLM call span
                span_type = (
                    span.span_type
                    if isinstance(span, LlmCallSpan)
                    else span.get("span_type")
                )
                if span_type != "llm_call":
                    wrong_indices.append(target_index)
                    reasons_for_wrong_indices.append(
                        f"Span at index {target_index} is not an LLM call span (span_type='{span_type}')."
                    )
                else:
                    return FindBadLlmCallSignature(
                        ai_description=ai_description,
                        conversation=conversation,
                        trace=trace,
                        reasoning_for_negative_rating=reasoning_for_negative_rating,
                        bad_llm_call_index=target_index,
                    )
            tries += 1

            target_index = (
                await self.fix_index.acall(
                    trace=trace,
                    wrong_indices=wrong_indices,
                    reasons_for_wrong_indices=reasons_for_wrong_indices,
                )
            ).bad_llm_call_index

        raise ValueError("Failed to identify the bad LLM call after multiple attempts.")


async def find_bad_llm_call(input: FindBadLlmCallInput) -> int:
    """DSPy chain-of-thought agent to identify the LLM call span in a trace
    that is most likely responsible for a problematic assistant response."""
    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
        agent = FindBadLlmCallAgent()
        res = await agent.acall(
            ai_description=input.ai_description,
            conversation=input.conversation,
            trace=input.trace,
            reasoning_for_negative_rating=input.reasoning_for_negative_rating,
        )
        return res.bad_llm_call_index


# ============================================================================
# Prompt-Based LLM Call Evaluation Agent
# ============================================================================


class PromptLlmCallEvalInput(BaseModel):
    """Input for evaluating an LLM call using prompt template context."""

    prompt_description: str
    input_messages: list[Any]
    output: Any
    tools: list[Any] | None = None
    output_type: Any | None = None


class PromptLlmCallEvalSignature(dspy.Signature):
    """Evaluate whether the LLM's immediate response is the correct NEXT STEP given the conversation so far.

    IMPORTANT: You are evaluating a SINGLE turn in what may be a multi-step interaction.
    The LLM may need multiple tool calls to complete the user's request. Do NOT penalize
    the response for not producing a final answer if the correct next step is a tool call.

    Rate as Good if:
    - The immediate tool call or text response is the logical next step
    - Tool arguments are reasonable and correct
    - The response moves toward fulfilling the user's request

    Rate as Bad if:
    - The wrong tool was called, or with clearly wrong arguments
    - The response is off-topic or nonsensical
    - A text response was given when a tool call was needed (or vice versa)
    """

    prompt_description: str = dspy.InputField(
        desc="Description of the prompt template used to generate part of the input messages."
    )
    input_messages: list[Any] = dspy.InputField(
        desc="The input messages sent to the LLM."
    )
    output: Any = dspy.InputField(
        desc="The immediate text/tool usage response returned by the LLM. This is a SINGLE step — not the final result."
    )
    tools: list[Any] | None = dspy.InputField(
        desc="Tool definitions available to the LLM, if any.", default=None
    )
    output_type: Any | None = dspy.InputField(
        desc="Expected output type/schema configuration, if any.", default=None
    )
    rating: Rating = dspy.OutputField(
        desc="Rate ONLY whether this immediate response is the correct next step. "
        "Do NOT penalize for not completing the overall task if this is an intermediate step in a multi-step workflow."
    )


async def rate_prompt_llm_call(
    rating_input: PromptLlmCallEvalInput,
) -> RatingResult:
    """DSPy chain-of-thought agent to evaluate the quality of an LLM call
    using prompt template context."""
    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini")):
        agent = dspy.ChainOfThought(PromptLlmCallEvalSignature)
        res = await agent.acall(
            prompt_description=rating_input.prompt_description,
            input_messages=rating_input.input_messages,
            output=rating_input.output,
            tools=rating_input.tools,
            output_type=rating_input.output_type,
        )
        return RatingResult(thoughts=res.reasoning, rating=res.rating)
