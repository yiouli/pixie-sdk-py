"""GraphQL schema elements for agent rating and optimization."""

import asyncio
import logging
from enum import Enum
from typing import AsyncGenerator, Optional

import strawberry
import strawberry.experimental.pydantic
from strawberry.scalars import JSON

from pixie.agents.optimizers import (
    list_optimized_evaluators as get_optimized_evaluators_list,
    get_latest_optimized_evaluator_path,
)
from pixie.agents.evaluators import (
    RatingResult as PydanticRatingResult,
    LlmCallRatingAgentInput as PydanticLlmCallRatingInput,
    AppRunRatingAgentInput as PydanticAppRunRatingInput,
    FindBadResponseInputSignature as PydanticFindBadResponseInput,
    FindBadLlmCallInput as PydanticFindBadLlmCallInput,
    LlmCallSpan as PydanticLlmCallSpan,
    PromptLlmCallEvalInput as PydanticPromptLlmCallEvalInput,
    rate_llm_call as execute_rate_llm_call,
    rate_app_run as execute_rate_app_run,
    find_bad_response as execute_find_bad_response,
    find_bad_llm_call as execute_find_bad_llm_call,
    rate_prompt_llm_call as execute_rate_prompt_llm_call,
)
from pixie.prompts.graphql import LlmCallInput, LlmCallResult, execute_single_llm_call
from pixie.strawberry_types import MessageInput, Rating

logger = logging.getLogger(__name__)

# Batch size for concurrent LLM calls
BATCH_LLM_CONCURRENCY = 5


@strawberry.enum
class BatchLlmCallStatus(str, Enum):
    """Status of a batch LLM call item."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    RATING = "RATING"
    """Rating is in progress for this call."""
    RATED = "RATED"
    """Rating has completed for this call."""


@strawberry.type
class BatchLlmCallRating:
    """Rating result for a batch LLM call."""

    rating: Rating
    thoughts: str


@strawberry.type
class BatchLlmCallUpdate:
    """Status update for a batch LLM call."""

    id: strawberry.ID
    """The id of the LLM call this update is for."""
    status: BatchLlmCallStatus
    result: Optional[LlmCallResult] = None
    error: Optional[str] = None
    rating_result: Optional[BatchLlmCallRating] = None
    """Rating result, present when status is RATED."""


@strawberry.type
class BatchLlmCallsUpdate:
    """Batched status updates for multiple LLM calls. Sent as a single message."""

    updates: list[BatchLlmCallUpdate]


@strawberry.type
class OptimizedEvaluatorInfo:
    """Information about an optimized evaluator version."""

    path: str
    """Full path to the evaluator file."""
    filename: str
    """Filename of the evaluator."""
    timestamp: Optional[str] = None
    """ISO timestamp when the evaluator was created."""


@strawberry.type
class OptimizeEvaluatorResult:
    """Result of evaluator optimization."""

    success: bool
    """Whether optimization was successful."""
    path: Optional[str] = None
    """Path to the saved optimized evaluator file."""
    error: Optional[str] = None
    """Error message if optimization failed."""


@strawberry.experimental.pydantic.type(model=PydanticRatingResult)
class RatingResult:
    """Result of a rating operation."""

    thoughts: strawberry.auto
    rating: Rating


@strawberry.experimental.pydantic.input(model=PydanticLlmCallSpan)
class LlmCallSpanInput:
    """LLM call span input for trace analysis."""

    span_type: strawberry.auto
    llm_input: JSON
    llm_output: JSON


@strawberry.type
class FindBadResponseResult:
    """Result of finding a bad response in a conversation."""

    bad_response_index: int
    """Index of the problematic assistant message in the conversation."""
    thoughts: str
    """Reasoning behind identifying this message as the bad response."""


@strawberry.type
class FindBadLlmCallResult:
    """Result of finding a bad LLM call span in a trace."""

    bad_span_index: int
    """Index of the problematic LLM call span in the trace."""
    thoughts: str
    """Reasoning behind identifying this span as the bad LLM call."""


@strawberry.type
class AgentMutation:
    """GraphQL mutations for agent ratings and optimization."""

    @strawberry.mutation
    async def rate_llm_call(
        self,
        app_description: str,
        interaction_logs_before_llm_call: list[MessageInput],
        llm_input: JSON,
        llm_output: JSON,
        llm_configuration: JSON,
        internal_logs_after_llm_call: list[JSON],
        interaction_logs_after_llm_call: list[MessageInput],
    ) -> RatingResult:
        """Rate the quality of a specific LLM call within an application execution."""

        messages_before = [
            msg.to_pydantic() for msg in interaction_logs_before_llm_call
        ]
        messages_after = [msg.to_pydantic() for msg in interaction_logs_after_llm_call]

        rating_input = PydanticLlmCallRatingInput(
            app_description=app_description,
            interaction_logs_before_llm_call=messages_before,
            llm_input=llm_input,
            llm_output=llm_output,
            llm_configuration=llm_configuration,
            internal_logs_after_llm_call=internal_logs_after_llm_call,
            interaction_logs_after_llm_call=messages_after,
        )

        result = await execute_rate_llm_call(rating_input)
        return RatingResult.from_pydantic(result)

    @strawberry.mutation
    async def rate_run(
        self,
        run_description: str,
        interaction_logs: list[MessageInput],
    ) -> RatingResult:
        """Rate the overall quality of an app/session run."""

        messages = [msg.to_pydantic() for msg in interaction_logs]

        rating_input = PydanticAppRunRatingInput(
            app_description=run_description,
            interaction_logs=messages,
        )

        result = await execute_rate_app_run(rating_input)
        return RatingResult.from_pydantic(result)

    @strawberry.mutation
    async def find_bad_response(
        self,
        ai_description: str,
        conversation: list[MessageInput],
        reasoning_for_negative_rating: str,
    ) -> FindBadResponseResult:
        """Find the problematic assistant message in a conversation."""

        messages = [msg.to_pydantic() for msg in conversation]

        find_input = PydanticFindBadResponseInput(
            ai_description=ai_description,
            conversation=messages,
            reasoning_for_negative_rating=reasoning_for_negative_rating,
        )

        result = await execute_find_bad_response(find_input)
        return FindBadResponseResult(
            bad_response_index=result.bad_index,
            thoughts=result.thoughts,
        )

    @strawberry.mutation
    async def find_bad_llm_call(
        self,
        ai_description: str,
        conversation: list[MessageInput],
        trace: list[JSON],
        reasoning_for_negative_rating: str,
    ) -> FindBadLlmCallResult:
        """Find the problematic LLM call span in a trace that led to a bad response."""

        messages = [msg.to_pydantic() for msg in conversation]
        trace_spans: list[PydanticLlmCallSpan | dict] = [
            dict(span) for span in trace  # type: ignore[arg-type]
        ]

        find_input = PydanticFindBadLlmCallInput(
            ai_description=ai_description,
            conversation=messages,
            trace=trace_spans,
            reasoning_for_negative_rating=reasoning_for_negative_rating,
        )

        result = await execute_find_bad_llm_call(find_input)
        return FindBadLlmCallResult(
            bad_span_index=result.bad_index,
            thoughts=result.thoughts,
        )

    @strawberry.mutation
    async def rate_prompt_llm_call(
        self,
        prompt_description: str,
        input_messages: list[JSON],
        output: JSON,
        tools: list[JSON] | None = None,
        output_type: JSON | None = None,
    ) -> RatingResult:
        """Evaluate the quality of an LLM call using prompt template context."""

        eval_input = PydanticPromptLlmCallEvalInput(
            prompt_description=prompt_description,
            input_messages=input_messages,
            output=output,
            tools=tools,
            output_type=output_type,
        )

        result = await execute_rate_prompt_llm_call(eval_input)
        return RatingResult.from_pydantic(result)

    @strawberry.mutation
    async def get_optimized_evaluators(
        self,
        prompt_id: str,
    ) -> list[OptimizedEvaluatorInfo]:
        """List all optimized evaluator versions for a prompt."""

        evaluators = get_optimized_evaluators_list(prompt_id)
        return [
            OptimizedEvaluatorInfo(
                path=e["path"],
                filename=e["filename"],
                timestamp=e.get("timestamp"),
            )
            for e in evaluators
        ]

    @strawberry.mutation
    async def get_latest_evaluator_path(
        self,
        prompt_id: str,
    ) -> Optional[str]:
        """Get the path to the latest optimized evaluator for a prompt."""

        path = get_latest_optimized_evaluator_path(prompt_id)
        return str(path) if path else None


@strawberry.type
class AgentSubscription:
    """GraphQL subscriptions for agent operations."""

    @strawberry.subscription
    async def batch_call_llm(
        self,
        calls: list[LlmCallInput],
        prompt_description: Optional[str] = None,
    ) -> AsyncGenerator[BatchLlmCallsUpdate, None]:
        """Execute multiple LLM calls and stream batched status updates."""

        if not calls:
            return

        pending_updates = [
            BatchLlmCallUpdate(
                id=call.id,
                status=BatchLlmCallStatus.PENDING,
                result=None,
                error=None,
                rating_result=None,
            )
            for call in calls
        ]
        yield BatchLlmCallsUpdate(updates=pending_updates)

        update_queue: asyncio.Queue[BatchLlmCallUpdate | None] = asyncio.Queue()

        async def process_single_call(call: LlmCallInput) -> None:
            call_id = call.id
            result: LlmCallResult | None = None

            await update_queue.put(
                BatchLlmCallUpdate(
                    id=call_id,
                    status=BatchLlmCallStatus.RUNNING,
                    result=None,
                    error=None,
                    rating_result=None,
                )
            )

            try:
                result = await execute_single_llm_call(call)
                await update_queue.put(
                    BatchLlmCallUpdate(
                        id=call_id,
                        status=BatchLlmCallStatus.COMPLETED,
                        result=result,
                        error=None,
                        rating_result=None,
                    )
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Error executing LLM call id=%s: %s", call_id, str(exc))
                await update_queue.put(
                    BatchLlmCallUpdate(
                        id=call_id,
                        status=BatchLlmCallStatus.ERROR,
                        result=None,
                        error=str(exc),
                        rating_result=None,
                    )
                )
                return

            if result:
                await update_queue.put(
                    BatchLlmCallUpdate(
                        id=call_id,
                        status=BatchLlmCallStatus.RATING,
                        result=result,
                        error=None,
                        rating_result=None,
                    )
                )

                try:
                    rating_input = PydanticPromptLlmCallEvalInput(
                        prompt_description=prompt_description
                        or "No description provided",
                        input_messages=result.input,  # type: ignore[arg-type]
                        output=result.output,
                        tools=result.tool_calls,
                        output_type=call.output_schema,
                    )

                    rating_result = await execute_rate_prompt_llm_call(rating_input)

                    rating_enum = Rating(rating_result.rating)

                    await update_queue.put(
                        BatchLlmCallUpdate(
                            id=call_id,
                            status=BatchLlmCallStatus.RATED,
                            result=result,
                            error=None,
                            rating_result=BatchLlmCallRating(
                                rating=rating_enum,
                                thoughts=rating_result.thoughts,
                            ),
                        )
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    logger.error("Error rating LLM call id=%s: %s", call_id, str(exc))
                    await update_queue.put(
                        BatchLlmCallUpdate(
                            id=call_id,
                            status=BatchLlmCallStatus.ERROR,
                            result=result,
                            error=f"Rating failed: {str(exc)}",
                            rating_result=None,
                        )
                    )

        async def producer(call_list: list[LlmCallInput]) -> None:
            semaphore = asyncio.Semaphore(BATCH_LLM_CONCURRENCY)

            async def limited_process(call: LlmCallInput) -> None:
                async with semaphore:
                    await process_single_call(call)

            await asyncio.gather(
                *[limited_process(call) for call in call_list],
                return_exceptions=True,
            )
            await update_queue.put(None)

        producer_task = asyncio.create_task(producer(calls))

        try:
            while True:
                update = await update_queue.get()
                if update is None:
                    break
                yield BatchLlmCallsUpdate(updates=[update])
        finally:
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass
