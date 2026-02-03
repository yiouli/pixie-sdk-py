"""Test rating agent GraphQL mutations."""

import pytest
from unittest.mock import AsyncMock, patch
from strawberry.schema import Schema

from pixie.schema import Query, Mutation, Subscription, MessageInput, Rating
from pixie.agents.rating_agent import (
    RatingResult as PydanticRatingResult,
    LlmCallRatingAgentInput,
    AppRunRatingAgentInput,
    Message as PydanticMessage,
)


@pytest.fixture
def schema():
    """Create GraphQL schema for testing."""
    return Schema(query=Query, mutation=Mutation, subscription=Subscription)


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        MessageInput(
            role="user",
            content={"text": "Hello"},
            user_rating=None,
            user_feedback=None,
        ),
        MessageInput(
            role="assistant",
            content={"text": "Hi there!"},
            user_rating=Rating.good,
            user_feedback="Great response",
        ),
    ]


@pytest.fixture
def sample_llm_config():
    """Sample LLM configuration."""
    return {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 1000,
    }


class TestRateLlmCallMutation:
    """Test rate_llm_call mutation."""

    @pytest.mark.asyncio
    async def test_rate_llm_call_success(
        self, schema, sample_messages, sample_llm_config
    ):
        """Test successful LLM call rating."""
        mutation = """
            mutation RateLlmCall(
                $appDescription: String!,
                $interactionLogsBefore: [MessageInput!]!,
                $llmInput: JSON!,
                $llmOutput: JSON!,
                $llmConfig: JSON!,
                $internalLogsAfter: [JSON!]!,
                $interactionLogsAfter: [MessageInput!]!
            ) {
                rateLlmCall(
                    appDescription: $appDescription,
                    interactionLogsBeforeLlmCall: $interactionLogsBefore,
                    llmInput: $llmInput,
                    llmOutput: $llmOutput,
                    llmConfiguration: $llmConfig,
                    internalLogsAfterLlmCall: $internalLogsAfter,
                    interactionLogsAfterLlmCall: $interactionLogsAfter
                ) {
                    thoughts
                    rating
                }
            }
        """

        variables = {
            "appDescription": "A helpful chatbot",
            "interactionLogsBefore": [
                {
                    "role": "user",
                    "content": {"text": "What's the weather?"},
                    "userRating": None,
                    "userFeedback": None,
                }
            ],
            "llmInput": {
                "messages": [{"role": "user", "content": "What's the weather?"}]
            },
            "llmOutput": {"response": "It's sunny today."},
            "llmConfig": sample_llm_config,
            "internalLogsAfter": [{"log": "LLM call completed"}],
            "interactionLogsAfter": [
                {
                    "role": "assistant",
                    "content": {"text": "It's sunny today."},
                    "userRating": "good",
                    "userFeedback": None,
                }
            ],
        }

        # Mock the rating agent function
        expected_result = PydanticRatingResult(
            thoughts="The LLM response is accurate and helpful.",
            rating="good",
        )

        with patch(
            "pixie.schema.execute_rate_llm_call",
            new_callable=AsyncMock,
            return_value=expected_result,
        ) as mock_rate:
            result = await schema.execute(mutation, variable_values=variables)

            # Check no errors
            assert result.errors is None
            assert result.data is not None

            # Check result structure
            assert result.data["rateLlmCall"]["thoughts"] == expected_result.thoughts
            assert result.data["rateLlmCall"]["rating"] == "good"

            # Verify the function was called
            assert mock_rate.called
            call_args = mock_rate.call_args[0][0]
            assert isinstance(call_args, LlmCallRatingAgentInput)
            assert call_args.app_description == "A helpful chatbot"

    @pytest.mark.asyncio
    async def test_rate_llm_call_with_bad_rating(self, schema, sample_llm_config):
        """Test LLM call rating with bad result."""
        mutation = """
            mutation RateLlmCall(
                $appDescription: String!,
                $interactionLogsBefore: [MessageInput!]!,
                $llmInput: JSON!,
                $llmOutput: JSON!,
                $llmConfig: JSON!,
                $internalLogsAfter: [JSON!]!,
                $interactionLogsAfter: [MessageInput!]!
            ) {
                rateLlmCall(
                    appDescription: $appDescription,
                    interactionLogsBeforeLlmCall: $interactionLogsBefore,
                    llmInput: $llmInput,
                    llmOutput: $llmOutput,
                    llmConfiguration: $llmConfig,
                    internalLogsAfterLlmCall: $internalLogsAfter,
                    interactionLogsAfterLlmCall: $interactionLogsAfter
                ) {
                    thoughts
                    rating
                }
            }
        """

        variables = {
            "appDescription": "A helpful chatbot",
            "interactionLogsBefore": [
                {
                    "role": "user",
                    "content": {"text": "Calculate 2+2"},
                    "userRating": None,
                    "userFeedback": None,
                }
            ],
            "llmInput": {"messages": [{"role": "user", "content": "Calculate 2+2"}]},
            "llmOutput": {"response": "The answer is 5."},
            "llmConfig": sample_llm_config,
            "internalLogsAfter": [],
            "interactionLogsAfter": [
                {
                    "role": "assistant",
                    "content": {"text": "The answer is 5."},
                    "userRating": "bad",
                    "userFeedback": "Incorrect calculation",
                }
            ],
        }

        expected_result = PydanticRatingResult(
            thoughts="The LLM provided an incorrect mathematical answer.",
            rating="bad",
        )

        with patch(
            "pixie.schema.execute_rate_llm_call",
            new_callable=AsyncMock,
            return_value=expected_result,
        ):
            result = await schema.execute(mutation, variable_values=variables)

            assert result.errors is None
            assert result.data["rateLlmCall"]["rating"] == "bad"

    @pytest.mark.asyncio
    async def test_rate_llm_call_with_user_feedback(self, schema, sample_llm_config):
        """Test LLM call rating with user feedback in messages."""
        mutation = """
            mutation RateLlmCall(
                $appDescription: String!,
                $interactionLogsBefore: [MessageInput!]!,
                $llmInput: JSON!,
                $llmOutput: JSON!,
                $llmConfig: JSON!,
                $internalLogsAfter: [JSON!]!,
                $interactionLogsAfter: [MessageInput!]!
            ) {
                rateLlmCall(
                    appDescription: $appDescription,
                    interactionLogsBeforeLlmCall: $interactionLogsBefore,
                    llmInput: $llmInput,
                    llmOutput: $llmOutput,
                    llmConfiguration: $llmConfig,
                    internalLogsAfterLlmCall: $internalLogsAfter,
                    interactionLogsAfterLlmCall: $interactionLogsAfter
                ) {
                    thoughts
                    rating
                }
            }
        """

        variables = {
            "appDescription": "Customer support bot",
            "interactionLogsBefore": [
                {
                    "role": "user",
                    "content": {"text": "I need help with my order"},
                    "userRating": None,
                    "userFeedback": None,
                }
            ],
            "llmInput": {
                "messages": [{"role": "user", "content": "I need help with my order"}]
            },
            "llmOutput": {"response": "I'll help you with that."},
            "llmConfig": sample_llm_config,
            "internalLogsAfter": [{"retrieved_order": "12345"}],
            "interactionLogsAfter": [
                {
                    "role": "assistant",
                    "content": {"text": "I'll help you with that."},
                    "userRating": "good",
                    "userFeedback": "Very helpful and quick response",
                }
            ],
        }

        expected_result = PydanticRatingResult(
            thoughts="The response was helpful according to user feedback.",
            rating="good",
        )

        with patch(
            "pixie.schema.execute_rate_llm_call",
            new_callable=AsyncMock,
            return_value=expected_result,
        ) as mock_rate:
            result = await schema.execute(mutation, variable_values=variables)

            assert result.errors is None
            assert result.data["rateLlmCall"]["rating"] == "good"

            # Verify user feedback was passed through
            call_args = mock_rate.call_args[0][0]
            after_messages = call_args.interaction_logs_after_llm_call
            assert len(after_messages) == 1
            assert after_messages[0].user_feedback == "Very helpful and quick response"


class TestRateRunMutation:
    """Test rate_app_run mutation."""

    @pytest.mark.asyncio
    async def test_rate_app_run_success(self, schema, sample_messages):
        """Test successful app run rating."""
        mutation = """
            mutation RateRun(
                $runDescription: String!,
                $interactionLogs: [MessageInput!]!
            ) {
                rateRun(
                    runDescription: $runDescription,
                    interactionLogs: $interactionLogs
                ) {
                    thoughts
                    rating
                }
            }
        """

        variables = {
            "runDescription": "A Q&A assistant",
            "interactionLogs": [
                {
                    "role": "user",
                    "content": {"text": "What is Python?"},
                    "userRating": None,
                    "userFeedback": None,
                },
                {
                    "role": "assistant",
                    "content": {"text": "Python is a high-level programming language."},
                    "userRating": "good",
                    "userFeedback": None,
                },
                {
                    "role": "user",
                    "content": {"text": "Thanks!"},
                    "userRating": None,
                    "userFeedback": None,
                },
            ],
        }

        expected_result = PydanticRatingResult(
            thoughts="The application provided accurate and helpful information.",
            rating="good",
        )

        with patch(
            "pixie.schema.execute_rate_app_run",
            new_callable=AsyncMock,
            return_value=expected_result,
        ) as mock_rate:
            result = await schema.execute(mutation, variable_values=variables)

            # Check no errors
            assert result.errors is None
            assert result.data is not None

            # Check result structure
            assert result.data["rateRun"]["thoughts"] == expected_result.thoughts
            assert result.data["rateRun"]["rating"] == "good"

            # Verify the function was called with correct args
            assert mock_rate.called
            call_args = mock_rate.call_args[0][0]
            assert isinstance(call_args, AppRunRatingAgentInput)
            assert call_args.app_description == "A Q&A assistant"
            assert len(call_args.interaction_logs) == 3

    @pytest.mark.asyncio
    async def test_rate_app_run_with_mixed_ratings(self, schema):
        """Test app run rating with mixed user ratings."""
        mutation = """
            mutation RateRun(
                $runDescription: String!,
                $interactionLogs: [MessageInput!]!
            ) {
                rateRun(
                    runDescription: $runDescription,
                    interactionLogs: $interactionLogs
                ) {
                    thoughts
                    rating
                }
            }
        """

        variables = {
            "runDescription": "Multi-turn conversation assistant",
            "interactionLogs": [
                {
                    "role": "user",
                    "content": {"text": "First question"},
                    "userRating": None,
                    "userFeedback": None,
                },
                {
                    "role": "assistant",
                    "content": {"text": "Good answer"},
                    "userRating": "good",
                    "userFeedback": "Helpful",
                },
                {
                    "role": "user",
                    "content": {"text": "Second question"},
                    "userRating": None,
                    "userFeedback": None,
                },
                {
                    "role": "assistant",
                    "content": {"text": "Bad answer"},
                    "userRating": "bad",
                    "userFeedback": "Not accurate",
                },
            ],
        }

        expected_result = PydanticRatingResult(
            thoughts="The application had mixed performance with both good and bad responses.",
            rating="undecided",
        )

        with patch(
            "pixie.schema.execute_rate_app_run",
            new_callable=AsyncMock,
            return_value=expected_result,
        ):
            result = await schema.execute(mutation, variable_values=variables)

            assert result.errors is None
            assert result.data["rateRun"]["rating"] == "undecided"

    @pytest.mark.asyncio
    async def test_rate_app_run_empty_logs(self, schema):
        """Test app run rating with empty interaction logs."""
        mutation = """
            mutation RateRun(
                $runDescription: String!,
                $interactionLogs: [MessageInput!]!
            ) {
                rateRun(
                    runDescription: $runDescription,
                    interactionLogs: $interactionLogs
                ) {
                    thoughts
                    rating
                }
            }
        """

        variables = {
            "runDescription": "Test app",
            "interactionLogs": [],
        }

        expected_result = PydanticRatingResult(
            thoughts="No interactions to evaluate.",
            rating="undecided",
        )

        with patch(
            "pixie.schema.execute_rate_app_run",
            new_callable=AsyncMock,
            return_value=expected_result,
        ):
            result = await schema.execute(mutation, variable_values=variables)

            assert result.errors is None
            assert result.data["rateRun"]["rating"] == "undecided"

    @pytest.mark.asyncio
    async def test_rate_app_run_message_conversion(self, schema):
        """Test that MessageInput is properly converted to pydantic Message."""
        mutation = """
            mutation RateRun(
                $runDescription: String!,
                $interactionLogs: [MessageInput!]!
            ) {
                rateRun(
                    runDescription: $runDescription,
                    interactionLogs: $interactionLogs
                ) {
                    thoughts
                    rating
                }
            }
        """

        variables = {
            "runDescription": "Test conversion",
            "interactionLogs": [
                {
                    "role": "user",
                    "content": {"text": "test"},
                    "userRating": "good",
                    "userFeedback": "test feedback",
                }
            ],
        }

        expected_result = PydanticRatingResult(
            thoughts="Test",
            rating="good",
        )

        with patch(
            "pixie.schema.execute_rate_app_run",
            new_callable=AsyncMock,
            return_value=expected_result,
        ) as mock_rate:
            result = await schema.execute(mutation, variable_values=variables)

            assert result.errors is None

            # Verify the conversion
            call_args = mock_rate.call_args[0][0]
            messages = call_args.interaction_logs
            assert len(messages) == 1
            assert isinstance(messages[0], PydanticMessage)
            assert messages[0].role == "user"
            assert messages[0].content == {"text": "test"}
            assert messages[0].user_rating == "good"
            assert messages[0].user_feedback == "test feedback"


class TestRatingEnumConversion:
    """Test Rating enum handling in mutations."""

    @pytest.mark.asyncio
    async def test_rating_enum_values(self, schema):
        """Test all rating enum values are properly handled."""
        mutation = """
            mutation RateRun(
                $runDescription: String!,
                $interactionLogs: [MessageInput!]!
            ) {
                rateRun(
                    runDescription: $runDescription,
                    interactionLogs: $interactionLogs
                ) {
                    thoughts
                    rating
                }
            }
        """

        # Test each rating value
        for rating_value, expected_graphql in [
            ("good", "good"),
            ("bad", "bad"),
            ("undecided", "undecided"),
        ]:
            variables = {
                "runDescription": "Test",
                "interactionLogs": [
                    {
                        "role": "user",
                        "content": {"text": "test"},
                        "userRating": expected_graphql,
                        "userFeedback": None,
                    }
                ],
            }

            expected_result = PydanticRatingResult(
                thoughts=f"Test {rating_value}",
                rating=rating_value,  # type: ignore
            )

            with patch(
                "pixie.schema.execute_rate_app_run",
                new_callable=AsyncMock,
                return_value=expected_result,
            ):
                result = await schema.execute(mutation, variable_values=variables)

                assert result.errors is None
                assert result.data["rateRun"]["rating"] == expected_graphql
