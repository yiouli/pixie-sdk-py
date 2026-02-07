"""Tests for evaluator optimizer module."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import dspy

from pixie.agents.evaluator_optimizer import (
    EVALUATORS_BASE_DIR,
    _get_evaluator_dir,
    _get_evaluator_filename,
    _record_to_eval_input,
    _record_to_example,
    _to_openai_tool_format,
    delete_optimized_evaluator,
    evaluator_metric,
    fetch_training_data,
    get_latest_optimized_evaluator_path,
    list_optimized_evaluators,
    load_optimized_evaluator,
    optimize_evaluator,
)
from pixie.storage.types import (
    LlmCallRecord,
    PromptInfoRecord,
    RatingDetails,
    ToolDefinition,
)


@pytest.fixture
def sample_llm_call_record() -> LlmCallRecord:
    """Create a sample LLM call record for testing."""
    return LlmCallRecord(
        span_id="span-123",
        trace_id="trace-456",
        run_id="run-789",
        prompt_info=PromptInfoRecord(
            prompt_id="test-prompt",
            version_id="v1",
            variables={"name": "Test"},
        ),
        llm_input={
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Hello!",
                    "tools": [{"name": "search"}],  # Should be removed
                },
                {
                    "role": "assistant",
                    "content": None,
                    # ChatML format tool_calls - should be reformatted
                    "tool_calls": [
                        {
                            "name": "search",
                            "arguments": '{"query": "test"}',
                            "id": "call_123",
                            "type": "function",
                        }
                    ],
                },
            ]
        },
        # llm_output with tool_calls - this is what the evaluator receives as 'tools'
        llm_output={
            "role": "assistant",
            "content": "Hi there!",
            "tool_calls": [
                {
                    "id": "call_456",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "hello"}'},
                }
            ],
        },
        tools=[
            ToolDefinition(
                name="search",
                description="Search the web",
                parameters={"query": {"type": "string"}},
            )
        ],
        output_type={"type": "object", "properties": {"response": {"type": "string"}}},
        model_name="gpt-4o-mini",
        rating=RatingDetails(
            value="good",
            rated_at="1706745600000",
            rated_by="user",
        ),
    )


@pytest.fixture
def sample_llm_call_record_list_input() -> LlmCallRecord:
    """Create a sample LLM call record with list input."""
    return LlmCallRecord(
        span_id="span-list-123",
        trace_id="trace-456",
        prompt_info=PromptInfoRecord(
            prompt_id="test-prompt",
            version_id="v1",
        ),
        llm_input=[
            {"role": "user", "content": "Hello!"},
        ],
        llm_output={"role": "assistant", "content": "Hi!"},
        rating=RatingDetails(
            value="bad",
            rated_at="1706745600000",
            rated_by="system",
        ),
    )


@pytest.fixture
def sample_prompt_registration():
    """Create a mock prompt registration."""
    mock = MagicMock()
    mock.description = "A test prompt for greeting users"
    mock.prompt = MagicMock()
    mock.module = "test_module"
    return mock


class TestRecordTransformation:
    """Tests for record to eval input transformation."""

    def test_record_to_eval_input_with_messages_dict(
        self, sample_llm_call_record: LlmCallRecord
    ):
        """Test transforming record with messages in dict format."""
        result = _record_to_eval_input(sample_llm_call_record, "Test prompt")

        assert result.prompt_description == "Test prompt"
        assert len(result.input_messages) == 3
        assert result.input_messages[0]["role"] == "system"
        assert result.input_messages[1]["role"] == "user"
        assert result.input_messages[2]["role"] == "assistant"

        # Verify 'tools' is removed from messages
        assert "tools" not in result.input_messages[1]

        # Verify tool_calls are reformatted to OpenAI format
        tool_calls = result.input_messages[2].get("tool_calls", [])
        assert len(tool_calls) == 1
        assert "function" in tool_calls[0]
        assert tool_calls[0]["function"]["name"] == "search"
        assert tool_calls[0]["id"] == "call_123"

        # Output is from llm_output
        assert result.output["role"] == "assistant"
        assert result.output["content"] == "Hi there!"

        # tools param comes from record.tools, converted to OpenAI format
        assert result.tools is not None
        assert len(result.tools) == 1
        assert result.tools[0]["type"] == "function"
        assert result.tools[0]["function"]["name"] == "search"
        assert result.tools[0]["function"]["description"] == "Search the web"
        assert result.tools[0]["function"]["parameters"] == {
            "query": {"type": "string"}
        }

        assert result.output_type is not None

    def test_record_to_eval_input_removes_tools_from_messages(self):
        """Test that 'tools' property is removed from each message."""
        record = LlmCallRecord(
            span_id="span-tools",
            trace_id="trace-tools",
            llm_input={
                "messages": [
                    {
                        "role": "user",
                        "content": "test",
                        "tools": [{"name": "func1"}, {"name": "func2"}],
                    }
                ]
            },
            llm_output={"content": "response"},
        )
        result = _record_to_eval_input(record, "Test")

        assert len(result.input_messages) == 1
        assert "tools" not in result.input_messages[0]
        assert result.input_messages[0]["content"] == "test"

    def test_record_to_eval_input_reformats_tool_calls(self):
        """Test that tool_calls are reformatted from ChatML to OpenAI format."""
        record = LlmCallRecord(
            span_id="span-tc",
            trace_id="trace-tc",
            llm_input={
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        # ChatML format: {name, arguments, id, type}
                        "tool_calls": [
                            {
                                "name": "get_weather",
                                "arguments": '{"city": "NYC"}',
                                "id": "tc_001",
                                "type": "function",
                            }
                        ],
                    }
                ]
            },
            llm_output={"content": "done"},
        )
        result = _record_to_eval_input(record, "Test")

        tool_calls = result.input_messages[0].get("tool_calls", [])
        assert len(tool_calls) == 1
        # Should be in OpenAI format: {function: {name, arguments}, id, type}
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[0]["function"]["arguments"] == '{"city": "NYC"}'
        assert tool_calls[0]["id"] == "tc_001"
        assert tool_calls[0]["type"] == "function"

    def test_record_to_eval_input_converts_tools_to_openai_format(self):
        """Test that tools param comes from record.tools converted to OpenAI format."""
        record = LlmCallRecord(
            span_id="span-out-tc",
            trace_id="trace-out-tc",
            llm_input=[{"role": "user", "content": "call a tool"}],
            llm_output={
                "role": "assistant",
                "tool_calls": [{"id": "out_tc_1", "function": {"name": "do_thing"}}],
            },
            # record.tools (definitions) should be converted to OpenAI format
            tools=[
                ToolDefinition(
                    name="other_tool",
                    description="This tool",
                    parameters={"x": {"type": "int"}},
                )
            ],
        )
        result = _record_to_eval_input(record, "Test")

        # tools should be from record.tools, converted to OpenAI format
        assert result.tools is not None
        assert len(result.tools) == 1
        assert result.tools[0]["type"] == "function"
        assert result.tools[0]["function"]["name"] == "other_tool"
        assert result.tools[0]["function"]["description"] == "This tool"
        assert result.tools[0]["function"]["parameters"] == {"x": {"type": "int"}}

    def test_record_to_eval_input_with_list_input(
        self, sample_llm_call_record_list_input: LlmCallRecord
    ):
        """Test transforming record with list input."""
        result = _record_to_eval_input(sample_llm_call_record_list_input, "Test prompt")

        assert len(result.input_messages) == 1
        assert result.input_messages[0]["role"] == "user"
        # No record.tools, so tools should be None
        assert result.tools is None

    def test_record_to_eval_input_with_none_input(self):
        """Test transforming record with None input."""
        record = LlmCallRecord(
            span_id="span-none",
            trace_id="trace-none",
            llm_input=None,
            llm_output=None,
        )
        result = _record_to_eval_input(record, "Test prompt")

        assert result.input_messages == []
        assert result.output is None
        assert result.tools is None


class TestToOpenAIToolFormat:
    """Tests for tool format conversion."""

    def test_convert_tool_definitions_to_openai_format(self):
        """Test converting ToolDefinition objects to OpenAI format."""
        tools = [
            ToolDefinition(
                name="search",
                description="Search the web",
                parameters={"query": {"type": "string"}},
            )
        ]
        result = _to_openai_tool_format(tools)

        assert result is not None
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search"
        assert result[0]["function"]["description"] == "Search the web"
        assert result[0]["function"]["parameters"] == {"query": {"type": "string"}}

    def test_convert_tool_dicts_to_openai_format(self):
        """Test converting tool dicts to OpenAI format."""
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {"city": {"type": "string"}},
            }
        ]
        result = _to_openai_tool_format(tools)

        assert result is not None
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather info"

    def test_convert_tool_without_optional_fields(self):
        """Test converting tool without description or parameters."""
        tools = [ToolDefinition(name="simple_tool")]
        result = _to_openai_tool_format(tools)

        assert result is not None
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "simple_tool"
        assert "description" not in result[0]["function"]
        assert "parameters" not in result[0]["function"]

    def test_convert_empty_list_returns_none(self):
        """Test that empty list returns None."""
        result = _to_openai_tool_format([])
        assert result is None

    def test_convert_none_returns_none(self):
        """Test that None input returns None."""
        result = _to_openai_tool_format(None)
        assert result is None

    def test_convert_multiple_tools(self):
        """Test converting multiple tools."""
        tools = [
            ToolDefinition(name="tool1", description="First tool"),
            ToolDefinition(name="tool2", parameters={"x": {"type": "int"}}),
        ]
        result = _to_openai_tool_format(tools)

        assert result is not None
        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool1"
        assert result[0]["function"]["description"] == "First tool"
        assert result[1]["function"]["name"] == "tool2"
        assert result[1]["function"]["parameters"] == {"x": {"type": "int"}}


class TestRecordToExample:
    """Tests for record to DSPy example transformation."""

    def test_record_to_example(self, sample_llm_call_record: LlmCallRecord):
        """Test transforming record to DSPy example."""
        example = _record_to_example(sample_llm_call_record, "Test prompt")

        assert example is not None
        assert example.prompt_description == "Test prompt"
        assert example.rating == "good"
        assert len(example.input_messages) == 3


class TestEvaluatorPath:
    """Tests for evaluator path utilities."""

    def test_get_evaluator_dir(self):
        """Test getting evaluator directory."""
        dir_path = _get_evaluator_dir("my-prompt")
        assert dir_path == Path(f"{EVALUATORS_BASE_DIR}/my-prompt")

    def test_get_evaluator_filename(self):
        """Test generating evaluator filename."""
        filename = _get_evaluator_filename()
        assert filename.endswith(".json")
        # Should match format YYYYMMDD_HHMMSS.json
        name_part = filename.replace(".json", "")
        assert len(name_part) == 15  # 8 + 1 + 6


class TestEvaluatorMetric:
    """Tests for the evaluator metric function."""

    def test_metric_matching_rating(self):
        """Test metric returns True when ratings match."""
        example = dspy.Example(
            prompt_description="Test",
            input_messages=[],
            output="test",
            rating="good",
            rating_notes=None,
        )
        prediction = MagicMock()
        prediction.rating = "good"

        assert evaluator_metric(example, prediction) is True

    def test_metric_non_matching_rating(self):
        """Test metric returns False when ratings don't match."""
        example = dspy.Example(
            prompt_description="Test",
            input_messages=[],
            output="test",
            rating="good",
            rating_notes=None,
        )
        prediction = MagicMock()
        prediction.rating = "bad"

        assert evaluator_metric(example, prediction) is False


class TestFetchTrainingData:
    """Tests for fetching training data."""

    @pytest.mark.asyncio
    async def test_fetch_training_data_success(self, sample_prompt_registration):
        """Test fetching training data with valid prompt."""
        mock_records = [
            LlmCallRecord(
                span_id="span-1",
                trace_id="trace-1",
                llm_input=[{"role": "user", "content": "Hi"}],
                llm_output={"content": "Hello"},
                rating=RatingDetails(value="good", rated_at="123", rated_by="user"),
            ),
            LlmCallRecord(
                span_id="span-2",
                trace_id="trace-2",
                llm_input=[{"role": "user", "content": "Bad input"}],
                llm_output={"content": "Bad output"},
                rating=RatingDetails(value="bad", rated_at="124", rated_by="system"),
            ),
        ]

        with patch(
            "pixie.agents.evaluator_optimizer.get_prompt",
            return_value=sample_prompt_registration,
        ):
            with patch(
                "pixie.agents.evaluator_optimizer.get_llm_call_records",
                new_callable=AsyncMock,
                return_value=mock_records,
            ) as mock_get_records:
                examples, description = await fetch_training_data("test-prompt")

                assert len(examples) == 2
                assert description == "A test prompt for greeting users"
                # Verify filters were applied correctly
                call_filters = mock_get_records.call_args[0][0]
                assert call_filters.prompt_id == "test-prompt"
                assert call_filters.rating_values == ["good", "bad"]
                assert call_filters.rated_by_values == ["user", "system"]

    @pytest.mark.asyncio
    async def test_fetch_training_data_prompt_not_found(self):
        """Test error when prompt is not found."""
        with patch("pixie.agents.evaluator_optimizer.get_prompt", return_value=None):
            with pytest.raises(ValueError, match="not found in registry"):
                await fetch_training_data("nonexistent-prompt")

    @pytest.mark.asyncio
    async def test_fetch_training_data_no_description(self, sample_prompt_registration):
        """Test error when prompt has no description."""
        sample_prompt_registration.description = None

        with patch(
            "pixie.agents.evaluator_optimizer.get_prompt",
            return_value=sample_prompt_registration,
        ):
            with pytest.raises(ValueError, match="has no description"):
                await fetch_training_data("test-prompt")


class TestListOptimizedEvaluators:
    """Tests for listing optimized evaluators."""

    def test_list_evaluators_empty_dir(self, tmp_path: Path):
        """Test listing when no evaluators exist."""
        with patch(
            "pixie.agents.evaluator_optimizer._get_evaluator_dir",
            return_value=tmp_path / "nonexistent",
        ):
            result = list_optimized_evaluators("test-prompt")
            assert result == []

    def test_list_evaluators_with_files(self, tmp_path: Path):
        """Test listing evaluators with existing files."""
        evaluator_dir = tmp_path / "test-prompt"
        evaluator_dir.mkdir(parents=True)

        # Create test evaluator files
        (evaluator_dir / "20240201_120000.json").write_text("{}")
        (evaluator_dir / "20240202_130000.json").write_text("{}")

        with patch(
            "pixie.agents.evaluator_optimizer._get_evaluator_dir",
            return_value=evaluator_dir,
        ):
            result = list_optimized_evaluators("test-prompt")

            assert len(result) == 2
            # Should be sorted in reverse order (newest first)
            assert "20240202" in result[0]["filename"]
            assert result[0]["timestamp"] is not None


class TestGetLatestEvaluatorPath:
    """Tests for getting latest evaluator path."""

    def test_get_latest_no_evaluators(self, tmp_path: Path):
        """Test when no evaluators exist."""
        with patch(
            "pixie.agents.evaluator_optimizer._get_evaluator_dir",
            return_value=tmp_path / "nonexistent",
        ):
            result = get_latest_optimized_evaluator_path("test-prompt")
            assert result is None

    def test_get_latest_with_evaluators(self, tmp_path: Path):
        """Test getting latest evaluator when multiple exist."""
        evaluator_dir = tmp_path / "test-prompt"
        evaluator_dir.mkdir(parents=True)

        # Create test evaluator files
        (evaluator_dir / "20240201_120000.json").write_text("{}")
        (evaluator_dir / "20240202_130000.json").write_text("{}")

        with patch(
            "pixie.agents.evaluator_optimizer._get_evaluator_dir",
            return_value=evaluator_dir,
        ):
            result = get_latest_optimized_evaluator_path("test-prompt")

            assert result is not None
            assert "20240202_130000" in str(result)


class TestDeleteOptimizedEvaluator:
    """Tests for deleting optimized evaluators."""

    def test_delete_existing_evaluator(self, tmp_path: Path):
        """Test deleting an existing evaluator."""
        evaluator_dir = tmp_path / "test-prompt"
        evaluator_dir.mkdir(parents=True)
        evaluator_file = evaluator_dir / "20240201_120000.json"
        evaluator_file.write_text("{}")

        with patch(
            "pixie.agents.evaluator_optimizer._get_evaluator_dir",
            return_value=evaluator_dir,
        ):
            result = delete_optimized_evaluator("test-prompt", "20240201_120000.json")

            assert result is True
            assert not evaluator_file.exists()

    def test_delete_nonexistent_evaluator(self, tmp_path: Path):
        """Test deleting a non-existent evaluator."""
        evaluator_dir = tmp_path / "test-prompt"
        evaluator_dir.mkdir(parents=True)

        with patch(
            "pixie.agents.evaluator_optimizer._get_evaluator_dir",
            return_value=evaluator_dir,
        ):
            result = delete_optimized_evaluator("test-prompt", "nonexistent.json")

            assert result is False


class TestOptimizeEvaluator:
    """Tests for the main optimization function."""

    @pytest.mark.asyncio
    async def test_optimize_insufficient_data(self, sample_prompt_registration):
        """Test error when insufficient training data."""
        mock_records = [
            LlmCallRecord(
                span_id="span-1",
                trace_id="trace-1",
                llm_input=[{"role": "user", "content": "Hi"}],
                llm_output={"content": "Hello"},
                rating=RatingDetails(value="good", rated_at="123", rated_by="user"),
            ),
        ]

        with patch(
            "pixie.agents.evaluator_optimizer.get_prompt",
            return_value=sample_prompt_registration,
        ):
            with patch(
                "pixie.agents.evaluator_optimizer.get_llm_call_records",
                new_callable=AsyncMock,
                return_value=mock_records,
            ):
                with pytest.raises(ValueError, match="Insufficient training data"):
                    await optimize_evaluator("test-prompt")

    @pytest.mark.asyncio
    async def test_optimize_evaluator_success(
        self, sample_prompt_registration, tmp_path: Path
    ):
        """Test successful evaluator optimization."""
        mock_records = [
            LlmCallRecord(
                span_id=f"span-{i}",
                trace_id=f"trace-{i}",
                llm_input=[{"role": "user", "content": f"Input {i}"}],
                llm_output={"content": f"Output {i}"},
                rating=RatingDetails(
                    value="good" if i % 2 == 0 else "bad",
                    rated_at=str(i),
                    rated_by="user",
                ),
            )
            for i in range(5)
        ]

        evaluator_dir = tmp_path / "test-prompt"

        # Mock all the necessary functions
        with patch(
            "pixie.agents.evaluator_optimizer.get_prompt",
            return_value=sample_prompt_registration,
        ):
            with patch(
                "pixie.agents.evaluator_optimizer.get_llm_call_records",
                new_callable=AsyncMock,
                return_value=mock_records,
            ):
                with patch(
                    "pixie.agents.evaluator_optimizer._get_evaluator_dir",
                    return_value=evaluator_dir,
                ):
                    # Mock DSPy components
                    mock_program = MagicMock(spec=dspy.ChainOfThought)
                    mock_program.save = MagicMock()

                    mock_optimizer = MagicMock()
                    mock_optimizer.compile.return_value = mock_program

                    with patch("dspy.ChainOfThought", return_value=mock_program):
                        with patch(
                            "dspy.BootstrapFewShot", return_value=mock_optimizer
                        ):
                            with patch("dspy.context"):
                                result = await optimize_evaluator("test-prompt")

                                assert result.parent == evaluator_dir
                                assert result.suffix == ".json"
                                mock_optimizer.compile.assert_called_once()
                                mock_program.save.assert_called_once()


class TestLoadOptimizedEvaluator:
    """Tests for loading optimized evaluators."""

    def test_load_no_evaluator_exists(self, tmp_path: Path):
        """Test loading when no evaluator exists."""
        with patch(
            "pixie.agents.evaluator_optimizer.get_latest_optimized_evaluator_path",
            return_value=None,
        ):
            result = load_optimized_evaluator("test-prompt")
            assert result is None

    def test_load_existing_evaluator(self, tmp_path: Path):
        """Test loading an existing evaluator."""
        evaluator_file = tmp_path / "20240201_120000.json"
        evaluator_file.write_text("{}")

        mock_program = MagicMock(spec=dspy.ChainOfThought)

        with patch(
            "pixie.agents.evaluator_optimizer.get_latest_optimized_evaluator_path",
            return_value=evaluator_file,
        ):
            with patch("dspy.ChainOfThought", return_value=mock_program):
                result = load_optimized_evaluator("test-prompt")

                assert result is mock_program
                mock_program.load.assert_called_once_with(str(evaluator_file))
