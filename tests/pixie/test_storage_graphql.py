"""Tests for GraphQL storage layer with timestamp handling.

These tests verify that:
1. Timestamps are properly converted from int (input) to string (output)
2. Large timestamps (nanoseconds) don't overflow
3. GraphQL query operations work correctly
"""

import pytest
import pytest_asyncio
from piccolo.engine.sqlite import SQLiteEngine

from pixie.storage.tables import RunRecord, LlmCallRecord
from pixie.storage.operations import (
    create_run_record as db_create_run_record,
    create_llm_call_record as db_create_llm_call_record,
    get_run_record as db_get_run_record,
    get_llm_call_record as db_get_llm_call_record,
    update_run_record as db_update_run_record,
    update_llm_call_record as db_update_llm_call_record,
    get_llm_call_records as db_get_llm_call_records,
    get_llm_call_records_by_run as db_get_llm_call_records_by_run,
)
from pixie.storage.types import (
    RunRecordInput,
    RunRecordUpdate,
    LlmCallRecordInput,
    LlmCallRecordUpdate,
    AppInfoRecord,
    PromptInfoRecord,
    RatingDetails,
    RecordFilters,
)


@pytest_asyncio.fixture(scope="function")
async def test_db(tmp_path):
    """Set up a test database."""
    import os

    # Create test DB
    test_db_path = str(tmp_path / "test_storage_graphql.db")
    engine = SQLiteEngine(path=test_db_path)

    # Override the engine in tables
    RunRecord._meta.db = engine
    LlmCallRecord._meta.db = engine

    # Start connection and create tables
    await engine.start_connection_pool()
    await RunRecord.create_table(if_not_exists=True)
    await LlmCallRecord.create_table(if_not_exists=True)

    yield engine

    # Cleanup
    await engine.close_connection_pool()
    if os.path.exists(test_db_path):
        os.unlink(test_db_path)


class TestTimestampStringConversion:
    """Test that timestamps are stored as strings and returned as strings."""

    @pytest.mark.asyncio
    async def test_run_record_timestamps_returned_as_strings(self, test_db):
        """Run record timestamps should be returned as strings."""
        run_input = RunRecordInput(
            id="run-ts-001",
            source="apps",
            start_time="1700000000000",  # Input is string (milliseconds)
            messages=[],
            prompt_ids=[],
            metadata={},
        )

        result = await db_create_run_record(run_input)

        # Output should be string
        assert result.start_time == "1700000000000"
        assert isinstance(result.start_time, str)

    @pytest.mark.asyncio
    async def test_llm_call_nanosecond_timestamps(self, test_db):
        """LLM call nanosecond timestamps should not overflow."""
        # Large nanosecond timestamp that would overflow 32-bit int
        large_nano = "1770081192876543210"  # > 2^31, as string

        llm_input = LlmCallRecordInput(
            span_id="span-nano-001",
            trace_id="trace-001",
            run_source="apps",
            start_time=large_nano,
            end_time="1770081193876543210",  # large_nano + 1000000000
            model_name="gpt-4",
            metadata={},
        )

        result = await db_create_llm_call_record(llm_input)

        # Verify large timestamps are preserved as strings
        assert result.start_time == large_nano
        assert result.end_time == "1770081193876543210"
        assert isinstance(result.start_time, str)
        assert isinstance(result.end_time, str)

    @pytest.mark.asyncio
    async def test_rating_timestamp_as_string(self, test_db):
        """Rating timestamps should be stored and returned as strings."""
        run_input = RunRecordInput(
            id="run-rating-001",
            source="apps",
            start_time="1700000000000",
            messages=[],
            prompt_ids=[],
            metadata={},
        )

        await db_create_run_record(run_input)

        # Update with rating (rated_at is already a string in RatingDetails)
        update = RunRecordUpdate(
            rating=RatingDetails(
                value="good",
                rated_at="1700000001000",
                notes="Great!",
            )
        )

        updated = await db_update_run_record("run-rating-001", update)

        assert updated is not None
        assert updated.rating is not None
        assert updated.rating.rated_at == "1700000001000"
        assert isinstance(updated.rating.rated_at, str)

    @pytest.mark.asyncio
    async def test_retrieved_timestamps_are_strings(self, test_db):
        """Retrieved records should have string timestamps."""
        run_input = RunRecordInput(
            id="run-retrieve-001",
            source="sessions",
            start_time="1700000000000",
            messages=[],
            prompt_ids=[],
            metadata={},
        )

        await db_create_run_record(run_input)

        # Retrieve and verify
        result = await db_get_run_record("run-retrieve-001")

        assert result is not None
        assert isinstance(result.start_time, str)
        assert result.start_time == "1700000000000"


class TestGraphQLCRUDOperations:
    """Test GraphQL CRUD operations for storage records."""

    @pytest.mark.asyncio
    async def test_create_and_get_run_record(self, test_db):
        """Should create and retrieve run record."""
        app_info = AppInfoRecord(
            id="app-1",
            name="Test App",
            module="test",
            qualified_name="test.App",
        )

        run_input = RunRecordInput(
            id="run-crud-001",
            source="apps",
            app_info=app_info,
            start_time="1700000000000",
            messages=[],
            prompt_ids=["prompt-1", "prompt-2"],
            metadata={"key": "value"},
        )

        # Create
        created = await db_create_run_record(run_input)
        assert created.id == "run-crud-001"
        assert created.prompt_ids == ["prompt-1", "prompt-2"]

        # Get
        retrieved = await db_get_run_record("run-crud-001")
        assert retrieved is not None
        assert retrieved.id == "run-crud-001"
        assert retrieved.prompt_ids == ["prompt-1", "prompt-2"]
        assert retrieved.app_info is not None
        assert retrieved.app_info.id == "app-1"

    @pytest.mark.asyncio
    async def test_update_run_record_partial(self, test_db):
        """Should allow partial updates to run record."""
        run_input = RunRecordInput(
            id="run-update-001",
            source="apps",
            start_time="1700000000000",
            messages=[],
            prompt_ids=[],
            metadata={},
        )

        await db_create_run_record(run_input)

        # Partial update - only end_time
        update = RunRecordUpdate(end_time="1700000100000")
        updated = await db_update_run_record("run-update-001", update)

        assert updated is not None
        # Both should be strings in output
        assert updated.start_time == "1700000000000"
        assert updated.end_time == "1700000100000"

    @pytest.mark.asyncio
    async def test_create_and_get_llm_call_record(self, test_db):
        """Should create and retrieve LLM call record."""
        prompt_info = PromptInfoRecord(
            prompt_id="prompt-1",
            version_id="v1",
            variables={"name": "test"},
        )

        llm_input = LlmCallRecordInput(
            span_id="span-crud-001",
            trace_id="trace-001",
            run_id="run-001",
            run_source="apps",
            prompt_info=prompt_info,
            start_time="1700000000000000",  # nanoseconds
            end_time="1700000001000000",
            model_name="gpt-4",
            metadata={},
        )

        # Create
        created = await db_create_llm_call_record(llm_input)
        assert created.span_id == "span-crud-001"
        assert created.model_name == "gpt-4"

        # Get
        retrieved = await db_get_llm_call_record("span-crud-001")
        assert retrieved is not None
        assert retrieved.span_id == "span-crud-001"
        assert retrieved.start_time == "1700000000000000"
        assert retrieved.prompt_info is not None
        assert retrieved.prompt_info.prompt_id == "prompt-1"

    @pytest.mark.asyncio
    async def test_filter_llm_calls_by_prompt(self, test_db):
        """Should filter LLM calls by prompt ID."""
        # Create multiple LLM calls with different prompts
        for i in range(3):
            llm_input = LlmCallRecordInput(
                span_id=f"span-filter-{i}",
                trace_id=f"trace-filter-{i}",
                run_source="apps",
                prompt_info=PromptInfoRecord(
                    prompt_id=f"prompt-{i % 2}",  # alternating prompts
                    version_id="v1",
                ),
                model_name="gpt-4",
                start_time="1700000000000000",
                metadata={},
            )
            await db_create_llm_call_record(llm_input)

        # Filter by prompt-0
        filters = RecordFilters(prompt_id="prompt-0")
        results = await db_get_llm_call_records(filters)

        # Should get 2 results (span-filter-0 and span-filter-2)
        assert len(results) == 2
        assert all(
            r.prompt_info is not None and r.prompt_info.prompt_id == "prompt-0"
            for r in results
        )

    @pytest.mark.asyncio
    async def test_filter_records_by_rating(self, test_db):
        """Should filter records by rating value."""
        # Create LLM calls with different ratings
        ratings = ["good", "bad", None]
        for i, rating_value in enumerate(ratings):
            llm_input = LlmCallRecordInput(
                span_id=f"span-rate-{i}",
                trace_id=f"trace-rate-{i}",
                run_source="apps",
                model_name="gpt-4",
                start_time="1700000000000000",
                metadata={},
            )
            created = await db_create_llm_call_record(llm_input)

            # Add rating if not None
            if rating_value:
                update = LlmCallRecordUpdate(
                    rating=RatingDetails(
                        value=rating_value,  # type: ignore[arg-type]
                        rated_at="1700000001000",
                    )
                )
                await db_update_llm_call_record(created.span_id, update)

        # Filter by rating value
        filters = RecordFilters(rating_value="good")
        results = await db_get_llm_call_records(filters)

        assert len(results) >= 1
        assert all(r.rating is not None and r.rating.value == "good" for r in results)

    @pytest.mark.asyncio
    async def test_get_llm_calls_by_run(self, test_db):
        """Should get all LLM calls for a specific run."""
        run_id = "run-with-calls"

        # Create multiple LLM calls for the same run
        for i in range(3):
            llm_input = LlmCallRecordInput(
                span_id=f"span-byrun-{i}",
                trace_id=f"trace-byrun-{i}",
                run_id=run_id,
                run_source="apps",
                model_name="gpt-4",
                start_time="1700000000000000",
                metadata={},
            )
            await db_create_llm_call_record(llm_input)

        # Get all calls for the run
        results = await db_get_llm_call_records_by_run(run_id)

        assert len(results) == 3
        assert all(r.run_id == run_id for r in results)


class TestTimestampPrecisionPreservation:
    """Test that timestamp precision is maintained through the stack."""

    @pytest.mark.asyncio
    async def test_nanosecond_precision_preserved(self, test_db):
        """Should preserve full nanosecond precision."""
        # Nanosecond timestamp with full precision
        precise_nano = "1770081192876543210"

        llm_input = LlmCallRecordInput(
            span_id="span-precise-nano",
            trace_id="trace-precise",
            run_source="apps",
            start_time=precise_nano,
            model_name="gpt-4",
            metadata={},
        )

        created = await db_create_llm_call_record(llm_input)

        # Verify exact match (as string)
        assert created.start_time == precise_nano

    @pytest.mark.asyncio
    async def test_millisecond_precision_preserved(self, test_db):
        """Should preserve millisecond precision for run records."""
        precise_milli = "1700000000123"

        run_input = RunRecordInput(
            id="run-precise-milli",
            source="apps",
            start_time=precise_milli,
            messages=[],
            prompt_ids=[],
            metadata={},
        )

        created = await db_create_run_record(run_input)

        # Verify exact match (as string)
        assert created.start_time == precise_milli

    @pytest.mark.asyncio
    async def test_no_precision_loss_on_retrieval(self, test_db):
        """Should not lose precision when retrieving records."""
        # Use a timestamp that would lose precision as float64
        problematic = "1770081192876543999"

        llm_input = LlmCallRecordInput(
            span_id="span-no-loss",
            trace_id="trace-no-loss",
            run_source="apps",
            start_time=problematic,
            model_name="gpt-4",
            metadata={},
        )

        await db_create_llm_call_record(llm_input)

        # Retrieve and verify exact match
        retrieved = await db_get_llm_call_record("span-no-loss")

        assert retrieved is not None
        assert retrieved.start_time == problematic
        assert isinstance(retrieved.start_time, str)
