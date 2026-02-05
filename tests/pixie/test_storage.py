"""Tests for pixie.storage module using TDD approach."""

import json
import os
import pytest
import pytest_asyncio

from pixie.storage.types import (
    Message,
    RatingDetails,
    AppInfoRecord,
    SessionInfoRecord,
    PromptInfoRecord,
    RunRecordInput,
    RunRecordUpdate,
    LlmCallRecordInput,
    LlmCallRecordUpdate,
    RecordFilters,
)
from pixie.storage.operations import (
    _serialize_json,
    _parse_json,
    _parse_messages,
    _parse_rating,
    _parse_prompt_ids,
    create_run_record,
    get_run_record,
    update_run_record,
    get_run_records,
    create_llm_call_record,
    get_llm_call_record,
    update_llm_call_record,
    get_llm_call_records,
    get_llm_call_records_by_run,
)

# Configure test database - import tables AFTER changing the DB path
TEST_DB_PATH = "test_pixie.db"


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_app_info() -> AppInfoRecord:
    """Sample AppInfoRecord for testing."""
    return AppInfoRecord(
        id="app-123",
        name="Test App",
        module="test.module",
        qualified_name="test.module.TestApp",
        short_description="A test app",
        full_description="A full description of the test app",
    )


@pytest.fixture
def sample_session_info() -> SessionInfoRecord:
    """Sample SessionInfoRecord for testing."""
    return SessionInfoRecord(
        session_id="session-456",
        name="Test Session",
        module="test.session",
        qualname="test.session.TestSession",
        description="A test session",
    )


@pytest.fixture
def sample_messages() -> list[Message]:
    """Sample messages for testing."""
    return [
        Message(
            role="user",
            content="Hello, how are you?",
            time_unix_nano="1000000000",
        ),
        Message(
            role="assistant",
            content="I'm doing well, thank you!",
            time_unix_nano="1000000001",
        ),
    ]


@pytest.fixture
def sample_rating() -> RatingDetails:
    """Sample rating for testing."""
    return RatingDetails(
        value="good",
        rated_at="1700000000000",  # Unix timestamp in milliseconds as string
        rated_by="user",
        notes="Great response!",
    )


@pytest.fixture
def sample_prompt_info() -> PromptInfoRecord:
    """Sample PromptInfoRecord for testing."""
    return PromptInfoRecord(
        prompt_id="prompt-789",
        version_id="v1.0",
        variables={"name": "John", "topic": "AI"},
    )


# ============================================================================
# Test _serialize_json
# ============================================================================


class TestSerializeJson:
    """Tests for _serialize_json function."""

    def test_serialize_none(self):
        """Should return None for None input."""
        assert _serialize_json(None) is None

    def test_serialize_dict(self):
        """Should serialize dict to JSON string."""
        result = _serialize_json({"key": "value"})
        assert result == '{"key": "value"}'

    def test_serialize_pydantic_model(self, sample_app_info: AppInfoRecord):
        """Should serialize Pydantic model to JSON string."""
        result = _serialize_json(sample_app_info)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["id"] == "app-123"
        assert parsed["name"] == "Test App"

    def test_serialize_complex_dict(self):
        """Should serialize complex nested dict."""
        data = {"nested": {"key": [1, 2, 3]}}
        result = _serialize_json(data)
        assert result is not None
        assert json.loads(result) == data


# ============================================================================
# Test _parse_json (generic with overloads)
# ============================================================================


class TestParseJson:
    """Tests for _parse_json function."""

    def test_parse_none(self):
        """Should return None for None input."""
        assert _parse_json(None) is None
        assert _parse_json(None, AppInfoRecord) is None

    def test_parse_dict_no_model(self):
        """Should return dict when no model class provided."""
        data = {"key": "value"}
        result = _parse_json(data)
        assert result == data

    def test_parse_string_no_model(self):
        """Should parse JSON string to dict when no model class."""
        result = _parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_string_with_model(self):
        """Should parse JSON string to Pydantic model."""
        json_str = (
            '{"id": "app-1", "name": "App", "module": "m", "qualified_name": "m.App"}'
        )
        result = _parse_json(json_str, AppInfoRecord)
        assert isinstance(result, AppInfoRecord)
        assert result.id == "app-1"
        assert result.name == "App"

    def test_parse_dict_with_model(self):
        """Should parse dict to Pydantic model."""
        data = {"id": "app-1", "name": "App", "module": "m", "qualified_name": "m.App"}
        result = _parse_json(data, AppInfoRecord)
        assert isinstance(result, AppInfoRecord)
        assert result.id == "app-1"

    def test_parse_rating_details(self):
        """Should parse RatingDetails correctly."""
        # Unix timestamp in milliseconds as string
        data = {
            "value": "good",
            "rated_at": "1700000000000",
            "rated_by": "user",
            "notes": "Great!",
        }
        result = _parse_json(data, RatingDetails)
        assert isinstance(result, RatingDetails)
        assert result.value == "good"
        assert result.rated_at == "1700000000000"
        assert result.rated_by == "user"
        assert result.notes == "Great!"

    def test_type_safety_returns_correct_type(self):
        """Should return correct type based on model_class."""
        # Test with AppInfoRecord
        app_data = '{"id": "a", "name": "A", "module": "m", "qualified_name": "m.A"}'
        app_result = _parse_json(app_data, AppInfoRecord)
        assert isinstance(app_result, AppInfoRecord)

        # Test with SessionInfoRecord
        session_data = (
            '{"session_id": "s", "name": "S", "module": "m", "qualname": "m.S"}'
        )
        session_result = _parse_json(session_data, SessionInfoRecord)
        assert isinstance(session_result, SessionInfoRecord)


# ============================================================================
# Test _parse_messages
# ============================================================================


class TestParseMessages:
    """Tests for _parse_messages function."""

    def test_parse_none(self):
        """Should return empty list for None."""
        assert _parse_messages(None) == []

    def test_parse_empty_list(self):
        """Should return empty list for empty list."""
        assert _parse_messages([]) == []

    def test_parse_list(self):
        """Should parse list of message dicts."""
        data = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = _parse_messages(data)
        assert len(result) == 2
        assert all(isinstance(m, Message) for m in result)
        assert result[0].role == "user"
        assert result[1].content == "Hi!"

    def test_parse_json_string(self):
        """Should parse JSON string of messages."""
        data = '[{"role": "user", "content": "Test"}]'
        result = _parse_messages(data)
        assert len(result) == 1
        assert result[0].role == "user"


# ============================================================================
# Test _parse_rating
# ============================================================================


class TestParseRating:
    """Tests for _parse_rating function."""

    def test_parse_none(self):
        """Should return None for None."""
        assert _parse_rating(None) is None

    def test_parse_dict(self):
        """Should parse dict to RatingDetails."""
        data = {
            "value": "bad",
            "rated_at": "1700000000000",
            "rated_by": "user",
        }  # Unix timestamp in milliseconds as string
        result = _parse_rating(data)
        assert isinstance(result, RatingDetails)
        assert result.value == "bad"
        assert result.rated_by == "user"

    def test_parse_json_string(self):
        """Should parse JSON string to RatingDetails."""
        data = '{"value": "good", "rated_at": "1700000000000", "rated_by": "user"}'
        result = _parse_rating(data)
        assert isinstance(result, RatingDetails)
        assert result.rated_by == "user"
        assert result.value == "good"


# ============================================================================
# Test _parse_prompt_ids
# ============================================================================


class TestParsePromptIds:
    """Tests for _parse_prompt_ids function."""

    def test_parse_none(self):
        """Should return empty list for None."""
        assert _parse_prompt_ids(None) == []

    def test_parse_list(self):
        """Should return list as-is."""
        data = ["prompt-1", "prompt-2"]
        assert _parse_prompt_ids(data) == data

    def test_parse_json_string(self):
        """Should parse JSON string to list."""
        data = '["prompt-1", "prompt-2"]'
        result = _parse_prompt_ids(data)
        assert result == ["prompt-1", "prompt-2"]


# ============================================================================
# Test Database Operations (Integration Tests)
# ============================================================================


# Use a separate test database by patching the engine before importing tables
@pytest_asyncio.fixture(scope="function")
async def test_db(tmp_path):
    """Setup and teardown for database tests using a temporary test DB."""
    from piccolo.engine.sqlite import SQLiteEngine
    from pixie.storage import tables
    from pixie.storage import operations

    # Create a test-specific database in a temporary directory
    test_db_path = str(tmp_path / f"test_pixie_{os.getpid()}.db")
    test_engine = SQLiteEngine(path=test_db_path)

    # Patch the DB engine in tables module
    original_db = tables.DB
    tables.DB = test_engine
    tables.RunRecord._meta._db = test_engine
    tables.LlmCallRecord._meta._db = test_engine

    # Also patch the imported references in operations module
    operations.RunRecordTable._meta._db = test_engine
    operations.LlmCallRecordTable._meta._db = test_engine

    # Create tables
    await tables.create_tables()

    yield test_engine

    # Clean up - delete all records
    try:
        await tables.RunRecord.delete(force=True)
        await tables.LlmCallRecord.delete(force=True)
    except Exception:
        pass  # Ignore errors during cleanup

    # Restore original DB
    tables.DB = original_db
    tables.RunRecord._meta._db = original_db
    tables.LlmCallRecord._meta._db = original_db
    operations.RunRecordTable._meta._db = original_db
    operations.LlmCallRecordTable._meta._db = original_db

    # Remove test database file
    try:
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
    except Exception:
        pass  # Ignore errors during cleanup


class TestRunRecordOperations:
    """Integration tests for run record operations."""

    @pytest.mark.asyncio
    async def test_create_run_record(
        self, test_db, sample_app_info: AppInfoRecord, sample_messages: list[Message]
    ):
        """Should create a run record and return it."""
        input_data = RunRecordInput(
            id="run-001",
            source="apps",
            app_info=sample_app_info,
            messages=sample_messages,
            prompt_ids=["prompt-1"],
            start_time="1700000000000",
            metadata={"test": "value"},
        )

        result = await create_run_record(input_data)

        assert result.id == "run-001"
        assert result.source == "apps"
        assert result.app_info is not None
        assert result.app_info.id == "app-123"
        assert result.messages is not None
        assert len(result.messages) == 2
        assert result.prompt_ids == ["prompt-1"]
        assert result.metadata == {"test": "value"}
        assert result.created_at is not None

    @pytest.mark.asyncio
    async def test_get_run_record(self, test_db, sample_app_info: AppInfoRecord):
        """Should retrieve a run record by ID."""
        # Create a record first
        input_data = RunRecordInput(
            id="run-002",
            source="apps",
            app_info=sample_app_info,
            messages=[],
            prompt_ids=[],
            metadata={},
        )
        await create_run_record(input_data)

        # Retrieve it
        result = await get_run_record("run-002")

        assert result is not None
        assert result.id == "run-002"
        assert result.source == "apps"

    @pytest.mark.asyncio
    async def test_get_run_record_not_found(self, test_db):
        """Should return None for non-existent record."""
        result = await get_run_record("nonexistent-run")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_run_record(self, test_db, sample_app_info: AppInfoRecord):
        """Should update a run record."""
        # Create a record first
        input_data = RunRecordInput(
            id="run-003",
            source="apps",
            app_info=sample_app_info,
            messages=[],
            prompt_ids=[],
            metadata={},
        )
        await create_run_record(input_data)

        # Update it
        update_data = RunRecordUpdate(
            messages=[Message(role="user", content="Updated message")],
            prompt_ids=["new-prompt"],
            end_time="1700000001000",
        )
        result = await update_run_record("run-003", update_data)

        assert result is not None
        assert result.messages is not None
        assert len(result.messages) == 1
        assert result.messages[0].content == "Updated message"
        assert result.prompt_ids == ["new-prompt"]
        assert result.end_time == "1700000001000"

    @pytest.mark.asyncio
    async def test_get_run_records_with_filter(
        self, test_db, sample_app_info: AppInfoRecord
    ):
        """Should filter run records."""
        # Create multiple records
        for i in range(3):
            input_data = RunRecordInput(
                id=f"run-filter-{i}",
                source="apps" if i < 2 else "sessions",
                app_info=sample_app_info if i < 2 else None,
                messages=[],
                prompt_ids=[],
                metadata={},
            )
            await create_run_record(input_data)

        # Filter by source
        filters = RecordFilters(run_source="apps")
        results = await get_run_records(filters)

        assert len(results) >= 2
        assert all(r.source == "apps" for r in results)


class TestLlmCallRecordOperations:
    """Integration tests for LLM call record operations."""

    @pytest.mark.asyncio
    async def test_create_llm_call_record(
        self,
        test_db,
        sample_app_info: AppInfoRecord,
        sample_prompt_info: PromptInfoRecord,
    ):
        """Should create an LLM call record and return it."""
        input_data = LlmCallRecordInput(
            span_id="span-001",
            trace_id="trace-001",
            run_id="run-001",
            run_source="apps",
            app_info=sample_app_info,
            prompt_info=sample_prompt_info,
            llm_input={"messages": [{"role": "user", "content": "Hello"}]},
            llm_output={"content": "Hi!"},
            model_name="gpt-4",
            model_parameters={"temperature": 0.7},
            start_time="1700000000000000",
            end_time="1700000001000000",
            metadata={"test": "value"},
        )

        result = await create_llm_call_record(input_data)

        assert result.span_id == "span-001"
        assert result.trace_id == "trace-001"
        assert result.run_id == "run-001"
        assert result.app_info is not None
        assert result.prompt_info is not None
        assert result.prompt_info.prompt_id == "prompt-789"
        assert result.model_name == "gpt-4"

    @pytest.mark.asyncio
    async def test_get_llm_call_record(self, test_db, sample_app_info: AppInfoRecord):
        """Should retrieve an LLM call record by span ID."""
        # Create a record first
        input_data = LlmCallRecordInput(
            span_id="span-002",
            trace_id="trace-002",
            run_source="apps",
            app_info=sample_app_info,
            llm_input={},
            llm_output={},
            metadata={},
        )
        await create_llm_call_record(input_data)

        # Retrieve it
        result = await get_llm_call_record("span-002")

        assert result is not None
        assert result.span_id == "span-002"
        assert result.trace_id == "trace-002"

    @pytest.mark.asyncio
    async def test_update_llm_call_record_rating(
        self, test_db, sample_app_info: AppInfoRecord
    ):
        """Should update LLM call record with rating."""
        # Create a record first
        input_data = LlmCallRecordInput(
            span_id="span-003",
            trace_id="trace-003",
            run_source="apps",
            llm_input={},
            llm_output={},
            metadata={},
        )
        await create_llm_call_record(input_data)

        # Update with rating
        update_data = LlmCallRecordUpdate(
            rating=RatingDetails(
                value="good", rated_at="1700000000000", rated_by="user"
            )  # Unix timestamp in milliseconds as string
        )
        result = await update_llm_call_record("span-003", update_data)

        assert result is not None
        assert result.rating is not None
        assert result.rating.value == "good"

    @pytest.mark.asyncio
    async def test_get_llm_call_records_by_prompt_id(
        self,
        test_db,
        sample_app_info: AppInfoRecord,
        sample_prompt_info: PromptInfoRecord,
    ):
        """Should filter LLM call records by prompt ID."""
        # Create records with different prompt IDs
        for i in range(3):
            prompt_info = PromptInfoRecord(
                prompt_id=f"prompt-{i % 2}",  # 0, 1, 0
                version_id="v1",
            )
            input_data = LlmCallRecordInput(
                span_id=f"span-prompt-{i}",
                trace_id=f"trace-prompt-{i}",
                run_source="apps",
                prompt_info=prompt_info,
                llm_input={},
                llm_output={},
                metadata={},
            )
            await create_llm_call_record(input_data)

        # Filter by prompt ID
        filters = RecordFilters(prompt_id="prompt-0")
        results = await get_llm_call_records(filters)

        assert len(results) >= 2
        assert all(
            r.prompt_info and r.prompt_info.prompt_id == "prompt-0" for r in results
        )

    @pytest.mark.asyncio
    async def test_get_llm_call_records_by_run(
        self, test_db, sample_app_info: AppInfoRecord
    ):
        """Should get all LLM call records for a specific run."""
        # Create records for different runs
        for i in range(4):
            run_id = "run-A" if i < 2 else "run-B"
            input_data = LlmCallRecordInput(
                span_id=f"span-run-{i}",
                trace_id=f"trace-run-{i}",
                run_id=run_id,
                run_source="apps",
                llm_input={},
                llm_output={},
                metadata={},
            )
            await create_llm_call_record(input_data)

        # Get records for run-A
        results = await get_llm_call_records_by_run("run-A")

        assert len(results) == 2
        assert all(r.run_id == "run-A" for r in results)

    @pytest.mark.asyncio
    async def test_get_llm_call_records_by_multiple_app_ids(
        self, test_db, sample_prompt_info: PromptInfoRecord
    ):
        """Should filter LLM call records by multiple app IDs."""
        # Create records with different app IDs
        app_infos = [
            AppInfoRecord(
                id="app-a", name="App A", module="test.a", qualified_name="test.a.AppA"
            ),
            AppInfoRecord(
                id="app-b", name="App B", module="test.b", qualified_name="test.b.AppB"
            ),
            AppInfoRecord(
                id="app-c", name="App C", module="test.c", qualified_name="test.c.AppC"
            ),
        ]
        for i, app_info in enumerate(app_infos):
            input_data = LlmCallRecordInput(
                span_id=f"span-app-multi-{i}",
                trace_id=f"trace-app-multi-{i}",
                run_source="apps",
                app_info=app_info,
                prompt_info=sample_prompt_info,
                llm_input={},
                llm_output={},
                metadata={},
            )
            await create_llm_call_record(input_data)

        # Filter by multiple app IDs (app-a and app-b)
        filters = RecordFilters(app_ids=["app-a", "app-b"])
        results = await get_llm_call_records(filters)

        assert len(results) == 2
        app_ids_in_results = {r.app_info.id for r in results if r.app_info}
        assert app_ids_in_results == {"app-a", "app-b"}

    @pytest.mark.asyncio
    async def test_get_llm_call_records_by_multiple_rating_values(
        self, test_db, sample_app_info: AppInfoRecord
    ):
        """Should filter LLM call records by multiple rating values."""
        # Create records with different ratings
        ratings: list[RatingDetails] = [
            RatingDetails(value="good", rated_at="1700000000000", rated_by="user"),
            RatingDetails(value="bad", rated_at="1700000000001", rated_by="user"),
            RatingDetails(value="undecided", rated_at="1700000000002", rated_by="user"),
        ]
        for i, rating in enumerate(ratings):
            input_data = LlmCallRecordInput(
                span_id=f"span-rating-multi-{i}",
                trace_id=f"trace-rating-multi-{i}",
                run_source="apps",
                app_info=sample_app_info,
                llm_input={},
                llm_output={},
                metadata={},
            )
            await create_llm_call_record(input_data)
            # Update with rating
            await update_llm_call_record(
                f"span-rating-multi-{i}", LlmCallRecordUpdate(rating=rating)
            )

        # Filter by multiple rating values (good and bad)
        filters = RecordFilters(rating_values=["good", "bad"])
        results = await get_llm_call_records(filters)

        assert len(results) == 2
        rating_values_in_results = {r.rating.value for r in results if r.rating}
        assert rating_values_in_results == {"good", "bad"}

    @pytest.mark.asyncio
    async def test_get_llm_call_records_by_multiple_rated_by_values(
        self, test_db, sample_app_info: AppInfoRecord
    ):
        """Should filter LLM call records by multiple rated_by values."""
        # Create records with different rated_by values
        ratings: list[RatingDetails] = [
            RatingDetails(value="good", rated_at="1700000000000", rated_by="user"),
            RatingDetails(value="good", rated_at="1700000000001", rated_by="ai"),
            RatingDetails(value="good", rated_at="1700000000002", rated_by="system"),
        ]
        for i, rating in enumerate(ratings):
            input_data = LlmCallRecordInput(
                span_id=f"span-ratedby-multi-{i}",
                trace_id=f"trace-ratedby-multi-{i}",
                run_source="apps",
                app_info=sample_app_info,
                llm_input={},
                llm_output={},
                metadata={},
            )
            await create_llm_call_record(input_data)
            # Update with rating
            await update_llm_call_record(
                f"span-ratedby-multi-{i}", LlmCallRecordUpdate(rating=rating)
            )

        # Filter by multiple rated_by values (ai and system - automated)
        filters = RecordFilters(rated_by_values=["ai", "system"])
        results = await get_llm_call_records(filters)

        assert len(results) == 2
        rated_by_values_in_results = {r.rating.rated_by for r in results if r.rating}
        assert rated_by_values_in_results == {"ai", "system"}

    @pytest.mark.asyncio
    async def test_get_llm_call_records_combined_multi_filters(self, test_db):
        """Should filter LLM call records with combined multi-select filters."""
        # Create records with various combinations
        test_cases: list[tuple[str, str, str, str, str, str]] = [
            ("app-x", "App X", "test.x", "test.x.X", "good", "user"),
            ("app-x", "App X", "test.x", "test.x.X", "bad", "ai"),
            ("app-y", "App Y", "test.y", "test.y.Y", "good", "system"),
            ("app-y", "App Y", "test.y", "test.y.Y", "bad", "user"),
        ]
        for i, (app_id, app_name, module, qname, rating_val, rated_by) in enumerate(
            test_cases
        ):
            app_info = AppInfoRecord(
                id=app_id, name=app_name, module=module, qualified_name=qname
            )
            input_data = LlmCallRecordInput(
                span_id=f"span-combined-{i}",
                trace_id=f"trace-combined-{i}",
                run_source="apps",
                app_info=app_info,
                llm_input={},
                llm_output={},
                metadata={},
            )
            await create_llm_call_record(input_data)
            rating = RatingDetails(
                value=rating_val,  # type: ignore[arg-type]
                rated_at=f"170000000000{i}",
                rated_by=rated_by,  # type: ignore[arg-type]
            )
            await update_llm_call_record(
                f"span-combined-{i}", LlmCallRecordUpdate(rating=rating)
            )

        # Filter by app-x AND good rating
        filters = RecordFilters(app_ids=["app-x"], rating_values=["good"])
        results = await get_llm_call_records(filters)

        assert len(results) == 1
        assert results[0].app_info is not None
        assert results[0].app_info.id == "app-x"
        assert results[0].rating is not None
        assert results[0].rating.value == "good"
