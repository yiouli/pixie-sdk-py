"""Tests for InputRequired, AppRunUpdate, and user input schema handling.

These tests validate:
1. InputRequired can accept both type hints and JSON schemas directly
2. AppRunUpdate has a standard user_input_schema field (JsonValue)
3. emit_status_update correctly converts InputRequired to JSON schema
4. SessionUpdate no longer needs custom serialization
"""

from pydantic import BaseModel

from pixie.types import (
    InputRequired,
    AppRunUpdate,
    InputMixin,
)


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    age: int


class TestInputRequiredWithType:
    """Tests for InputRequired with type parameter (existing behavior)."""

    def test_input_required_with_str_type(self):
        """Test InputRequired accepts string type."""
        req = InputRequired(str)
        assert req.expected_type is str

    def test_input_required_with_int_type(self):
        """Test InputRequired accepts int type."""
        req = InputRequired(int)
        assert req.expected_type is int

    def test_input_required_with_pydantic_model(self):
        """Test InputRequired accepts Pydantic model type."""
        req = InputRequired(SampleModel)
        assert req.expected_type is SampleModel

    def test_input_required_with_dict_type(self):
        """Test InputRequired accepts dict type."""
        req = InputRequired(dict)
        assert req.expected_type is dict


class TestInputRequiredWithJsonSchema:
    """Tests for InputRequired with JSON schema (new behavior)."""

    def test_input_required_with_json_schema_dict(self):
        """Test InputRequired accepts a JSON schema dict directly."""
        schema = {"type": "string", "minLength": 1}
        req = InputRequired(schema)
        # When given a schema directly, expected_type should be None
        # and json_schema should be the schema
        assert req.json_schema == schema
        assert req.expected_type is None

    def test_input_required_with_complex_json_schema(self):
        """Test InputRequired accepts complex JSON schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        req = InputRequired(schema)
        assert req.json_schema == schema
        assert req.expected_type is None

    def test_input_required_get_schema_from_type(self):
        """Test InputRequired.get_json_schema() returns schema from type."""
        req = InputRequired(str)
        schema = req.get_json_schema()
        assert schema == {"type": "string"}

    def test_input_required_get_schema_from_pydantic_model(self):
        """Test InputRequired.get_json_schema() returns schema from Pydantic model."""
        req = InputRequired(SampleModel)
        schema = req.get_json_schema()
        # Should return the model's JSON schema
        assert schema is not None
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]

    def test_input_required_get_schema_from_direct_schema(self):
        """Test InputRequired.get_json_schema() returns direct schema."""
        direct_schema = {"type": "string", "format": "email"}
        req = InputRequired(direct_schema)
        schema = req.get_json_schema()
        assert schema == direct_schema


class TestAppRunUpdateUserInputSchema:
    """Tests for AppRunUpdate with standard user_input_schema field."""

    def test_app_run_update_has_user_input_schema_field(self):
        """Test AppRunUpdate has user_input_schema as a standard field."""
        update = AppRunUpdate(
            run_id="test-run",
            status="running",
            user_input_schema={"type": "string"},
        )
        assert update.user_input_schema == {"type": "string"}

    def test_app_run_update_user_input_schema_none(self):
        """Test AppRunUpdate user_input_schema can be None."""
        update = AppRunUpdate(
            run_id="test-run",
            status="running",
        )
        assert update.user_input_schema is None

    def test_app_run_update_serialization(self):
        """Test AppRunUpdate serializes user_input_schema correctly."""
        schema = {"type": "integer", "minimum": 0}
        update = AppRunUpdate(
            run_id="test-run",
            status="waiting",
            user_input_schema=schema,
        )
        data = update.model_dump()
        assert data["user_input_schema"] == schema

    def test_app_run_update_deserialization(self):
        """Test AppRunUpdate deserializes user_input_schema correctly."""
        data = {
            "run_id": "test-run",
            "status": "waiting",
            "user_input_schema": {"type": "string"},
        }
        update = AppRunUpdate.model_validate(data)
        assert update.user_input_schema == {"type": "string"}

    def test_app_run_update_no_private_attrs_needed(self):
        """Test AppRunUpdate doesn't need private attributes for schema."""
        # This tests that we can round-trip without custom serialization
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        original = AppRunUpdate(
            run_id="test-run",
            status="waiting",
            user_input_schema=schema,
            user_input=None,
            data={"result": 42},
        )

        # Serialize and deserialize
        json_str = original.model_dump_json()
        restored = AppRunUpdate.model_validate_json(json_str)

        assert restored.user_input_schema == schema
        assert restored.run_id == "test-run"
        assert restored.status == "waiting"
        assert restored.data == {"result": 42}


class TestInputMixinSimplified:
    """Tests for simplified InputMixin without private attributes."""

    def test_input_mixin_user_input_schema_field(self):
        """Test InputMixin has user_input_schema as standard field."""

        # Create a concrete class using InputMixin for testing
        class TestUpdate(InputMixin):
            id: str

        update = TestUpdate(
            id="test",
            user_input_schema={"type": "string"},
        )
        assert update.user_input_schema == {"type": "string"}

    def test_input_mixin_serialization_roundtrip(self):
        """Test InputMixin serializes without custom serializer."""

        class TestUpdate(InputMixin):
            id: str

        original = TestUpdate(
            id="test",
            user_input_schema={"type": "integer"},
            user_input="42",
        )

        json_str = original.model_dump_json()
        restored = TestUpdate.model_validate_json(json_str)

        assert restored.user_input_schema == {"type": "integer"}
        assert restored.user_input == "42"


class TestSessionUpdateSimplified:
    """Tests for SessionUpdate without custom serialization."""

    def test_session_update_user_input_schema_field(self):
        """Test SessionUpdate has user_input_schema as standard field."""
        from pixie.session.types import SessionUpdate

        update = SessionUpdate(
            session_id="test-session",
            status="waiting",
            user_input_schema={"type": "string"},
        )
        assert update.user_input_schema == {"type": "string"}

    def test_session_update_standard_serialization(self):
        """Test SessionUpdate uses standard serialization (no custom serializer)."""
        from pixie.session.types import SessionUpdate

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        original = SessionUpdate(
            session_id="test-session",
            status="waiting",
            user_input_schema=schema,
        )

        # Standard serialization
        json_str = original.model_dump_json()
        restored = SessionUpdate.model_validate_json(json_str)

        assert restored.user_input_schema == schema
        assert restored.session_id == "test-session"
        assert restored.status == "waiting"

    def test_session_update_no_schema_serialization_key(self):
        """Test SessionUpdate doesn't use special serialization key."""
        from pixie.session.types import SessionUpdate

        update = SessionUpdate(
            session_id="test-session",
            status="running",
            user_input_schema={"type": "string"},
        )

        data = update.model_dump()

        # Should have user_input_schema, not user_input_requirement_schema
        assert "user_input_schema" in data
        assert "user_input_requirement_schema" not in data


class TestEmitStatusUpdateConversion:
    """Tests for emit_status_update converting InputRequired to schema."""

    def test_emit_creates_update_with_schema_from_type(self):
        """Test emit_status_update converts InputRequired(type) to JSON schema."""
        # This test validates the behavior of emit_status_update
        # when given an InputRequired with a type

        # Mock the execution context
        import pixie.execution_context as exec_ctx
        from pixie.types import ExecutionContext
        import janus
        import threading

        # Create a mock context with a queue
        queue = janus.Queue()
        ctx = ExecutionContext(
            run_id="test-run",
            status_queue=queue,
            resume_event=threading.Event(),
            breakpoint_config=None,
        )

        # Set up the context
        exec_ctx._execution_context.set(ctx)

        try:
            # Emit with InputRequired(str)
            exec_ctx.emit_status_update(
                status="waiting",
                user_input_requirement=InputRequired(str),
            )

            # Get the update from queue
            update = queue.sync_q.get_nowait()

            # The update should have user_input_schema set
            assert update is not None
            assert update.user_input_schema == {"type": "string"}
            assert update.status == "waiting"
        finally:
            exec_ctx._execution_context.set(None)
            queue.close()

    def test_emit_creates_update_with_schema_from_pydantic(self):
        """Test emit_status_update converts InputRequired(Model) to JSON schema."""
        import pixie.execution_context as exec_ctx
        from pixie.types import ExecutionContext
        import janus
        import threading

        queue = janus.Queue()
        ctx = ExecutionContext(
            run_id="test-run",
            status_queue=queue,
            resume_event=threading.Event(),
            breakpoint_config=None,
        )

        exec_ctx._execution_context.set(ctx)

        try:
            exec_ctx.emit_status_update(
                status="waiting",
                user_input_requirement=InputRequired(SampleModel),
            )

            update = queue.sync_q.get_nowait()

            assert update is not None
            assert update.user_input_schema is not None
            assert "properties" in update.user_input_schema
            assert "name" in update.user_input_schema["properties"]
        finally:
            exec_ctx._execution_context.set(None)
            queue.close()

    def test_emit_creates_update_with_direct_schema(self):
        """Test emit_status_update passes through direct JSON schema."""
        import pixie.execution_context as exec_ctx
        from pixie.types import ExecutionContext
        import janus
        import threading

        queue = janus.Queue()
        ctx = ExecutionContext(
            run_id="test-run",
            status_queue=queue,
            resume_event=threading.Event(),
            breakpoint_config=None,
        )

        exec_ctx._execution_context.set(ctx)

        try:
            direct_schema = {"type": "string", "format": "email"}
            exec_ctx.emit_status_update(
                status="waiting",
                user_input_requirement=InputRequired(direct_schema),
            )

            update = queue.sync_q.get_nowait()

            assert update is not None
            assert update.user_input_schema == direct_schema
        finally:
            exec_ctx._execution_context.set(None)
            queue.close()

    def test_emit_without_requirement_has_no_schema(self):
        """Test emit_status_update without requirement has no schema."""
        import pixie.execution_context as exec_ctx
        from pixie.types import ExecutionContext
        import janus
        import threading

        queue = janus.Queue()
        ctx = ExecutionContext(
            run_id="test-run",
            status_queue=queue,
            resume_event=threading.Event(),
            breakpoint_config=None,
        )

        exec_ctx._execution_context.set(ctx)

        try:
            exec_ctx.emit_status_update(
                status="running",
                data={"message": "processing"},
            )

            update = queue.sync_q.get_nowait()

            assert update is not None
            assert update.user_input_schema is None
        finally:
            exec_ctx._execution_context.set(None)
            queue.close()


class TestStrawberryConversion:
    """Tests for strawberry AppRunUpdate conversion."""

    def test_strawberry_conversion_uses_standard_field(self):
        """Test strawberry AppRunUpdate.from_pydantic uses standard field."""
        from pixie.schema import AppRunUpdate as StrawberryAppRunUpdate

        pydantic_update = AppRunUpdate(
            run_id="test-run",
            status="waiting",
            user_input_schema={"type": "string"},
        )

        strawberry_update = StrawberryAppRunUpdate.from_pydantic(pydantic_update)

        # Should directly use the schema without conversion
        assert strawberry_update.user_input_schema is not None
        # The schema should match (wrapped in JSON type)
        assert strawberry_update.user_input_schema == {"type": "string"}

    def test_strawberry_conversion_no_schema(self):
        """Test strawberry conversion when no schema present."""
        from pixie.schema import AppRunUpdate as StrawberryAppRunUpdate

        pydantic_update = AppRunUpdate(
            run_id="test-run",
            status="running",
        )

        strawberry_update = StrawberryAppRunUpdate.from_pydantic(pydantic_update)

        assert strawberry_update.user_input_schema is None
