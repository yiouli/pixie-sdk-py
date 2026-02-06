"""Tests for session info capture and communication."""

import asyncio
import struct
import pytest

from pixie.session.rpc import (
    connect_to_server,
    disconnect_from_server,
    listen_to_client_connections,
    stop_server,
    get_connected_sessions,
    _get_server_state,
)
from pixie.session.types import SessionInfo


# Test timeout for socket operations
TIMEOUT = 10.0

# Base port for tests - use different ranges to avoid conflicts
TEST_PORT_BASE = 18000


def get_test_port(test_id: int) -> int:
    """Get a unique port for a test to avoid conflicts."""
    return TEST_PORT_BASE + test_id


@pytest.fixture
def cleanup():
    """Cleanup fixture to ensure connections are closed after each test."""
    yield
    disconnect_from_server()
    stop_server()


async def start_session_server(port: int) -> asyncio.Task:
    """Start the session server and wait briefly for it to listen."""
    task = asyncio.create_task(listen_to_client_connections(port))
    await asyncio.sleep(0.05)  # Minimal delay for server to start listening
    return task


class TestSessionInfo:
    """Tests for SessionInfo type."""

    def test_session_info_creation(self):
        """Test that SessionInfo can be created with all required fields."""
        info = SessionInfo(
            session_id="test-123",
            name="my_function",
            module="my_module",
            qualname="my_function",
            description="A test function",
        )
        assert info.session_id == "test-123"
        assert info.name == "my_function"
        assert info.module == "my_module"
        assert info.qualname == "my_function"
        assert info.description == "A test function"

    def test_session_info_optional_description(self):
        """Test that SessionInfo works without description."""
        info = SessionInfo(
            session_id="test-123",
            name="my_function",
            module="my_module",
            qualname="my_function",
        )
        assert info.description is None

    def test_session_info_serialization(self):
        """Test that SessionInfo serializes and deserializes correctly."""
        info = SessionInfo(
            session_id="test-123",
            name="my_function",
            module="my_module",
            qualname="MyClass.my_function",
            description="A test function",
        )
        json_str = info.model_dump_json()
        parsed = SessionInfo.model_validate_json(json_str)
        assert parsed == info


class TestSessionInfoHandshake:
    """Tests for session info in the client-server handshake."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_client_sends_session_info(self, cleanup):
        """Test that client sends SessionInfo during registration."""
        port = get_test_port(1)
        server_task = await start_session_server(port)

        session_info = SessionInfo(
            session_id="info-test-session",
            name="test_function",
            module="test_module",
            qualname="test_function",
            description="Test description",
        )

        # Connect with session info
        connected_port = await connect_to_server(
            "localhost", port, session_info=session_info
        )

        assert connected_port == port
        await asyncio.sleep(0.1)  # Give server time to process

        # Verify server received session info
        server_state = _get_server_state()
        assert "info-test-session" in server_state.client_connections
        conn = server_state.client_connections["info-test-session"]
        assert conn.session_info is not None
        assert conn.session_info.name == "test_function"
        assert conn.session_info.module == "test_module"
        assert conn.session_info.description == "Test description"

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_get_connected_sessions_returns_session_info(self, cleanup):
        """Test that get_connected_sessions returns SessionInfo objects."""
        port = get_test_port(2)
        server_task = await start_session_server(port)

        session_info = SessionInfo(
            session_id="sessions-test",
            name="my_session_func",
            module="my.module",
            qualname="MyClass.my_session_func",
            description="My session description",
        )

        await connect_to_server("localhost", port, session_info=session_info)
        await asyncio.sleep(0.1)

        # Get connected sessions
        sessions = await get_connected_sessions()

        assert len(sessions) == 1
        assert sessions[0].session_id == "sessions-test"
        assert sessions[0].name == "my_session_func"
        assert sessions[0].module == "my.module"
        assert sessions[0].qualname == "MyClass.my_session_func"
        assert sessions[0].description == "My session description"

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_get_connected_sessions_multiple_clients(self, cleanup):
        """Test get_connected_sessions with multiple connected clients."""
        port = get_test_port(3)
        server_task = await start_session_server(port)

        # Connect first client
        session_info1 = SessionInfo(
            session_id="multi-session-1",
            name="func_one",
            module="module_one",
            qualname="func_one",
        )
        await connect_to_server("localhost", port, session_info=session_info1)
        await asyncio.sleep(0.1)
        disconnect_from_server()

        # Connect second client
        session_info2 = SessionInfo(
            session_id="multi-session-2",
            name="func_two",
            module="module_two",
            qualname="func_two",
            description="Second function",
        )
        await connect_to_server("localhost", port, session_info=session_info2)
        await asyncio.sleep(0.1)

        # Get connected sessions - first one may have been cleaned up
        sessions = await get_connected_sessions()

        # At least the second session should be present
        session_ids = [s.session_id for s in sessions]
        assert "multi-session-2" in session_ids

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_backward_compatibility_session_id_only(self, cleanup):
        """Test that old clients sending only session_id still work."""
        port = get_test_port(4)
        server_task = await start_session_server(port)

        # Simulate old client sending just session_id
        reader, writer = await asyncio.open_connection("localhost", port)

        # Send old format: just session_id
        session_id = b"legacy-session"
        length_prefix = struct.pack(">I", len(session_id))
        writer.write(length_prefix + session_id)
        await writer.drain()

        # Wait for ACK
        try:
            header = await asyncio.wait_for(reader.readexactly(4), timeout=2.0)
            msg_len = struct.unpack(">I", header)[0]
            ack = await reader.readexactly(msg_len)
            assert ack == b"ACK"
        except Exception:
            pytest.fail("Server should accept legacy session_id format")

        await asyncio.sleep(0.1)

        # Verify session is registered with basic info
        sessions = await get_connected_sessions()
        session_ids = [s.session_id for s in sessions]
        assert "legacy-session" in session_ids

        # Find the legacy session and verify it has minimal info
        legacy_session = next(s for s in sessions if s.session_id == "legacy-session")
        assert legacy_session.name == "legacy-session"  # Falls back to session_id

        writer.close()
        await writer.wait_closed()

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


class TestSessionDecoratorCapture:
    """Tests for @session decorator capturing function metadata."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_session_decorator_captures_function_name(self, cleanup):
        """Test that @session decorator captures function name."""
        from unittest.mock import patch, AsyncMock
        from pixie.session.client import session

        captured_session_info = None

        async def mock_connect(*args, **kwargs):
            nonlocal captured_session_info
            captured_session_info = kwargs.get("session_info")

        @session
        async def my_test_function():
            """A test function for verification."""
            pass

        with patch("pixie.session.client.connect_to_server", new=mock_connect), patch(
            "pixie.session.client.notify_server", new=AsyncMock()
        ), patch("pixie.session.client.disconnect_from_server"), patch(
            "pixie.server_utils.enable_instrumentations"
        ), patch(
            "pixie.session.client.get_client",
            return_value=AsyncMock(auth_check=lambda: False),
        ), patch(
            "pixie.session.client.execution_context.init_run"
        ):
            await my_test_function()

        assert captured_session_info is not None
        assert captured_session_info.name == "my_test_function"
        assert "test_session_info" in captured_session_info.module
        assert captured_session_info.qualname == (
            "TestSessionDecoratorCapture.test_session_decorator_captures_function_name.<locals>.my_test_function"
        )
        assert captured_session_info.description == "A test function for verification."

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_session_decorator_handles_no_docstring(self, cleanup):
        """Test that @session decorator handles functions without docstrings."""
        from unittest.mock import patch, AsyncMock
        from pixie.session.client import session

        captured_session_info = None

        async def mock_connect(*args, **kwargs):
            nonlocal captured_session_info
            captured_session_info = kwargs.get("session_info")

        @session
        async def function_without_docstring():
            pass

        with patch("pixie.session.client.connect_to_server", new=mock_connect), patch(
            "pixie.session.client.notify_server", new=AsyncMock()
        ), patch("pixie.session.client.disconnect_from_server"), patch(
            "pixie.server_utils.enable_instrumentations"
        ), patch(
            "pixie.session.client.get_client",
            return_value=AsyncMock(auth_check=lambda: False),
        ), patch(
            "pixie.session.client.execution_context.init_run"
        ):
            await function_without_docstring()

        assert captured_session_info is not None
        assert captured_session_info.name == "function_without_docstring"
        assert captured_session_info.description is None

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_session_decorator_async_generator_captures_metadata(self, cleanup):
        """Test that @session decorator captures metadata for async generators."""
        from unittest.mock import patch, AsyncMock
        from pixie.session.client import session

        captured_session_info = None

        async def mock_connect(*args, **kwargs):
            nonlocal captured_session_info
            captured_session_info = kwargs.get("session_info")

        @session
        async def my_generator_session():
            """A generator session function."""
            yield "value1"
            yield "value2"

        with patch("pixie.session.client.connect_to_server", new=mock_connect), patch(
            "pixie.session.client.notify_server", new=AsyncMock()
        ), patch("pixie.session.client.disconnect_from_server"), patch(
            "pixie.server_utils.enable_instrumentations"
        ), patch(
            "pixie.session.client.get_client",
            return_value=AsyncMock(auth_check=lambda: False),
        ), patch(
            "pixie.session.client.execution_context.init_run"
        ):
            gen = my_generator_session()
            async for _ in gen:
                break  # Just iterate once to trigger setup

        assert captured_session_info is not None
        assert captured_session_info.name == "my_generator_session"
        assert captured_session_info.description == "A generator session function."
