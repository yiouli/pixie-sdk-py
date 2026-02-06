"""Tests for pixie.session module - TCP-based RPC client/server communication."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel

from pixie.session.rpc import (
    connect_to_server,
    disconnect_from_server,
    notify_server,
    wait_for_input,
    wait_for_client_update,
    listen_to_client_connections,
    stop_server,
    send_input_to_client,
    _get_client_state,
    _get_server_state,
    REGISTRATION_TIMEOUT,
)
from pixie.session.types import SessionUpdate
from pixie.session.client import print as session_print, input as session_input, session
from pixie.types import InputRequired


# Test timeout for socket operations
TIMEOUT = 10.0

# Base port for tests - use different ranges to avoid conflicts
TEST_PORT_BASE = 17000


class UserModel(BaseModel):
    """Test model for typed input."""

    name: str
    age: int


def get_test_port(test_id: int) -> int:
    """Get a unique port for a test to avoid conflicts.

    Args:
        test_id: Unique identifier for the test (use different numbers per test).

    Returns:
        Port number.
    """
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


async def expect_update(session_id: str, timeout: float = 2.0) -> SessionUpdate:
    """Fetch the next update for a session with a timeout."""
    return await asyncio.wait_for(wait_for_client_update(session_id), timeout)


class TestServerStartup:
    """Tests for server startup and port binding."""

    @pytest.mark.asyncio
    async def test_server_binds_to_port(self, cleanup):
        """Test that server successfully binds to specified port."""
        port = get_test_port(1)
        server_task = await start_session_server(port)

        state = _get_server_state()
        assert state is not None
        assert len(state.servers) == 1

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_server_already_running_raises(self, cleanup):
        """Test that starting server twice raises error."""
        port = get_test_port(2)
        server_task = await start_session_server(port)

        with pytest.raises(RuntimeError, match="Server already running"):
            await listen_to_client_connections(port)

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    def test_get_server_state_not_running(self, cleanup):
        """Test that _get_server_state raises when not running."""
        with pytest.raises(RuntimeError, match="not initialized"):
            _get_server_state()

    @pytest.mark.asyncio
    async def test_stop_server_signals_shutdown(self, cleanup):
        """Test that stop_server signals the server to stop."""
        port = get_test_port(3)
        server_task = await start_session_server(port)

        stop_server()
        # Give server time to stop
        await asyncio.sleep(0.3)

        # Server task should complete after stop_server is called
        assert server_task.done() or server_task.cancelled()


class TestClientConnection:
    """Tests for client-side connection management."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_client_connects_and_receives_ack(self, cleanup):
        """Test that client connects to server and receives ACK."""
        port = get_test_port(10)
        server_task = await start_session_server(port)

        # Connect should succeed and return the port used
        connected_port = await connect_to_server("localhost", port, "test-session")

        assert connected_port == port
        state = _get_client_state()
        assert state is not None
        assert state.session_id == "test-session"
        assert state.connected_port == connected_port

        stop_server()
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_client_registered_on_server(self, cleanup):
        """Test that client is properly registered on server after connect."""
        port = get_test_port(11)
        server_task = await start_session_server(port)

        await connect_to_server("localhost", port, "registered-session")
        await asyncio.sleep(0.1)  # Give server time to process registration

        # Verify server has the client registered
        server_state = _get_server_state()
        assert "registered-session" in server_state.client_connections

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_connect_twice_raises(self, cleanup):
        """Test that connecting twice raises error."""
        port = get_test_port(12)
        server_task = await start_session_server(port)

        await connect_to_server("localhost", port, "test-session")

        with pytest.raises(RuntimeError, match="Already connected"):
            await connect_to_server("localhost", port, "test-session-2")

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    def test_get_client_state_not_connected(self, cleanup):
        """Test that _get_client_state raises when not connected."""
        with pytest.raises(RuntimeError, match="not connected"):
            _get_client_state()

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up(self, cleanup):
        """Test that disconnect_from_server cleans up properly."""
        port = get_test_port(13)
        server_task = await start_session_server(port)

        await connect_to_server("localhost", port, "test-session")
        disconnect_from_server()

        with pytest.raises(RuntimeError, match="not connected"):
            _get_client_state()

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    def test_disconnect_safe_when_not_connected(self, cleanup):
        """Test that disconnect is safe when not connected."""
        disconnect_from_server()  # Should not raise

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_connect_fails_when_no_server(self, cleanup):
        """Test that connect fails when no server is available."""
        port = get_test_port(15)
        # Don't start a server

        with pytest.raises(RuntimeError, match="Failed to connect"):
            await connect_to_server("localhost", port, "no-server-test")


class TestRegistrationTimeout:
    """Tests for session registration timeout behavior."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_server_disconnects_slow_client(self, cleanup):
        """Test that server disconnects client that doesn't send session_id in time."""
        port = get_test_port(20)
        server_task = await start_session_server(port)

        # Connect directly without using our client (to simulate slow registration)
        reader, writer = await asyncio.open_connection("localhost", port)

        # Don't send session_id, wait for timeout
        await asyncio.sleep(REGISTRATION_TIMEOUT + 0.5)

        # Try to read - should get EOF (connection closed by server)
        try:
            data = await asyncio.wait_for(reader.read(100), timeout=1.0)
            # If we get empty data or connection closed
            assert data == b""
        except (ConnectionError, asyncio.TimeoutError):
            pass  # Expected - connection closed

        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_server_rejects_duplicate_session_id(self, cleanup):
        """Test that server rejects a second client with same session_id."""
        port = get_test_port(21)
        server_task = await start_session_server(port)

        # First client connects successfully
        await connect_to_server("localhost", port, "duplicate-session")
        await asyncio.sleep(0.1)

        # Try to connect second client manually with same session_id
        import struct

        reader, writer = await asyncio.open_connection("localhost", port)

        # Send session_id
        session_id = b"duplicate-session"
        length_prefix = struct.pack(">I", len(session_id))
        writer.write(length_prefix + session_id)
        await writer.drain()

        # Wait for response - should be closed (no ACK)
        try:
            data = await asyncio.wait_for(reader.read(100), timeout=2.0)
            # Should get empty data (connection closed) or possibly NAK
            assert data == b"" or b"ACK" not in data
        except (ConnectionError, asyncio.TimeoutError):
            pass  # Expected

        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


class TestClientServerCommunication:
    """Tests for client-server message passing."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_notify_server_sends_update(self, cleanup):
        """Test that notify_server sends SessionUpdate to server queue."""
        port = get_test_port(30)
        server_task = await start_session_server(port)

        await connect_to_server("localhost", port, "test-session-1")
        await asyncio.sleep(0.1)  # Give server time to register client

        update = SessionUpdate(
            session_id="test-session-1",
            status="running",
            time_unix_nano="1234567890000000000",
            data="hello server",
        )
        await notify_server(update)

        # Check queue received the update
        received = await expect_update("test-session-1")
        assert received.session_id == "test-session-1"
        assert received.status == "running"
        assert received.data == "hello server"

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_multiple_updates_queued(self, cleanup):
        """Test that multiple updates are queued in order."""
        port = get_test_port(31)
        server_task = await start_session_server(port)

        await connect_to_server("localhost", port, "session-multi")
        await asyncio.sleep(0.1)

        for i in range(3):
            await notify_server(
                SessionUpdate(
                    session_id="session-multi",
                    status="running",
                    time_unix_nano="1234567890000000000",
                    data=f"message {i}",
                )
            )

        for i in range(3):
            received = await expect_update("session-multi")
            assert received.session_id == "session-multi"
            assert received.data == f"message {i}"

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_notify_server_not_connected(self, cleanup):
        """Test that notify_server raises when not connected."""
        with pytest.raises(RuntimeError, match="not connected"):
            await notify_server(
                SessionUpdate(
                    session_id="test",
                    status="running",
                    time_unix_nano="1234567890000000000",
                )
            )


class TestInputFlow:
    """Tests for wait_for_input and send_input_to_client."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_wait_for_input_receives_string(self, cleanup):
        """Test that client receives string input from server."""
        port = get_test_port(40)
        server_task = await start_session_server(port)

        await connect_to_server("localhost", port, "input-test")
        await asyncio.sleep(0.1)

        # Send update so server knows client is waiting
        await notify_server(
            SessionUpdate(
                session_id="input-test",
                status="waiting",
                time_unix_nano="1234567890000000000",
            )
        )
        await asyncio.sleep(0.1)

        # Server sends input
        async def send_input():
            await asyncio.sleep(0.2)
            await send_input_to_client("input-test", "user input value")

        asyncio.create_task(send_input())

        # Client waits for input
        result = await wait_for_input(InputRequired(str))
        assert result == "user input value"

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_wait_for_input_receives_pydantic_model(self, cleanup):
        """Test that client receives and deserializes Pydantic model."""
        port = get_test_port(41)
        server_task = await start_session_server(port)

        await connect_to_server("localhost", port, "model-test")
        await asyncio.sleep(0.1)

        # Server sends Pydantic model
        async def send_input():
            await asyncio.sleep(0.2)
            await send_input_to_client("model-test", UserModel(name="Alice", age=30))

        asyncio.create_task(send_input())

        # Client waits for input
        result = await wait_for_input(InputRequired(UserModel))
        assert isinstance(result, UserModel)
        assert result.name == "Alice"
        assert result.age == 30

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_send_input_to_unknown_client_raises(self, cleanup):
        """Test that sending to unknown client raises KeyError."""
        port = get_test_port(42)
        server_task = await start_session_server(port)

        with pytest.raises(KeyError, match="No client connected"):
            await send_input_to_client("unknown-session", "data")

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_wait_for_input_not_connected(self, cleanup):
        """Test that wait_for_input raises when not connected."""
        with pytest.raises(RuntimeError, match="not connected"):
            await wait_for_input(InputRequired(str))

    @pytest.mark.asyncio
    async def test_send_input_server_not_running(self, cleanup):
        """Test that send_input_to_client raises when server not running."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await send_input_to_client("session", "data")


class TestConnectionCleanup:
    """Tests for connection cleanup and error handling."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_client_disconnect_removes_from_server_registry(self, cleanup):
        """Test that disconnecting client removes it from server registry."""
        port = get_test_port(50)
        server_task = await start_session_server(port)

        await connect_to_server("localhost", port, "cleanup-test")
        await asyncio.sleep(0.1)

        # Verify registered
        server_state = _get_server_state()
        assert "cleanup-test" in server_state.client_connections

        # Disconnect
        disconnect_from_server()
        await asyncio.sleep(0.3)  # Give server time to detect disconnect

        # Should be removed from registry
        assert "cleanup-test" not in server_state.client_connections

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_sequential_client_connections(self, cleanup):
        """Test that clients can connect sequentially on the same port."""
        port = get_test_port(51)
        server_task = await start_session_server(port)

        # Connect first client
        await connect_to_server("localhost", port, "client-1")
        await asyncio.sleep(0.1)

        # Store state and disconnect first client
        first_port = _get_client_state().connected_port
        assert first_port == port
        disconnect_from_server()
        await asyncio.sleep(0.1)

        # Connect second client
        await connect_to_server("localhost", port, "client-2")
        await asyncio.sleep(0.1)

        # Second client should work on same port
        state = _get_client_state()
        assert state.session_id == "client-2"
        assert state.connected_port == port

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


class TestFullRoundTrip:
    """Full round-trip integration tests."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_full_client_server_roundtrip(self, cleanup):
        """Test complete client-server communication cycle."""
        port = get_test_port(60)
        # Start server
        server_task = await start_session_server(port)

        # Connect client
        connected_port = await connect_to_server("localhost", port, "roundtrip")
        assert connected_port == port
        await asyncio.sleep(0.1)

        # Client sends running status
        await notify_server(
            SessionUpdate(
                session_id="roundtrip",
                status="running",
                time_unix_nano="1234567890000000000",
                data="starting",
            )
        )

        # Client sends waiting status
        await notify_server(
            SessionUpdate(
                session_id="roundtrip",
                status="waiting",
                time_unix_nano="1234567890000000000",
                data="need input",
            )
        )

        # Verify server received both updates
        update1 = await expect_update("roundtrip")
        assert update1.session_id == "roundtrip"
        assert update1.status == "running"

        update2 = await expect_update("roundtrip")
        assert update2.session_id == "roundtrip"
        assert update2.status == "waiting"

        # Server sends input back
        async def send_input():
            await asyncio.sleep(0.1)
            await send_input_to_client("roundtrip", {"response": "here you go"})

        asyncio.create_task(send_input())

        # Client receives input
        result = await wait_for_input(InputRequired(dict))
        assert result == {"response": "here you go"}

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_large_message_handling(self, cleanup):
        """Test handling of large messages."""
        port = get_test_port(61)
        server_task = await start_session_server(port)

        await connect_to_server("localhost", port, "large-msg-test")
        await asyncio.sleep(0.1)

        # Send a large message
        large_data = "x" * 100000  # 100KB of data
        await notify_server(
            SessionUpdate(
                session_id="large-msg-test",
                status="running",
                time_unix_nano="1234567890000000000",
                data=large_data,
            )
        )

        received = await expect_update("large-msg-test", timeout=5.0)
        assert received.data == large_data

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


class TestSessionClientFunctions:
    """Tests for session client.py functions."""

    @pytest.mark.asyncio
    async def test_session_print_sends_update(self):
        """Test that session.print sends data via RPC."""
        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.notify_server") as mock_notify:
                mock_notify.return_value = None

                mock_exec_ctx = MagicMock()
                mock_exec_ctx.run_id = "test-run-123"
                mock_ctx.get_current_context.return_value = mock_exec_ctx

                await session_print("Hello, world!")

                mock_notify.assert_called_once()
                call_args = mock_notify.call_args[0][0]
                assert isinstance(call_args, SessionUpdate)
                assert call_args.session_id == "test-run-123"
                assert call_args.status == "running"
                assert call_args.data == "Hello, world!"

    @pytest.mark.asyncio
    async def test_session_print_with_json_data(self):
        """Test that session.print handles JSON data."""
        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.notify_server") as mock_notify:
                mock_notify.return_value = None

                mock_exec_ctx = MagicMock()
                mock_exec_ctx.run_id = "test-run-123"
                mock_ctx.get_current_context.return_value = mock_exec_ctx

                await session_print({"key": "value", "count": 42})

                call_args = mock_notify.call_args[0][0]
                assert call_args.data == {"key": "value", "count": 42}

    @pytest.mark.asyncio
    async def test_session_print_no_context(self):
        """Test that session.print does nothing without execution context."""
        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.notify_server") as mock_notify:
                mock_ctx.get_current_context.return_value = None

                await session_print("Should not send")

                mock_notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_input_string(self):
        """Test that session.input requests string input."""
        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.notify_server") as mock_notify:
                with patch("pixie.session.client.wait_for_input") as mock_wait:
                    mock_notify.return_value = None
                    mock_wait.return_value = "user input"

                    mock_exec_ctx = MagicMock()
                    mock_exec_ctx.run_id = "test-run-123"
                    mock_ctx.get_current_context.return_value = mock_exec_ctx

                    result = await session_input("Enter your name: ")

                    assert result == "user input"

                    # Should have two notify_server calls: waiting and running with input
                    assert mock_notify.call_count == 2

                    # First call: waiting status
                    first_call = mock_notify.call_args_list[0][0][0]
                    assert first_call.status == "waiting"
                    assert first_call.data == "Enter your name: "

                    # Second call: running status with user input
                    second_call = mock_notify.call_args_list[1][0][0]
                    assert second_call.status == "running"
                    assert second_call.user_input == "user input"

    @pytest.mark.asyncio
    async def test_session_input_typed(self):
        """Test that session.input requests typed input."""
        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.notify_server") as mock_notify:
                with patch("pixie.session.client.wait_for_input") as mock_wait:
                    mock_notify.return_value = None
                    user_model = UserModel(name="Test", age=20)
                    mock_wait.return_value = user_model

                    mock_exec_ctx = MagicMock()
                    mock_exec_ctx.run_id = "test-run-123"
                    mock_ctx.get_current_context.return_value = mock_exec_ctx

                    result = await session_input(
                        "Enter user data: ",
                        expected_type=UserModel,
                    )

                    assert isinstance(result, UserModel)
                    assert result.name == "Test"

                    # Should have two notify_server calls: waiting and running with input
                    assert mock_notify.call_count == 2

                    # First call: waiting status
                    first_call = mock_notify.call_args_list[0][0][0]
                    assert first_call.status == "waiting"
                    assert first_call.data == "Enter user data: "

                    # Second call: running status with user input (serialized)
                    second_call = mock_notify.call_args_list[1][0][0]
                    assert second_call.status == "running"
                    assert second_call.user_input == user_model.model_dump(mode="json")

    @pytest.mark.asyncio
    async def test_session_input_no_context_raises(self):
        """Test that session.input raises without execution context."""
        with patch("pixie.session.client.execution_context") as mock_ctx:
            mock_ctx.get_current_context.return_value = None

            with pytest.raises(RuntimeError, match="No execution context"):
                await session_input("prompt")

    @pytest.mark.asyncio
    async def test_session_input_sends_user_input_notify(self):
        """Test that session.input sends a second notify with the received user input."""
        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.notify_server") as mock_notify:
                with patch("pixie.session.client.wait_for_input") as mock_wait:
                    mock_notify.return_value = None
                    mock_wait.return_value = {"key": "value", "number": 42}

                    mock_exec_ctx = MagicMock()
                    mock_exec_ctx.run_id = "test-run-456"
                    mock_ctx.get_current_context.return_value = mock_exec_ctx

                    result = await session_input("Enter data: ")

                    assert result == {"key": "value", "number": 42}

                    # Verify two notifies: waiting and running with user_input
                    assert mock_notify.call_count == 2

                    waiting_call = mock_notify.call_args_list[0][0][0]
                    assert waiting_call.status == "waiting"
                    assert waiting_call.data == "Enter data: "

                    running_call = mock_notify.call_args_list[1][0][0]
                    assert running_call.status == "running"
                    assert running_call.user_input == {"key": "value", "number": 42}
                    assert running_call.session_id == "test-run-456"


class TestSessionDecorator:
    """Tests for the @session decorator."""

    @pytest.mark.asyncio
    async def test_session_decorator_creates_context(self):
        """Test that @session creates execution context."""
        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.connect_to_server") as mock_connect:
                with patch("pixie.session.client.notify_server") as mock_notify:
                    with patch("pixie.session.client.disconnect_from_server"):
                        with patch("pixie.session.client.enable_instrumentations"):
                            with patch(
                                "pixie.session.client.get_client"
                            ) as mock_get_client:
                                mock_connect.return_value = 11111
                                mock_notify.return_value = None
                                mock_langfuse = MagicMock()
                                mock_langfuse.auth_check.return_value = True
                                mock_langfuse.flush = MagicMock()
                                mock_get_client.return_value = mock_langfuse

                                mock_queue = MagicMock()
                                mock_queue.async_q = MagicMock()
                                mock_queue.async_q.get = AsyncMock(return_value=None)

                                mock_exec_ctx = MagicMock()
                                mock_exec_ctx.status_queue = mock_queue
                                mock_ctx.init_run.return_value = mock_exec_ctx

                                @session
                                async def my_func():
                                    return "result"

                                result = await my_func()

                                assert result == "result"
                                mock_ctx.init_run.assert_called_once()
                                mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_decorator_sends_running_status(self):
        """Test that @session sends running status on start."""
        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.connect_to_server") as mock_connect:
                with patch("pixie.session.client.notify_server") as mock_notify:
                    with patch("pixie.session.client.disconnect_from_server"):
                        with patch("pixie.session.client.enable_instrumentations"):
                            with patch(
                                "pixie.session.client.get_client"
                            ) as mock_get_client:
                                mock_connect.return_value = 11111
                                mock_notify.return_value = None
                                mock_langfuse = MagicMock()
                                mock_langfuse.auth_check.return_value = True
                                mock_langfuse.flush = MagicMock()
                                mock_get_client.return_value = mock_langfuse

                                mock_queue = MagicMock()
                                mock_queue.async_q = MagicMock()
                                mock_queue.async_q.get = AsyncMock(return_value=None)

                                mock_exec_ctx = MagicMock()
                                mock_exec_ctx.status_queue = mock_queue
                                mock_ctx.init_run.return_value = mock_exec_ctx

                                @session
                                async def my_func():
                                    pass

                                await my_func()

                                calls = mock_notify.call_args_list
                                running_calls = [
                                    c for c in calls if c[0][0].status == "running"
                                ]
                                assert len(running_calls) >= 1

    @pytest.mark.asyncio
    async def test_session_decorator_sends_completed_status(self):
        """Test that @session sends completed status on finish."""
        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.connect_to_server") as mock_connect:
                with patch("pixie.session.client.notify_server") as mock_notify:
                    with patch("pixie.session.client.disconnect_from_server"):
                        with patch("pixie.session.client.enable_instrumentations"):
                            with patch(
                                "pixie.session.client.get_client"
                            ) as mock_get_client:
                                mock_connect.return_value = 11111
                                mock_notify.return_value = None
                                mock_langfuse = MagicMock()
                                mock_langfuse.auth_check.return_value = True
                                mock_langfuse.flush = MagicMock()
                                mock_get_client.return_value = mock_langfuse

                                mock_queue = MagicMock()
                                mock_queue.async_q = MagicMock()
                                mock_queue.async_q.get = AsyncMock(return_value=None)

                                mock_exec_ctx = MagicMock()
                                mock_exec_ctx.status_queue = mock_queue
                                mock_ctx.init_run.return_value = mock_exec_ctx

                                @session
                                async def my_func():
                                    pass

                                await my_func()

                                calls = mock_notify.call_args_list
                                completed_calls = [
                                    c for c in calls if c[0][0].status == "completed"
                                ]
                                assert len(completed_calls) >= 1

    @pytest.mark.asyncio
    async def test_session_decorator_calls_enable_instrumentations(self):
        """Test that @session calls enable_instrumentations."""
        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.connect_to_server") as mock_connect:
                with patch("pixie.session.client.notify_server") as mock_notify:
                    with patch("pixie.session.client.disconnect_from_server"):
                        with patch(
                            "pixie.session.client.enable_instrumentations"
                        ) as mock_enable:
                            with patch(
                                "pixie.session.client.get_client"
                            ) as mock_get_client:
                                mock_connect.return_value = 11111
                                mock_notify.return_value = None
                                mock_langfuse = MagicMock()
                                mock_langfuse.auth_check.return_value = True
                                mock_langfuse.flush = MagicMock()
                                mock_get_client.return_value = mock_langfuse

                                mock_queue = MagicMock()
                                mock_queue.async_q = MagicMock()
                                mock_queue.async_q.get = AsyncMock(return_value=None)

                                mock_exec_ctx = MagicMock()
                                mock_exec_ctx.status_queue = mock_queue
                                mock_ctx.init_run.return_value = mock_exec_ctx

                                @session
                                async def my_func():
                                    pass

                                await my_func()

                                mock_enable.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_decorator_calls_get_client(self):
        """Test that @session calls get_client."""
        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.connect_to_server") as mock_connect:
                with patch("pixie.session.client.notify_server") as mock_notify:
                    with patch("pixie.session.client.disconnect_from_server"):
                        with patch("pixie.session.client.enable_instrumentations"):
                            with patch(
                                "pixie.session.client.get_client"
                            ) as mock_get_client:
                                mock_connect.return_value = 11111
                                mock_notify.return_value = None
                                mock_langfuse = MagicMock()
                                mock_langfuse.auth_check.return_value = True
                                mock_get_client.return_value = mock_langfuse

                                mock_queue = MagicMock()
                                mock_queue.async_q = MagicMock()
                                mock_queue.async_q.get = AsyncMock(return_value=None)

                                mock_exec_ctx = MagicMock()
                                mock_exec_ctx.status_queue = mock_queue
                                mock_ctx.init_run.return_value = mock_exec_ctx

                                @session
                                async def my_func():
                                    pass

                                await my_func()

                                mock_get_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_decorator_supports_async_generator(self):
        """Test that @session supports async generator functions."""
        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.connect_to_server") as mock_connect:
                with patch("pixie.session.client.notify_server") as mock_notify:
                    with patch("pixie.session.client.disconnect_from_server"):
                        with patch("pixie.session.client.enable_instrumentations"):
                            with patch(
                                "pixie.session.client.get_client"
                            ) as mock_get_client:
                                mock_connect.return_value = 11111
                                mock_notify.return_value = None
                                mock_langfuse = MagicMock()
                                mock_langfuse.auth_check.return_value = True
                                mock_langfuse.flush = MagicMock()
                                mock_get_client.return_value = mock_langfuse

                                mock_queue = MagicMock()
                                mock_queue.async_q = MagicMock()
                                mock_queue.async_q.get = AsyncMock(return_value=None)

                                mock_exec_ctx = MagicMock()
                                mock_exec_ctx.status_queue = mock_queue
                                mock_ctx.init_run.return_value = mock_exec_ctx

                                @session
                                async def my_generator():
                                    yield "item1"
                                    yield "item2"
                                    yield "item3"

                                result = []
                                async for item in my_generator():
                                    result.append(item)

                                assert result == ["item1", "item2", "item3"]
                                mock_ctx.init_run.assert_called_once()
                                mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_decorator_async_generator_sends_completed(self):
        """Test that @session sends completed status after generator exhausted."""
        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.connect_to_server") as mock_connect:
                with patch("pixie.session.client.notify_server") as mock_notify:
                    with patch("pixie.session.client.disconnect_from_server"):
                        with patch("pixie.session.client.enable_instrumentations"):
                            with patch(
                                "pixie.session.client.get_client"
                            ) as mock_get_client:
                                mock_connect.return_value = 11111
                                mock_notify.return_value = None
                                mock_langfuse = MagicMock()
                                mock_langfuse.auth_check.return_value = True
                                mock_langfuse.flush = MagicMock()
                                mock_get_client.return_value = mock_langfuse

                                mock_queue = MagicMock()
                                mock_queue.async_q = MagicMock()
                                mock_queue.async_q.get = AsyncMock(return_value=None)

                                mock_exec_ctx = MagicMock()
                                mock_exec_ctx.status_queue = mock_queue
                                mock_ctx.init_run.return_value = mock_exec_ctx

                                @session
                                async def my_generator():
                                    yield "data"

                                async for _ in my_generator():
                                    pass

                                calls = mock_notify.call_args_list
                                completed_calls = [
                                    c for c in calls if c[0][0].status == "completed"
                                ]
                                assert len(completed_calls) >= 1

    @pytest.mark.asyncio
    async def test_session_decorator_async_generator_cleanup_on_early_exit(self):
        """Test that @session properly cleans up when generator exits early."""
        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.connect_to_server") as mock_connect:
                with patch("pixie.session.client.notify_server") as mock_notify:
                    with patch(
                        "pixie.session.client.disconnect_from_server"
                    ) as mock_disconnect:
                        with patch("pixie.session.client.enable_instrumentations"):
                            with patch(
                                "pixie.session.client.get_client"
                            ) as mock_get_client:
                                mock_connect.return_value = 11111
                                mock_notify.return_value = None
                                mock_langfuse = MagicMock()
                                mock_langfuse.auth_check.return_value = True
                                mock_langfuse.flush = MagicMock()
                                mock_get_client.return_value = mock_langfuse

                                mock_queue = MagicMock()
                                mock_queue.async_q = MagicMock()
                                mock_queue.async_q.get = AsyncMock(return_value=None)

                                mock_exec_ctx = MagicMock()
                                mock_exec_ctx.status_queue = mock_queue
                                mock_ctx.init_run.return_value = mock_exec_ctx

                                @session
                                async def my_generator():
                                    yield "item1"
                                    yield "item2"
                                    yield "item3"

                                # Exit early after first item using async with to ensure cleanup
                                gen = my_generator()
                                try:
                                    item = await gen.__anext__()
                                    assert item == "item1"
                                finally:
                                    await gen.aclose()  # Explicitly close the generator

                                # Should disconnect after explicit close
                                mock_disconnect.assert_called_once()


class TestSessionObservationLogic:
    """Test the newly added observation enter/exit logic for session."""

    @pytest.mark.asyncio
    @patch("pixie.session.client._langfuse")
    async def test_session_observation_started_and_ended(self, mock_langfuse):
        """Test that observations are properly started and ended for session functions."""
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=None)
        mock_span.__exit__ = MagicMock(return_value=None)
        mock_langfuse.start_as_current_observation.return_value = mock_span

        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.connect_to_server") as mock_connect:
                with patch("pixie.session.client.notify_server") as mock_notify:
                    with patch("pixie.session.client.disconnect_from_server"):
                        with patch("pixie.session.client.enable_instrumentations"):
                            with patch(
                                "pixie.session.client.get_client"
                            ) as mock_get_client:
                                mock_connect.return_value = 11111
                                mock_notify.return_value = None
                                mock_langfuse_client = MagicMock()
                                mock_langfuse_client.auth_check.return_value = True
                                mock_langfuse_client.flush = MagicMock()
                                mock_get_client.return_value = mock_langfuse_client

                                mock_queue = MagicMock()
                                mock_queue.async_q = MagicMock()
                                mock_queue.async_q.get = AsyncMock(return_value=None)

                                mock_exec_ctx = MagicMock()
                                mock_exec_ctx.status_queue = mock_queue
                                mock_ctx.init_run.return_value = mock_exec_ctx

                                @session
                                async def simple_session():
                                    return {"result": "success"}

                                result = await simple_session()

                                # Verify results
                                assert result == {"result": "success"}

                                # Verify observation was started with correct parameters
                                mock_langfuse.start_as_current_observation.assert_called_once_with(
                                    name="simple_session", as_type="chain"
                                )

                                # Verify span was entered and exited properly
                                mock_span.__enter__.assert_called_once()
                                mock_span.__exit__.assert_called_once_with(
                                    None, None, None
                                )

    @pytest.mark.asyncio
    @patch("pixie.session.client._langfuse")
    async def test_session_observation_exception_handling(self, mock_langfuse):
        """Test that exceptions in session functions properly exit observations."""
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=None)
        mock_span.__exit__ = MagicMock(return_value=None)
        mock_langfuse.start_as_current_observation.return_value = mock_span

        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.connect_to_server") as mock_connect:
                with patch("pixie.session.client.notify_server") as mock_notify:
                    with patch("pixie.session.client.disconnect_from_server"):
                        with patch("pixie.session.client.enable_instrumentations"):
                            with patch(
                                "pixie.session.client.get_client"
                            ) as mock_get_client:
                                mock_connect.return_value = 11111
                                mock_notify.return_value = None
                                mock_langfuse_client = MagicMock()
                                mock_langfuse_client.auth_check.return_value = True
                                mock_langfuse_client.flush = MagicMock()
                                mock_get_client.return_value = mock_langfuse_client

                                mock_queue = MagicMock()
                                mock_queue.async_q = MagicMock()
                                mock_queue.async_q.get = AsyncMock(return_value=None)

                                mock_exec_ctx = MagicMock()
                                mock_exec_ctx.status_queue = mock_queue
                                mock_ctx.init_run.return_value = mock_exec_ctx

                                @session
                                async def failing_session():
                                    raise ValueError("Test error")

                                # Call the session and expect exception
                                with pytest.raises(ValueError, match="Test error"):
                                    await failing_session()

                                # Verify observation was started
                                mock_langfuse.start_as_current_observation.assert_called_once_with(
                                    name="failing_session", as_type="chain"
                                )

                                # Verify span was entered
                                mock_span.__enter__.assert_called_once()

                                # Verify span was exited with exception info
                                mock_span.__exit__.assert_called_once()
                                call_args = mock_span.__exit__.call_args
                                assert call_args[0][0] == ValueError  # exc_type
                                assert isinstance(
                                    call_args[0][1], ValueError
                                )  # exc_value
                                assert call_args[0][2] is not None  # traceback

    @pytest.mark.asyncio
    @patch("pixie.session.client._langfuse")
    async def test_session_generator_observation_started_and_ended(self, mock_langfuse):
        """Test that observations are properly started and ended for session generators."""
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=None)
        mock_span.__exit__ = MagicMock(return_value=None)
        mock_langfuse.start_as_current_observation.return_value = mock_span

        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.connect_to_server") as mock_connect:
                with patch("pixie.session.client.notify_server") as mock_notify:
                    with patch("pixie.session.client.disconnect_from_server"):
                        with patch("pixie.session.client.enable_instrumentations"):
                            with patch(
                                "pixie.session.client.get_client"
                            ) as mock_get_client:
                                mock_connect.return_value = 11111
                                mock_notify.return_value = None
                                mock_langfuse_client = MagicMock()
                                mock_langfuse_client.auth_check.return_value = True
                                mock_langfuse_client.flush = MagicMock()
                                mock_get_client.return_value = mock_langfuse_client

                                mock_queue = MagicMock()
                                mock_queue.async_q = MagicMock()
                                mock_queue.async_q.get = AsyncMock(return_value=None)

                                mock_exec_ctx = MagicMock()
                                mock_exec_ctx.status_queue = mock_queue
                                mock_ctx.init_run.return_value = mock_exec_ctx

                                @session
                                async def simple_generator():
                                    yield {"step": 1}
                                    yield {"step": 2}

                                results = []
                                async for item in simple_generator():
                                    results.append(item)

                                # Verify results
                                assert len(results) == 2
                                assert results[0] == {"step": 1}
                                assert results[1] == {"step": 2}

                                # Verify observation was started with correct parameters
                                mock_langfuse.start_as_current_observation.assert_called_once_with(
                                    name="simple_generator", as_type="chain"
                                )

                                # Verify span was entered and exited properly
                                mock_span.__enter__.assert_called_once()
                                mock_span.__exit__.assert_called_once_with(
                                    None, None, None
                                )

    @pytest.mark.asyncio
    @patch("pixie.session.client._langfuse")
    async def test_session_generator_observation_exception_handling(
        self, mock_langfuse
    ):
        """Test that exceptions in session generators properly exit observations."""
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=None)
        mock_span.__exit__ = MagicMock(return_value=None)
        mock_langfuse.start_as_current_observation.return_value = mock_span

        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.connect_to_server") as mock_connect:
                with patch("pixie.session.client.notify_server") as mock_notify:
                    with patch("pixie.session.client.disconnect_from_server"):
                        with patch("pixie.session.client.enable_instrumentations"):
                            with patch(
                                "pixie.session.client.get_client"
                            ) as mock_get_client:
                                mock_connect.return_value = 11111
                                mock_notify.return_value = None
                                mock_langfuse_client = MagicMock()
                                mock_langfuse_client.auth_check.return_value = True
                                mock_langfuse_client.flush = MagicMock()
                                mock_get_client.return_value = mock_langfuse_client

                                mock_queue = MagicMock()
                                mock_queue.async_q = MagicMock()
                                mock_queue.async_q.get = AsyncMock(return_value=None)

                                mock_exec_ctx = MagicMock()
                                mock_exec_ctx.status_queue = mock_queue
                                mock_ctx.init_run.return_value = mock_exec_ctx

                                @session
                                async def failing_generator():
                                    yield {"step": 1}
                                    raise ValueError("Generator error")

                                # Call the generator and expect exception
                                with pytest.raises(ValueError, match="Generator error"):
                                    async for _ in failing_generator():
                                        pass

                                # Verify observation was started
                                mock_langfuse.start_as_current_observation.assert_called_once_with(
                                    name="failing_generator", as_type="chain"
                                )

                                # Verify span was entered
                                mock_span.__enter__.assert_called_once()

                                # Verify span was exited with exception info
                                mock_span.__exit__.assert_called_once()
                                call_args = mock_span.__exit__.call_args
                                assert call_args[0][0] == ValueError  # exc_type
                                assert isinstance(
                                    call_args[0][1], ValueError
                                )  # exc_value
                                assert call_args[0][2] is not None  # traceback

    @pytest.mark.asyncio
    @patch("pixie.session.client._langfuse")
    async def test_input_observation_for_waiting(self, mock_langfuse):
        """Test that observations are created when waiting for input."""
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=None)
        mock_span.__exit__ = MagicMock(return_value=None)
        mock_langfuse.start_as_current_observation.return_value = mock_span

        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.notify_server") as mock_notify:
                with patch("pixie.session.client.wait_for_input") as mock_wait:
                    mock_notify.return_value = None
                    mock_wait.return_value = "user response"

                    mock_exec_ctx = MagicMock()
                    mock_exec_ctx.run_id = "test-session"
                    mock_ctx.get_current_context.return_value = mock_exec_ctx

                    result = await session_input("Enter something:", expected_type=str)

                    # Verify result
                    assert result == "user response"

                    # Verify observation was created for waiting
                    mock_langfuse.start_as_current_observation.assert_called_once_with(
                        name="wait_for_input", as_type="tool"
                    )

                    # Verify span was used as context manager
                    mock_span.__enter__.assert_called_once()
                    mock_span.__exit__.assert_called_once()
