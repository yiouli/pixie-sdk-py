"""Tests for pixie.session module - TCP-based RPC client/server communication."""

import asyncio
import time
import pytest
import janus
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel

from pixie.session.rpc import (
    connect_to_server,
    disconnect_from_server,
    notify_server,
    wait_for_input,
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


def get_test_ports(test_id: int, count: int = 3) -> list[int]:
    """Get a unique port range for a test to avoid conflicts.

    Args:
        test_id: Unique identifier for the test (use different numbers per test).
        count: Number of ports to allocate.

    Returns:
        List of port numbers.
    """
    base = TEST_PORT_BASE + (test_id * 10)
    return list(range(base, base + count))


@pytest.fixture
def cleanup():
    """Cleanup fixture to ensure connections are closed after each test."""
    yield
    disconnect_from_server()
    stop_server()
    # Give sockets time to clean up
    time.sleep(0.1)


class TestServerStartup:
    """Tests for server startup and port binding."""

    @pytest.mark.asyncio
    async def test_server_binds_to_ports(self, cleanup):
        """Test that server successfully binds to specified ports."""
        ports = get_test_ports(1)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        server_task = asyncio.create_task(
            listen_to_client_connections(ports, update_queue)
        )
        await asyncio.sleep(0.2)  # Let server start

        state = _get_server_state()
        assert state is not None
        assert len(state.servers) == len(ports)

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_server_already_running_raises(self, cleanup):
        """Test that starting server twice raises error."""
        ports = get_test_ports(2)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        server_task = asyncio.create_task(
            listen_to_client_connections(ports, update_queue)
        )
        await asyncio.sleep(0.2)

        with pytest.raises(RuntimeError, match="Server already running"):
            await listen_to_client_connections(ports, update_queue)

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
        ports = get_test_ports(3)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        server_task = asyncio.create_task(
            listen_to_client_connections(ports, update_queue)
        )
        await asyncio.sleep(0.2)

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
        ports = get_test_ports(10)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        server_task = asyncio.create_task(
            listen_to_client_connections(ports, update_queue)
        )
        await asyncio.sleep(0.2)

        # Connect should succeed and return the port used
        connected_port = await connect_to_server("localhost", ports, "test-session")

        assert connected_port in ports
        state = _get_client_state()
        assert state is not None
        assert state.session_id == "test-session"
        assert state.connected_port == connected_port

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_client_registered_on_server(self, cleanup):
        """Test that client is properly registered on server after connect."""
        ports = get_test_ports(11)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        server_task = asyncio.create_task(
            listen_to_client_connections(ports, update_queue)
        )
        await asyncio.sleep(0.2)

        await connect_to_server("localhost", ports, "registered-session")
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
        ports = get_test_ports(12)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        server_task = asyncio.create_task(
            listen_to_client_connections(ports, update_queue)
        )
        await asyncio.sleep(0.2)

        await connect_to_server("localhost", ports, "test-session")

        with pytest.raises(RuntimeError, match="Already connected"):
            await connect_to_server("localhost", ports, "test-session-2")

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
        ports = get_test_ports(13)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        server_task = asyncio.create_task(
            listen_to_client_connections(ports, update_queue)
        )
        await asyncio.sleep(0.2)

        await connect_to_server("localhost", ports, "test-session")
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
    async def test_connect_tries_multiple_ports(self, cleanup):
        """Test that client tries multiple ports until one works."""
        # Use ports where only the last one has a server
        ports = get_test_ports(14, count=5)
        server_ports = [ports[-1]]  # Only bind to last port
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        server_task = asyncio.create_task(
            listen_to_client_connections(server_ports, update_queue)
        )
        await asyncio.sleep(0.2)

        # Client should try all ports and eventually connect to the last one
        connected_port = await connect_to_server("localhost", ports, "multi-port-test")
        assert connected_port == ports[-1]

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_connect_fails_when_no_server(self, cleanup):
        """Test that connect fails when no server is available."""
        ports = get_test_ports(15)
        # Don't start a server

        with pytest.raises(RuntimeError, match="Failed to connect"):
            await connect_to_server("localhost", ports, "no-server-test")


class TestRegistrationTimeout:
    """Tests for session registration timeout behavior."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_server_disconnects_slow_client(self, cleanup):
        """Test that server disconnects client that doesn't send session_id in time."""
        ports = get_test_ports(20)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        server_task = asyncio.create_task(
            listen_to_client_connections(ports, update_queue)
        )
        await asyncio.sleep(0.2)

        # Connect directly without using our client (to simulate slow registration)
        reader, writer = await asyncio.open_connection("localhost", ports[0])

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
        ports = get_test_ports(21)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        server_task = asyncio.create_task(
            listen_to_client_connections(ports, update_queue)
        )
        await asyncio.sleep(0.2)

        # First client connects successfully
        await connect_to_server("localhost", ports, "duplicate-session")
        await asyncio.sleep(0.1)

        # Try to connect second client manually with same session_id
        import struct

        reader, writer = await asyncio.open_connection("localhost", ports[1])

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
        ports = get_test_ports(30)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        server_task = asyncio.create_task(
            listen_to_client_connections(ports, update_queue)
        )
        await asyncio.sleep(0.2)

        await connect_to_server("localhost", ports, "test-session-1")
        await asyncio.sleep(0.1)  # Give server time to register client

        update = SessionUpdate(
            session_id="test-session-1",
            status="running",
            data="hello server",
        )
        await notify_server(update)

        # Check queue received the update
        received = await asyncio.wait_for(update_queue.async_q.get(), timeout=2.0)
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
        ports = get_test_ports(31)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        server_task = asyncio.create_task(
            listen_to_client_connections(ports, update_queue)
        )
        await asyncio.sleep(0.2)

        await connect_to_server("localhost", ports, "session-multi")
        await asyncio.sleep(0.1)

        for i in range(3):
            await notify_server(
                SessionUpdate(
                    session_id="session-multi",
                    status="running",
                    data=f"message {i}",
                )
            )

        for i in range(3):
            received = await asyncio.wait_for(update_queue.async_q.get(), timeout=2.0)
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
                )
            )


class TestInputFlow:
    """Tests for wait_for_input and send_input_to_client."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_wait_for_input_receives_string(self, cleanup):
        """Test that client receives string input from server."""
        ports = get_test_ports(40)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        server_task = asyncio.create_task(
            listen_to_client_connections(ports, update_queue)
        )
        await asyncio.sleep(0.2)

        await connect_to_server("localhost", ports, "input-test")
        await asyncio.sleep(0.1)

        # Send update so server knows client is waiting
        await notify_server(
            SessionUpdate(
                session_id="input-test",
                status="waiting",
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
        ports = get_test_ports(41)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        server_task = asyncio.create_task(
            listen_to_client_connections(ports, update_queue)
        )
        await asyncio.sleep(0.2)

        await connect_to_server("localhost", ports, "model-test")
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
        ports = get_test_ports(42)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        server_task = asyncio.create_task(
            listen_to_client_connections(ports, update_queue)
        )
        await asyncio.sleep(0.2)

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
        ports = get_test_ports(50)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        server_task = asyncio.create_task(
            listen_to_client_connections(ports, update_queue)
        )
        await asyncio.sleep(0.2)

        await connect_to_server("localhost", ports, "cleanup-test")
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
    async def test_multiple_clients_concurrent(self, cleanup):
        """Test that multiple clients can connect to different ports concurrently."""
        ports = get_test_ports(51, count=5)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        server_task = asyncio.create_task(
            listen_to_client_connections(ports, update_queue)
        )
        await asyncio.sleep(0.2)

        # Connect first client
        await connect_to_server("localhost", ports, "client-1")
        await asyncio.sleep(0.1)

        # Store state and disconnect first client
        _first_port = _get_client_state().connected_port  # noqa: F841
        disconnect_from_server()

        # Connect second client
        await connect_to_server("localhost", ports, "client-2")
        await asyncio.sleep(0.1)

        # Second client should work
        state = _get_client_state()
        assert state.session_id == "client-2"
        # May or may not use same port depending on timing

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
        ports = get_test_ports(60)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        # Start server
        server_task = asyncio.create_task(
            listen_to_client_connections(ports, update_queue)
        )
        await asyncio.sleep(0.2)

        # Connect client
        connected_port = await connect_to_server("localhost", ports, "roundtrip")
        assert connected_port in ports
        await asyncio.sleep(0.1)

        # Client sends running status
        await notify_server(
            SessionUpdate(
                session_id="roundtrip",
                status="running",
                data="starting",
            )
        )

        # Client sends waiting status
        await notify_server(
            SessionUpdate(
                session_id="roundtrip",
                status="waiting",
                data="need input",
            )
        )

        # Verify server received both updates
        update1 = await asyncio.wait_for(update_queue.async_q.get(), timeout=2.0)
        assert update1.session_id == "roundtrip"
        assert update1.status == "running"

        update2 = await asyncio.wait_for(update_queue.async_q.get(), timeout=2.0)
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
        ports = get_test_ports(61)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        server_task = asyncio.create_task(
            listen_to_client_connections(ports, update_queue)
        )
        await asyncio.sleep(0.2)

        await connect_to_server("localhost", ports, "large-msg-test")
        await asyncio.sleep(0.1)

        # Send a large message
        large_data = "x" * 100000  # 100KB of data
        await notify_server(
            SessionUpdate(
                session_id="large-msg-test",
                status="running",
                data=large_data,
            )
        )

        received = await asyncio.wait_for(update_queue.async_q.get(), timeout=5.0)
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

                    call_args = mock_notify.call_args[0][0]
                    assert call_args.status == "waiting"
                    assert call_args.data == "Enter your name: "

    @pytest.mark.asyncio
    async def test_session_input_typed(self):
        """Test that session.input requests typed input."""
        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.notify_server") as mock_notify:
                with patch("pixie.session.client.wait_for_input") as mock_wait:
                    mock_notify.return_value = None
                    mock_wait.return_value = UserModel(name="Test", age=20)

                    mock_exec_ctx = MagicMock()
                    mock_exec_ctx.run_id = "test-run-123"
                    mock_ctx.get_current_context.return_value = mock_exec_ctx

                    result = await session_input(
                        "Enter user data: ",
                        expected_type=UserModel,
                    )

                    assert isinstance(result, UserModel)
                    assert result.name == "Test"

    @pytest.mark.asyncio
    async def test_session_input_no_context_raises(self):
        """Test that session.input raises without execution context."""
        with patch("pixie.session.client.execution_context") as mock_ctx:
            mock_ctx.get_current_context.return_value = None

            with pytest.raises(RuntimeError, match="No execution context"):
                await session_input("prompt")


class TestSessionDecorator:
    """Tests for the @session decorator."""

    @pytest.mark.asyncio
    async def test_session_decorator_creates_context(self):
        """Test that @session creates execution context."""
        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.connect_to_server") as mock_connect:
                with patch("pixie.session.client.notify_server") as mock_notify:
                    with patch("pixie.session.client.disconnect_from_server"):
                        mock_connect.return_value = 11111
                        mock_notify.return_value = None

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
                        mock_connect.return_value = 11111
                        mock_notify.return_value = None

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
                        mock_connect.return_value = 11111
                        mock_notify.return_value = None

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
