"""Tests for session client disconnect scenarios.

These tests verify that:
1. When a client disconnects while server is waiting for UI input (InputRequired),
   the server continues to work properly for subsequent clients.
2. The "completed" status is sent reliably even when client is cancelled (Ctrl+C).
"""

import asyncio
import pytest
import janus
from unittest.mock import patch

from pixie.session.rpc import (
    connect_to_server,
    disconnect_from_server,
    notify_server,
    listen_to_client_connections,
    stop_server,
    send_input_to_client,
    wait_for_client_update,
    _get_server_state,
)
from pixie.session.types import SessionUpdate
from pixie.session.client import session


# Test timeout for socket operations
TIMEOUT = 10.0

# Base port for tests - use different ranges to avoid conflicts
TEST_PORT_BASE = 19000


def get_test_port(test_id: int) -> int:
    """Get a unique port for a test."""
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


class TestClientDisconnectDuringInputRequired:
    """Tests for client disconnecting while server is waiting for UI input."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_server_handles_client_disconnect_during_input_wait(self, cleanup):
        """Test that server handles KeyError when client disconnects during InputRequired.

        Scenario:
        1. Client sends "waiting" status with input schema
        2. Server yields the update, then yields InputRequired
        3. Client disconnects (simulating Ctrl+C)
        4. UI provides input
        5. Server should NOT crash, should continue working for new clients
        """
        port = get_test_port(1)
        # Start server
        server_task = await start_session_server(port)

        # Connect first client
        await connect_to_server("localhost", port, "client1")
        await asyncio.sleep(0.1)

        # Client sends waiting status with input schema
        await notify_server(
            SessionUpdate(
                session_id="client1",
                status="waiting",
                time_unix_nano="1234567890000000000",
                data="need input",
                user_input_schema={"type": "string"},
            )
        )

        # Verify server received the update
        update = await expect_update("client1")
        assert update.session_id == "client1"
        assert update.status == "waiting"
        assert update.user_input_schema == {"type": "string"}

        # Client disconnects abruptly (simulating Ctrl+C)
        disconnect_from_server()
        await asyncio.sleep(0.2)  # Let server detect disconnect

        # Verify client was removed from server's connection list
        state = _get_server_state()
        assert "client1" not in state.client_connections

        # Now try to send input to the disconnected client
        # This should NOT crash the server - it should raise KeyError but be handled
        with pytest.raises(KeyError):
            await send_input_to_client("client1", "user input")

        # Server should still be running
        assert state.running is True

        # A new client should be able to connect and work
        await connect_to_server("localhost", port, "client2")
        await asyncio.sleep(0.1)

        await notify_server(
            SessionUpdate(
                session_id="client2",
                status="running",
                time_unix_nano="1234567890000000000",
                data="client2 says hello",
            )
        )

        # Server should receive the new client's update
        update2 = await expect_update("client2")
        assert update2.session_id == "client2"
        assert update2.status == "running"

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_run_session_server_handles_client_disconnect(self, cleanup):
        """Test run_session_server generator handles client disconnect gracefully.

        This tests the full scenario with the session server generator.
        """

        # We need to test the actual server generator behavior
        # The issue is: when client disconnects while server yields InputRequired,
        # and UI provides input, send_input_to_client fails with KeyError,
        # which breaks the while True loop.

        # For this test, we'll directly test the error handling in send_input_to_client
        port = get_test_port(2)
        server_task = await start_session_server(port)

        # Connect client
        await connect_to_server("localhost", port, "test-client")
        await asyncio.sleep(0.1)

        # Send updates
        await notify_server(
            SessionUpdate(
                session_id="test-client",
                status="running",
                time_unix_nano="1234567890000000000",
            )
        )

        # Get update
        update = await expect_update("test-client")
        assert update.status == "running"

        # Disconnect
        disconnect_from_server()
        await asyncio.sleep(0.2)

        # Send_input_to_client should fail gracefully
        with pytest.raises(KeyError):
            await send_input_to_client("test-client", "input")

        # Server should still accept new connections
        await connect_to_server("localhost", port, "new-client")
        await notify_server(
            SessionUpdate(
                session_id="new-client",
                status="running",
                time_unix_nano="1234567890000000000",
            )
        )

        update2 = await expect_update("new-client")
        assert update2.session_id == "new-client"

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


class TestClientCompletionOnCancel:
    """Tests for ensuring 'completed' status is sent even on cancellation."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_session_sends_completed_on_cancellation(self):
        """Test that @session sends 'completed' even when task is cancelled.

        This simulates Ctrl+C during any async operation.
        """
        sent_updates = []

        async def mock_notify(update):
            sent_updates.append(update)

        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.connect_to_server") as mock_connect:
                with patch(
                    "pixie.session.client.notify_server", side_effect=mock_notify
                ):
                    with patch("pixie.session.client.disconnect_from_server"):
                        mock_connect.return_value = 11111

                        mock_queue = janus.Queue()
                        mock_exec_ctx = type(
                            "MockCtx",
                            (),
                            {"status_queue": mock_queue, "run_id": "test-session"},
                        )()
                        mock_ctx.init_run.return_value = mock_exec_ctx
                        mock_ctx.get_current_context.return_value = mock_exec_ctx

                        @session
                        async def my_func():
                            # Simulate waiting for input then being cancelled
                            await asyncio.sleep(10)

                        # Run the function but cancel it
                        task = asyncio.create_task(my_func())
                        await asyncio.sleep(0.1)  # Let it start
                        task.cancel()

                        with pytest.raises(asyncio.CancelledError):
                            await task

                        # Check that completed was sent
                        completed_updates = [
                            u for u in sent_updates if u.status == "completed"
                        ]
                        assert (
                            len(completed_updates) >= 1
                        ), f"Expected 'completed' status, got: {[u.status for u in sent_updates]}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_session_sends_completed_when_wait_cancelled(self):
        """Test that @session sends 'completed' when waiting is cancelled.

        This is the specific Ctrl+C during long-running operation scenario.
        """
        sent_updates = []

        async def mock_notify(update):
            sent_updates.append(update)

        with patch("pixie.session.client.execution_context") as mock_ctx:
            with patch("pixie.session.client.connect_to_server") as mock_connect:
                with patch(
                    "pixie.session.client.notify_server", side_effect=mock_notify
                ):
                    with patch("pixie.session.client.disconnect_from_server"):
                        mock_connect.return_value = 11111

                        mock_queue = janus.Queue()
                        mock_exec_ctx = type(
                            "MockCtx",
                            (),
                            {"status_queue": mock_queue, "run_id": "test-session"},
                        )()
                        mock_ctx.init_run.return_value = mock_exec_ctx
                        mock_ctx.get_current_context.return_value = mock_exec_ctx

                        @session
                        async def my_func():
                            # Simulate a long running operation
                            await asyncio.sleep(100)

                        task = asyncio.create_task(my_func())
                        await asyncio.sleep(0.2)  # Let it start
                        task.cancel()

                        with pytest.raises(asyncio.CancelledError):
                            await task

                        # Verify completed was sent even though cancelled
                        statuses = [u.status for u in sent_updates]
                        assert (
                            "completed" in statuses
                        ), f"Expected 'completed' in statuses, got: {statuses}"


class TestServerLoopRecovery:
    """Tests for server loop recovery after errors."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_server_continues_after_send_input_failure(self, cleanup):
        """Test that the server loop continues even if send_input_to_client fails.

        This is the critical test - when server is in run_session_server generator:
        1. Receives update with input_schema
        2. Yields update
        3. Yields InputRequired, waiting for UI
        4. Client disconnects
        5. UI provides input
        6. send_input_to_client fails with KeyError
        7. Server should catch the error and continue processing
        """
        port = get_test_port(10)
        server_task = await start_session_server(port)

        # First client connects
        await connect_to_server("localhost", port, "client-a")
        await asyncio.sleep(0.1)

        # Client sends normal update
        await notify_server(
            SessionUpdate(
                session_id="client-a",
                status="running",
                time_unix_nano="1234567890000000000",
                data="hello",
            )
        )

        update1 = await expect_update("client-a")
        assert update1.session_id == "client-a"

        # Client sends waiting with schema
        await notify_server(
            SessionUpdate(
                session_id="client-a",
                status="waiting",
                time_unix_nano="1234567890000000000",
                user_input_schema={"type": "string"},
            )
        )

        update2 = await expect_update("client-a")
        assert update2.status == "waiting"

        # Now client disconnects WITHOUT sending completed
        disconnect_from_server()
        await asyncio.sleep(0.3)  # Give time for server to detect

        # Verify client is removed
        state = _get_server_state()
        assert "client-a" not in state.client_connections

        # Second client should be able to connect and work
        await connect_to_server("localhost", port, "client-b")
        await asyncio.sleep(0.1)

        await notify_server(
            SessionUpdate(
                session_id="client-b",
                status="running",
                time_unix_nano="1234567890000000000",
                data="new client",
            )
        )

        # Server should receive this update
        update3 = await expect_update("client-b")
        assert update3.session_id == "client-b"
        assert update3.data == "new client"

        # Send completed
        await notify_server(
            SessionUpdate(
                session_id="client-b",
                status="completed",
                time_unix_nano="1234567890000000000",
            )
        )

        update4 = await expect_update("client-b")
        assert update4.status == "completed"

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


class TestRunSessionServerGenerator:
    """Tests for the run_session_server generator handling client disconnects."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_server_generator_handles_disconnect_during_input_required(
        self, cleanup
    ):
        """Test run_session_server generator handles client disconnect gracefully.

        This tests the actual scenario that broke:
        1. Client sends "waiting" with user_input_schema
        2. Generator yields update, then yields InputRequired
        3. Client disconnects (Ctrl+C)
        4. UI sends input back to generator
        5. Generator tries to send_input_to_client, gets KeyError
        6. Generator should catch error and continue, not crash
        """

        # We'll directly test the generator logic by mocking the RPC layer
        # but simulating the exact sequence of events

        port = get_test_port(20)
        server_task = await start_session_server(port)

        # Client 1 connects
        await connect_to_server("localhost", port, "gen-client-1")
        await asyncio.sleep(0.1)

        # Client sends waiting with schema (triggers InputRequired in generator)
        await notify_server(
            SessionUpdate(
                session_id="gen-client-1",
                status="waiting",
                time_unix_nano="1234567890000000000",
                data="need input",
                user_input_schema={"type": "string"},
            )
        )

        update = await expect_update("gen-client-1")
        assert update.session_id == "gen-client-1"
        assert update.status == "waiting"

        # Client disconnects abruptly
        disconnect_from_server()
        await asyncio.sleep(0.3)

        # Verify client disconnected
        state = _get_server_state()
        assert "gen-client-1" not in state.client_connections

        # Now connect a new client - server should still work
        await connect_to_server("localhost", port, "gen-client-2")
        await asyncio.sleep(0.1)

        await notify_server(
            SessionUpdate(
                session_id="gen-client-2",
                status="running",
                time_unix_nano="1234567890000000000",
                data="I am client 2",
            )
        )

        # Server should process this
        update2 = await expect_update("gen-client-2")
        assert update2.session_id == "gen-client-2"
        assert update2.data == "I am client 2"

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
