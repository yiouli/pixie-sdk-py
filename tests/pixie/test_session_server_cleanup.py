"""Tests for session server cleanup scenarios.

These tests verify that the socket server is properly cleaned up when:
1. The server task is cancelled (simulating GraphQL subscription disconnect)
2. Normal shutdown via stop_server()
"""

import asyncio
import threading
import pytest
import janus
import socket
from pixie.session.rpc import (
    listen_to_client_connections,
    stop_server,
    _get_server_state,
    _shutdown_server,
)
from pixie.session.server import _run_in_thread
from pixie.session.types import SessionUpdate


# Test timeout for socket operations
TIMEOUT = 10.0

# Base port for tests - use different ranges to avoid conflicts
TEST_PORT_BASE = 18000


def get_test_port(test_id: int) -> int:
    """Get a unique port for a test to avoid conflicts.

    Args:
        test_id: Unique identifier for the test.

    Returns:
        Port number.
    """
    return TEST_PORT_BASE + test_id


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is in use.

    Args:
        port: The port number to check.
        host: The host to check on.

    Returns:
        True if the port is in use, False otherwise.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        result = sock.connect_ex((host, port))
        return result == 0


@pytest.fixture
def cleanup():
    """Cleanup fixture to ensure server state is reset after each test."""
    yield
    stop_server()
    # Force shutdown server if still running
    _shutdown_server()
    # Give sockets time to clean up
    import time

    time.sleep(0.2)


class TestServerCleanupOnTaskCancel:
    """Tests for server cleanup when the task is cancelled.

    This simulates the scenario where the GraphQL subscription client disconnects,
    causing the subscription handler to cancel the server task.
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_server_state_cleared_on_task_cancel(self, cleanup):
        """Test that _server_state is cleared when server task is cancelled."""
        import pixie.session.rpc as rpc_module

        port = get_test_port(1)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        server_task = asyncio.create_task(
            listen_to_client_connections(port, update_queue)
        )
        await asyncio.sleep(0.3)  # Let server start

        # Verify server is running
        assert rpc_module._server_state is not None
        state = _get_server_state()
        assert state.running is True
        assert len(state.servers) == 1

        # Cancel the task (simulating GraphQL client disconnect)
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

        # Wait for cleanup
        await asyncio.sleep(0.3)

        # Server state should be cleared
        assert rpc_module._server_state is None

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_port_released_on_task_cancel(self, cleanup):
        """Test that ports are released when server task is cancelled."""
        port = get_test_port(2)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        server_task = asyncio.create_task(
            listen_to_client_connections(port, update_queue)
        )
        await asyncio.sleep(0.3)  # Let server start

        # Verify port is in use
        assert is_port_in_use(port), f"Port {port} should be in use"

        # Cancel the task
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

        # Wait for cleanup
        await asyncio.sleep(0.3)

        # Port should be released
        assert not is_port_in_use(port), f"Port {port} should be released"

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_server_restartable_after_task_cancel(self, cleanup):
        """Test that server can be restarted after task cancellation."""
        import pixie.session.rpc as rpc_module

        port = get_test_port(3)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        # Start server first time
        server_task = asyncio.create_task(
            listen_to_client_connections(port, update_queue)
        )
        await asyncio.sleep(0.3)

        # Cancel first server
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.3)

        # Should be able to start server again
        update_queue2: janus.Queue[SessionUpdate] = janus.Queue()
        server_task2 = asyncio.create_task(
            listen_to_client_connections(port, update_queue2)
        )
        await asyncio.sleep(0.3)

        # Verify second server is running
        assert rpc_module._server_state is not None
        state = _get_server_state()
        assert state.running is True

        # Cleanup
        server_task2.cancel()
        try:
            await server_task2
        except asyncio.CancelledError:
            pass


class TestServerCleanupInThread:
    """Tests for server cleanup when running in a separate thread.

    This more closely simulates the actual scenario in run_session_server
    where the server runs in a thread via _run_in_thread.
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_server_cleanup_via_thread_cancel(self, cleanup):
        """Test that server is cleaned up when thread task is cancelled."""
        import pixie.session.rpc as rpc_module

        port = get_test_port(10)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        # Run server in thread like run_session_server does
        run, cancel = _run_in_thread(listen_to_client_connections(port, update_queue))
        task = asyncio.create_task(run)
        await asyncio.sleep(0.5)  # Let server start

        # Verify server is running
        assert rpc_module._server_state is not None

        # Cancel like run_session_server.finally does
        cancel()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Wait for cleanup
        await asyncio.sleep(0.5)

        # Server state should be cleared
        assert (
            rpc_module._server_state is None
        ), "Server state should be cleared after cancel"

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_port_released_via_thread_cancel(self, cleanup):
        """Test that port is released when thread task is cancelled."""
        port = get_test_port(11)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        # Run server in thread
        run, cancel = _run_in_thread(listen_to_client_connections(port, update_queue))
        task = asyncio.create_task(run)
        await asyncio.sleep(0.5)

        # Verify port is in use
        assert is_port_in_use(port), f"Port {port} should be in use"

        # Cancel
        cancel()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.5)

        # Port should be released
        assert not is_port_in_use(port), f"Port {port} should be released after cancel"

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_server_restartable_after_thread_cancel(self, cleanup):
        """Test that server can be restarted after thread cancellation."""
        import pixie.session.rpc as rpc_module

        port = get_test_port(12)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        # Start and cancel server first time
        run, cancel = _run_in_thread(listen_to_client_connections(port, update_queue))
        task = asyncio.create_task(run)
        await asyncio.sleep(0.5)

        cancel()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.5)

        # Start server second time
        update_queue2: janus.Queue[SessionUpdate] = janus.Queue()
        run2, cancel2 = _run_in_thread(
            listen_to_client_connections(port, update_queue2)
        )
        task2 = asyncio.create_task(run2)
        await asyncio.sleep(0.5)

        # Should be running
        assert rpc_module._server_state is not None, "Server should be restartable"

        # Cleanup
        cancel2()
        task2.cancel()
        try:
            await task2
        except asyncio.CancelledError:
            pass


class TestServerCleanupCrossThread:
    """Tests for server cleanup when cancel is called from different thread.

    This simulates the exact scenario from the bug report where:
    1. listen_to_client_connections runs in a child thread with its own event loop
    2. The parent thread calls task.cancel() when cleaning up
    3. task.cancel() from different thread does NOT work properly
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_cancel_from_different_event_loop_needs_thread_safe(self, cleanup):
        """Test that task.cancel() from different event loop needs thread-safe call.

        This test demonstrates the potential issue: when we create a new event loop
        in a thread and call task.cancel() from the main event loop, it may not work
        reliably without thread-safe mechanisms.
        """
        import pixie.session.rpc as rpc_module

        port = get_test_port(20)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        # Create server in a separate event loop (like _run_in_thread does)
        child_loop = asyncio.new_event_loop()
        server_coro = listen_to_client_connections(port, update_queue)
        child_task = child_loop.create_task(server_coro)

        # Run the child loop in a thread
        def run_child_loop():
            asyncio.set_event_loop(child_loop)
            try:
                child_loop.run_until_complete(child_task)
            except asyncio.CancelledError:
                pass

        thread = threading.Thread(target=run_child_loop, daemon=True)
        thread.start()

        await asyncio.sleep(0.5)  # Let server start

        # Verify server is running and port is in use
        assert rpc_module._server_state is not None, "Server should be running"
        assert is_port_in_use(port), f"Port {port} should be in use"

        # Cancel using thread-safe call - THIS IS THE PROPER WAY
        child_loop.call_soon_threadsafe(child_task.cancel)

        await asyncio.sleep(0.5)  # Wait for cleanup

        # These assertions document the expected behavior after fix
        server_state_cleared = rpc_module._server_state is None
        port_released = not is_port_in_use(port)

        assert server_state_cleared, "Server state should be cleared after cancel"
        assert port_released, f"Port {port} should be released after cancel"

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_run_in_thread_cancel_cleans_up(self, cleanup):
        """Test that _run_in_thread cancel function properly cleans up server.

        This tests the actual pattern used by run_session_server.
        """
        import pixie.session.rpc as rpc_module

        port = get_test_port(21)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        # Use _run_in_thread like run_session_server does
        run, cancel = _run_in_thread(listen_to_client_connections(port, update_queue))
        task = asyncio.create_task(run)

        await asyncio.sleep(0.5)  # Let server start

        # Verify server is running
        assert rpc_module._server_state is not None, "Server should be running"
        assert is_port_in_use(port), f"Port {port} should be in use"

        # Cancel like run_session_server.finally does
        cancel()
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        await asyncio.sleep(0.5)  # Wait for cleanup

        # Verify cleanup happened
        assert rpc_module._server_state is None, "Server state should be cleared"
        assert not is_port_in_use(port), f"Port {port} should be released"

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_server_restartable_after_cross_thread_cancel(self, cleanup):
        """Test server can restart after being cancelled from different thread."""
        import pixie.session.rpc as rpc_module

        port = get_test_port(22)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        # First run
        run, cancel = _run_in_thread(listen_to_client_connections(port, update_queue))
        task = asyncio.create_task(run)
        await asyncio.sleep(0.5)

        # Cancel
        cancel()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.5)

        # Should be able to restart
        update_queue2: janus.Queue[SessionUpdate] = janus.Queue()
        run2, cancel2 = _run_in_thread(
            listen_to_client_connections(port, update_queue2)
        )
        task2 = asyncio.create_task(run2)
        await asyncio.sleep(0.5)

        # Verify second server is running
        assert rpc_module._server_state is not None, "Server should restart"
        assert is_port_in_use(port), f"Port {port} should be in use"

        # Cleanup
        cancel2()
        task2.cancel()
        try:
            await task2
        except asyncio.CancelledError:
            pass


class TestConnectionDuringShutdown:
    """Tests for race conditions when connections arrive during shutdown."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_connection_during_shutdown_handled_gracefully(self, cleanup):
        """Test that connections arriving during shutdown don't cause errors.

        This tests the race condition where _handle_new_connection is called
        after _server_state has been set to None.
        """
        port = get_test_port(30)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        server_task = asyncio.create_task(
            listen_to_client_connections(port, update_queue)
        )
        await asyncio.sleep(0.3)  # Let server start

        # Start shutdown
        stop_server()

        # Try to connect during shutdown - should not raise "Server not initialized"
        # The connection should be gracefully rejected
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection("localhost", port),
                timeout=1.0,
            )
            # If connection succeeded, close it
            writer.close()
            await writer.wait_closed()
        except (ConnectionRefusedError, asyncio.TimeoutError, OSError):
            # Connection refused is expected during shutdown
            pass

        # Wait for server to fully shut down
        try:
            await asyncio.wait_for(server_task, timeout=2.0)
        except asyncio.TimeoutError:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    @pytest.mark.timeout(TIMEOUT)
    async def test_multiple_connections_during_shutdown(self, cleanup):
        """Test that multiple connections during shutdown are handled gracefully."""
        port = get_test_port(31)
        update_queue: janus.Queue[SessionUpdate] = janus.Queue()

        server_task = asyncio.create_task(
            listen_to_client_connections(port, update_queue)
        )
        await asyncio.sleep(0.3)  # Let server start

        # Start shutdown
        stop_server()

        # Try multiple concurrent connections during shutdown
        async def try_connect():
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection("localhost", port),
                    timeout=0.5,
                )
                writer.close()
                await writer.wait_closed()
            except (ConnectionRefusedError, asyncio.TimeoutError, OSError):
                pass

        # Fire off multiple connection attempts
        tasks = [asyncio.create_task(try_connect()) for _ in range(5)]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Wait for server to fully shut down
        try:
            await asyncio.wait_for(server_task, timeout=2.0)
        except asyncio.TimeoutError:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
