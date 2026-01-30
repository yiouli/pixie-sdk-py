"""RPC communication layer for pixie session using asyncio TCP streams.

This module provides cross-process communication between session clients
and the session server using asyncio TCP streams for proper connection management.

Server-side:
- listen_to_client_connections() runs as an async task, accepting connections
  on a single port and queuing all received SessionUpdates into an asyncio queue.
- After accepting a connection, server waits for session_id within 2 seconds.
- Once registered, a worker task handles messages from that client.

Client-side:
- connect_to_server() creates a singleton connection for the entire process.
- Connects to the server port and registers with session_id.
- notify_server() sends SessionUpdate messages.
- wait_for_input() blocks waiting for input data from server.
"""

import asyncio
import json
import logging
import struct
import threading
from dataclasses import dataclass, field
from typing import TypeVar

import janus
from pydantic import BaseModel, JsonValue

from pixie.session.constants import SESSION_RPC_SERVER_HOST
from pixie.session.types import SessionUpdate
from pixie.types import InputRequired, InputType


logger = logging.getLogger(__name__)


T = TypeVar("T", bound=BaseModel)


# Protocol constants
REGISTRATION_TIMEOUT = 2.0  # seconds to wait for session_id after connection
MESSAGE_HEADER_SIZE = 4  # 4 bytes for message length prefix


async def _send_message(writer: asyncio.StreamWriter, data: bytes) -> None:
    """Send a length-prefixed message.

    Args:
        writer: The stream writer.
        data: The message data to send.
    """
    # Send 4-byte length prefix (big-endian) followed by data
    length_prefix = struct.pack(">I", len(data))
    writer.write(length_prefix + data)
    await writer.drain()


async def _recv_message(
    reader: asyncio.StreamReader,
    timeout: float | None = None,
) -> bytes | None:
    """Receive a length-prefixed message.

    Args:
        reader: The stream reader.
        timeout: Optional timeout in seconds.

    Returns:
        The message data, or None if connection closed or timeout.
    """

    async def _read():
        # Read 4-byte length prefix
        length_data = await reader.readexactly(MESSAGE_HEADER_SIZE)
        message_length = struct.unpack(">I", length_data)[0]

        # Read message data
        return await reader.readexactly(message_length)

    try:
        if timeout is not None:
            return await asyncio.wait_for(_read(), timeout=timeout)
        else:
            return await _read()
    except (asyncio.IncompleteReadError, asyncio.TimeoutError, ConnectionError):
        return None


# =============================================================================
# Server-side state and functions
# =============================================================================


@dataclass
class _ClientConnection:
    """Represents a single client connection on the server side.

    Attributes:
        session_id: The session ID of the client.
        reader: The asyncio stream reader.
        writer: The asyncio stream writer.
        task: The worker task handling this client.
    """

    session_id: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    task: asyncio.Task | None = None


@dataclass
class _ServerState:
    """Server state tracking all listeners and client connections.

    Attributes:
        servers: List of asyncio.Server instances (one per port).
        update_queues: Map of session_id to janus.Queue where all client SessionUpdates are placed.
        client_connections: Map of session_id to client connection info.
        running: Flag to control the server lifecycle.
        lock: Lock for client_connections access.
    """

    servers: list[asyncio.Server] = field(default_factory=list)
    client_connections: dict[str, _ClientConnection] = field(default_factory=dict)
    running: bool = True
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    update_queues: dict[str, janus.Queue[SessionUpdate]] = field(default_factory=dict)


_server_state: _ServerState | None = None
_server_lock = threading.Lock()


def _get_server_state() -> _ServerState:
    """Get the server state singleton.

    Raises:
        RuntimeError: If server is not initialized.
    """
    if _server_state is None:
        raise RuntimeError(
            "Server not initialized. Call listen_to_client_connections first."
        )
    return _server_state


# Timeout for polling server running status in worker loop
WORKER_POLL_TIMEOUT = 0.5  # seconds


async def _handle_client_messages(
    connection: _ClientConnection,
    update_queue: janus.Queue[SessionUpdate],
) -> None:
    """Worker task to handle messages from a single client.

    Reads messages from the client and puts them in the update queue.
    Cleans up resources when connection is lost.

    Args:
        connection: The client connection info.
        update_queue: The queue to put received updates.
    """
    state = _get_server_state()

    try:
        while state.running:
            # Use timeout so we can check state.running periodically
            data = await _recv_message(connection.reader, timeout=WORKER_POLL_TIMEOUT)
            if data is None:
                # Connection lost or timeout - check if we should continue
                if connection.reader.at_eof():
                    break  # Connection actually closed
                continue  # Just timeout, try again

            try:
                update = SessionUpdate.model_validate_json(data.decode())
                update_queue.sync_q.put(update)
            except Exception as e:
                logger.error(f"Error parsing message from {connection.session_id}: {e}")
    finally:
        # Cleanup: remove from registry and close connection
        async with state.lock:
            if connection.session_id in state.client_connections:
                del state.client_connections[connection.session_id]
                # Note: queue cleanup is handled by _async_shutdown_server

        try:
            connection.writer.close()
            await connection.writer.wait_closed()
            logger.info(f"Closed connection for session_id: {connection.session_id}")

        except Exception:
            logger.error("Error closing client connection", exc_info=True)


async def _handle_new_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """Handle a new incoming connection.

    Waits for session_id registration within timeout, then spawns a worker.

    Args:
        reader: The stream reader for this connection.
        writer: The stream writer for this connection.
    """
    # Check if server is still running - may have been shut down during accept
    if _server_state is None:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
        return

    state = _server_state

    try:
        logger.info("New client connection received, waiting for registration")
        # Wait for registration message (session_id) within timeout
        data = await _recv_message(reader, timeout=REGISTRATION_TIMEOUT)

        if data is None:
            logger.warning("Client registration timed out or connection error")
            # Timeout or connection error - close the connection
            writer.close()
            await writer.wait_closed()
            return

        session_id = data.decode()

        if not session_id:
            logger.warning("Empty session_id received, rejecting connection")
            # Empty session_id - reject
            writer.close()
            await writer.wait_closed()
            return

        # Check if session_id already registered
        async with state.lock:
            if session_id in state.client_connections:
                logger.warning(
                    f"Duplicate session_id {session_id} received, rejecting connection"
                )
                # Duplicate session - reject
                writer.close()
                await writer.wait_closed()
                return

            # Create connection record
            connection = _ClientConnection(
                session_id=session_id,
                reader=reader,
                writer=writer,
            )
            state.client_connections[session_id] = connection
            queue = state.update_queues[session_id] = janus.Queue[SessionUpdate]()
            logger.info(f"Client registered with session_id: {session_id}")

        # Send ACK to client (empty message means success)
        await _send_message(writer, b"ACK")

        # Spawn worker task to handle messages
        task = asyncio.create_task(_handle_client_messages(connection, queue))
        connection.task = task

    except Exception as e:
        logger.error(f"Error handling new connection: {e}")
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            logger.error("Error closing writer", exc_info=True)


async def listen_to_client_connections(port: int) -> None:
    """Listen for client connections on a port and queue received SessionUpdates.

    This is a long-running async function. It starts a TCP server on the specified
    port to accept connections from clients. All received SessionUpdate messages
    are placed into a shared asyncio queue.

    Protocol:
    - Client connects to the port
    - Client must send session_id within 2 seconds
    - Server sends ACK on successful registration
    - Subsequent messages are SessionUpdate JSON data (length-prefixed)

    Args:
        port: Port to listen on.
        update_queue: The Janus queue to place received SessionUpdates into.

    Raises:
        RuntimeError: If server is already running.
    """
    global _server_state

    with _server_lock:
        if _server_state is not None:
            raise RuntimeError("Server already running.")

        _server_state = _ServerState()

    state = _server_state

    try:
        # Start server on the port
        try:
            server = await asyncio.start_server(
                _handle_new_connection,
                host=SESSION_RPC_SERVER_HOST,
                port=port,
            )
            state.servers.append(server)
            logger.info(f"Session RPC server listening on port {port}")
        except OSError as e:
            raise RuntimeError(f"Failed to bind to port {port}: {e}") from e

        # Wait until shutdown is signaled
        while state.running:
            await asyncio.sleep(0.1)

    finally:
        logger.info("Shutting down session RPC server")
        await _async_shutdown_server()


async def _async_shutdown_server() -> None:
    """Clean up server resources asynchronously."""
    global _server_state

    with _server_lock:
        if _server_state is None:
            return
        state = _server_state
        state.running = False
        # Clear state immediately to prevent "server already running" errors
        _server_state = None

    tasks_to_wait = []
    async with state.lock:
        # First, close all queues and wait for them
        for session_id, queue in list(state.update_queues.items()):
            try:
                queue.close()
                await asyncio.wait_for(queue.wait_closed(), timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for queue {session_id} to close")
            except Exception:
                logger.error(
                    f"Error closing update queue for {session_id}", exc_info=True
                )
        state.update_queues.clear()

        # Cancel all client worker tasks and close connections BEFORE waiting for server
        # This ensures connections are dropped so server.wait_closed() doesn't block
        for conn in state.client_connections.values():
            # Feed EOF to reader to unblock pending reads
            try:
                conn.reader.feed_eof()
            except Exception:
                logger.error("Error feeding EOF to client reader", exc_info=True)
            # Cancel the worker task
            if conn.task and not conn.task.done():
                conn.task.cancel()
                tasks_to_wait.append(conn.task)
            try:
                conn.writer.close()
                await conn.writer.wait_closed()
            except Exception:
                logger.error("Error closing client writer", exc_info=True)

        # Now close servers - connections are already dropped so this should be fast
        for server in state.servers:
            try:
                server.close()
                await server.wait_closed()
            except Exception:
                logger.error("Error closing server", exc_info=True)

    # Wait for all tasks to complete with a timeout
    if tasks_to_wait:
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks_to_wait, return_exceptions=True),
                timeout=2.0,
            )
            # Log any exceptions from tasks
            for i, result in enumerate(results):
                if isinstance(result, Exception) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    logger.error(f"Task {i} raised exception during shutdown: {result}")
            logger.info("Session RPC server shut down successfully")
        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout waiting for {len(tasks_to_wait)} client tasks to cancel"
            )
            # Force cancel any remaining tasks
            for task in tasks_to_wait:
                if isinstance(task, asyncio.Task) and not task.done():
                    task.cancel()


def stop_server() -> None:
    """Signal the server to stop listening."""
    global _server_state
    with _server_lock:
        if _server_state is not None:
            _server_state.running = False


async def send_input_to_client(
    session_id: str,
    input_data: JsonValue | BaseModel,
) -> None:
    """Send input data back to a specific client.

    Args:
        session_id: The session ID of the client to send to.
        input_data: The input data to send (JSON or Pydantic model).

    Raises:
        RuntimeError: If server is not initialized.
        KeyError: If no client with the given session_id is connected.
    """
    state = _get_server_state()

    # Get client connection (lock needed for reading)
    async with state.lock:
        if session_id not in state.client_connections:
            raise KeyError(f"No client connected with session_id: {session_id}")
        connection = state.client_connections[session_id]

    # Serialize data
    if isinstance(input_data, BaseModel):
        data = input_data.model_dump_json().encode()
    else:
        data = json.dumps(input_data).encode()

    # Send via length-prefixed protocol
    await _send_message(connection.writer, data)


def get_connected_session_ids() -> list[str]:
    """Get a list of currently connected session IDs.

    Returns:
        List of connected session IDs.

    Raises:
        RuntimeError: If server is not initialized.
    """
    state = _get_server_state()

    async def _get_ids():
        async with state.lock:
            return list(state.client_connections.keys())

    return asyncio.run(_get_ids())


async def wait_for_client_update(session_id: str) -> SessionUpdate:
    """Wait for the next SessionUpdate from any client.

    Returns:
        The next SessionUpdate received from a client.
    Raises:
        RuntimeError: If server is not initialized.
    """
    state = _get_server_state()

    queue = state.update_queues.get(session_id)

    if queue is None:
        raise KeyError(f"No update queue for session_id: {session_id}")

    update = await queue.async_q.get()
    return update


# =============================================================================
# Client-side state and functions
# =============================================================================


@dataclass
class _ClientState:
    """Singleton client state.

    Attributes:
        reader: The asyncio stream reader.
        writer: The asyncio stream writer.
        session_id: The session ID this client is registered with.
        connected_port: The port that was successfully connected to.
    """

    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    session_id: str
    connected_port: int


_client_state: _ClientState | None = None
_client_lock = threading.Lock()


def _get_client_state() -> _ClientState:
    """Get the client state singleton.

    Raises:
        RuntimeError: If client is not connected.
    """
    if _client_state is None:
        raise RuntimeError("Client not connected. Call connect_to_server first.")
    return _client_state


async def connect_to_server(
    host: str,
    port: int,
    session_id: str,
    connection_timeout: float = 5.0,
) -> int:
    """Connect to the session server and register this client.

    Creates a singleton connection for the entire client process.
    Connects to the specified port, sends the session_id as a registration
    message and waits for ACK.

    Args:
        host: The server hostname or IP address.
        port: Port number to connect to.
        session_id: The session ID for this client.
        connection_timeout: Timeout for connection attempt in seconds.

    Returns:
        The port number that was successfully connected to.

    Raises:
        RuntimeError: If already connected or if connection fails.
    """
    global _client_state

    with _client_lock:
        if _client_state is not None:
            raise RuntimeError(
                "Already connected to server. Call disconnect_from_server first."
            )

    try:
        # Try to connect to the port
        logger.info(f"Attempting to connect to server at {host}:{port}")
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=connection_timeout,
        )
        logger.info(f"Connected to server at {host}:{port}")

        try:
            # Send registration message with session_id
            await _send_message(writer, session_id.encode())

            # Wait for ACK from server
            ack = await _recv_message(reader, timeout=REGISTRATION_TIMEOUT)

            if ack is None or ack != b"ACK":
                writer.close()
                await writer.wait_closed()
                raise RuntimeError(
                    f"Server at {host}:{port} rejected registration or timed out"
                )

            # Connection successful
            with _client_lock:
                if _client_state is not None:
                    # Race condition - another connection succeeded
                    writer.close()
                    await writer.wait_closed()
                    raise RuntimeError(
                        "Already connected to server. "
                        "Call disconnect_from_server first."
                    )

                _client_state = _ClientState(
                    reader=reader,
                    writer=writer,
                    session_id=session_id,
                    connected_port=port,
                )

            return port

        except Exception as e:
            logger.error(f"Error during registration with server at {host}:{port}: {e}")
            # Clean up on error
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                logger.error("Error closing writer", exc_info=True)
            raise

    except asyncio.TimeoutError as e:
        raise RuntimeError(f"Connection to {host}:{port} timed out") from e
    except OSError as e:
        raise RuntimeError(f"Failed to connect to server at {host}:{port}: {e}") from e


def disconnect_from_server() -> None:
    """Disconnect from the server and clean up resources.

    Safe to call even if not connected or if event loop is closed.
    """
    global _client_state

    with _client_lock:
        if _client_state is not None:
            try:
                _client_state.writer.close()
                logger.info(
                    f"Disconnected from server on port {_client_state.connected_port}"
                )
            except RuntimeError as e:
                # Event loop may be closed during test teardown - this is expected
                if "Event loop is closed" not in str(e):
                    logger.error("Error closing client writer", exc_info=True)
            except Exception:
                logger.error("Error closing client writer", exc_info=True)
            _client_state = None


async def notify_server(update: SessionUpdate) -> None:
    """Send a SessionUpdate to the server.

    Args:
        update: The SessionUpdate to send.

    Raises:
        RuntimeError: If not connected to server.
    """
    state = _get_client_state()

    data = update.model_dump_json().encode()
    logger.debug(f"Sending SessionUpdate to server: {data.decode()}")
    await _send_message(state.writer, data)
    logger.debug("SessionUpdate sent successfully")


async def wait_for_input(requirement: InputRequired[InputType]) -> InputType:
    """Wait for input data from the server.

    Blocks until the server sends input data back.

    Args:
        requirement: The input requirement specifying the expected type.

    Returns:
        The input data, deserialized to the expected type if it's a Pydantic model.

    Raises:
        RuntimeError: If not connected to server.
        ConnectionError: If connection is lost while waiting.
    """
    state = _get_client_state()

    logger.debug("Waiting for input data from server")
    data = await _recv_message(state.reader)

    if data is None:
        raise ConnectionError("Connection lost while waiting for input")

    logger.debug(f"Received input data from server: {data.decode()}")
    # Deserialize based on expected type
    expected_type = requirement.expected_type
    if isinstance(expected_type, type) and issubclass(expected_type, BaseModel):
        return expected_type.model_validate_json(data.decode())
    else:
        return json.loads(data.decode())
