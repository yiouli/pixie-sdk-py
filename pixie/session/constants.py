# Port range for session RPC connections
# Server listens on multiple ports to support multiple concurrent clients
SESSION_RPC_PORT_START = 11111
SESSION_RPC_PORT_COUNT = 10  # Support up to 10 concurrent clients
SESSION_RPC_PORTS = list(
    range(SESSION_RPC_PORT_START, SESSION_RPC_PORT_START + SESSION_RPC_PORT_COUNT)
)

SESSION_RPC_SERVER_HOST = "127.0.0.1"
