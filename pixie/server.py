"""FastAPI server for SDK."""

import argparse
from contextlib import asynccontextmanager
import os
import logging
import threading
import time
from urllib.parse import quote
import webbrowser
import dotenv
import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter

from pixie.prompts.file_watcher import discover_and_load_modules, init_prompt_storage
from pixie.schema import schema
from pixie.server_utils import enable_instrumentations, setup_logging
from pixie.session.server import start_session_server


logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance with GraphQL router.
    """
    # Setup logging first (use global logging mode)
    setup_logging()

    discover_and_load_modules()

    enable_instrumentations()

    dotenv.load_dotenv(os.getcwd() + "/.env")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for FastAPI app.

        Initializes prompt storage on startup.
        """
        storage_lifespan = init_prompt_storage()
        async with storage_lifespan(app):
            session_server = await start_session_server()
            try:
                yield
            finally:
                session_server.task.cancel()
                session_server.cancel()

    app = FastAPI(
        title="Pixie SDK Server",
        description="Server for running AI applications and agents",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )

    # Add GraphQL router with GraphiQL enabled
    graphql_app = GraphQLRouter(
        schema,
        graphiql=True,
    )

    app.include_router(graphql_app, prefix="/graphql")

    REMOTE_URL = "https://gopixie.ai"

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy_all(request: Request, path: str):
        url = f"{REMOTE_URL}/{path}"
        if request.url.query:
            url += f"?{request.url.query}"

        logger.debug("Proxying request to: %s", url)

        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            response = await client.request(
                method=request.method,
                url=url,
                headers={
                    k: v
                    for k, v in request.headers.items()
                    if k.lower() not in ["host"]
                },
                content=await request.body(),
            )

            # Explicitly remove compression-related headers
            headers = {
                k: v
                for k, v in response.headers.items()
                if k.lower()
                not in ["content-encoding", "content-length", "transfer-encoding"]
            }

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=headers,
            )

    return app


def start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    log_mode: str = "default",
) -> None:
    """Start the SDK server.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
        log_mode: Logging mode - "default", "verbose", or "debug"
        storage_directory: Directory to store prompt definitions
    """
    from pixie.registry import list_applications

    # Setup logging (will be called again in create_app for reload scenarios)
    setup_logging(log_mode)

    # Determine server URL
    server_url = f"http://{host}:{port}"
    if host == "0.0.0.0":
        server_url = f"http://127.0.0.1:{port}"

    # Log server start info
    logger.info("Starting Pixie SDK Server")
    logger.info("Server: %s", server_url)
    logger.info("GraphQL: %s/graphql", server_url)

    # Log registered applications
    apps = list_applications()
    if apps:
        logger.info("Registered applications: %d", len(apps))
        for app_id in apps:
            logger.info("  • %s", app_id)
    else:
        logger.warning("No applications registered yet")

    if server_url == "http://127.0.0.1:8000":
        query_param = ""
    else:
        query_param = f'?url={quote(f"{server_url}/graphql", safe="")}'
    logger.info("")
    logger.info("=" * 60)
    logger.info("")
    logger.info("🎨 Open Pixie Web UI:")
    logger.info("")
    logger.info("   %s", f"https://gopixie.ai{query_param}")
    logger.info("")
    logger.info("   or")
    logger.info("")
    logger.info("   %s", f"{server_url}{query_param}")
    logger.info("")
    logger.info("=" * 60)
    logger.info("")

    # Open browser after a short delay (in a separate thread)
    def open_browser():
        time.sleep(1.5)  # Wait for server to start
        webbrowser.open(server_url)

    threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(
        "pixie.server:create_app",
        host=host,
        port=port,
        loop="asyncio",
        reload=reload,
        factory=True,
        log_config=None,
    )


def main():
    """Start the Pixie server.

    Loads environment variables and starts the server with auto-reload enabled.
    Supports --verbose and --debug flags for enhanced logging.
    """
    parser = argparse.ArgumentParser(
        description="Pixie SDK Server - Interactive debugging for AI applications"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (INFO+ for all modules)",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug logging (DEBUG+ for all modules)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=None,
        help="Port to run the server on (overrides PIXIE_SDK_PORT env var)",
    )
    args = parser.parse_args()

    # Determine logging mode
    log_mode = "default"
    if args.debug:
        log_mode = "debug"
    elif args.verbose:
        log_mode = "verbose"

    dotenv.load_dotenv(os.getcwd() + "/.env")
    port = args.port or int(os.getenv("PIXIE_SDK_PORT", "8000"))

    start_server(port=port, reload=True, log_mode=log_mode)


if __name__ == "__main__":
    main()
