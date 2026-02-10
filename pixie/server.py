"""FastAPI server for SDK."""

import argparse
from contextlib import asynccontextmanager
import os
import logging
from urllib.parse import quote
import dotenv
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from piccolo.engine import engine_finder
from strawberry.fastapi import GraphQLRouter

from pixie.prompts.file_watcher import discover_and_load_modules, init_prompt_storage
from pixie.schema import schema
from pixie.server_utils import enable_instrumentations, setup_logging
from pixie.session.server import start_session_server
from pixie.storage.tables import create_tables

logger = logging.getLogger(__name__)


_sdk_server_address = "https://gopixie.ai"  # Default address for the Pixie Web UI


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance with GraphQL router.
    """
    # Set the Piccolo config path BEFORE importing engine_finder
    os.environ.setdefault("PICCOLO_CONF", "pixie.piccolo_conf")

    # Setup logging first (use global logging mode)
    setup_logging()

    logger.info("Loading apps and prompts...")
    discover_and_load_modules()

    logger.info("Enabling instrumentations...")
    enable_instrumentations()

    dotenv.load_dotenv(os.getcwd() + "/.env")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for FastAPI app.

        Initializes prompt storage on startup.
        """
        engine = engine_finder()
        if engine is None:
            logger.error("No Piccolo engine found, storage will not work")
        else:
            await engine.start_connection_pool()
        try:
            if engine:
                await create_tables()
            storage_lifespan = init_prompt_storage()
            async with storage_lifespan(app):
                session_server = await start_session_server()
                try:
                    yield
                finally:
                    session_server.task.cancel()
                    session_server.cancel()
        finally:
            if engine:
                await engine.close_connection_pool()

    admin_app = None
    try:
        from piccolo_admin.endpoints import create_admin

        # Import your Piccolo tables here
        from pixie.storage.tables import RunRecord, LlmCallRecord

        logger.info("Setting up Piccolo Admin...")
        admin_app = create_admin(
            tables=[RunRecord, LlmCallRecord], site_name="Pixie Admin"
        )

        logger.info("Piccolo Admin enabled at /admin/")
    except ImportError:
        logger.warning("Piccolo Admin not installed - admin UI disabled")

    app = FastAPI(
        title="Pixie SDK Server",
        description="Server for running AI applications and agents",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Matches:
    # 1. http://localhost followed by an optional port (:8080, :3000, etc.)
    # 2. http://127.0.0.1 followed by an optional port
    # 3. https://yourdomain.com (the production domain)
    origins_regex = r"http://(localhost|127\.0\.0\.1)(:\d+)?|https://gopixie\.ai"
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=origins_regex,
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

    if admin_app:
        app.mount("/admin", admin_app)

    logger.info("")
    logger.info("=" * 60)
    logger.info("")
    logger.info("🎨 Open Pixie Web UI:")
    logger.info("")
    logger.info("   %s", _sdk_server_address)
    logger.info("")
    logger.info("=" * 60)
    logger.info("")

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

    if server_url == "http://127.0.0.1:8000":
        query_param = ""
    else:
        query_param = f'?url={quote(f"{server_url}/graphql", safe="")}'

    global _sdk_server_address
    _sdk_server_address = f"https://gopixie.ai{query_param}"

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
