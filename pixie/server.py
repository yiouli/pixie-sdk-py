"""FastAPI server for SDK."""

import argparse
import os
import colorlog
import logging
from urllib.parse import quote
import dotenv
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter

from pixie.prompts.file_watcher import discover_and_load_modules, init_prompt_storage
from pixie.schema import schema


logger = logging.getLogger(__name__)

# Global logging mode
_logging_mode: str = "default"


def setup_logging(mode: str = "default"):
    """Configure logging for the entire application.

    Sets up colored logging with consistent formatting for all loggers.

    Args:
        mode: Logging mode - "default", "verbose", or "debug"
            - default: INFO for server events, WARNING+ for all modules
            - verbose: INFO+ for all modules
            - debug: DEBUG+ for all modules
    """
    global _logging_mode
    _logging_mode = mode

    # Determine log level based on mode
    if mode == "debug":
        level = logging.DEBUG
    elif mode == "verbose":
        level = logging.INFO
    else:  # default
        level = logging.INFO

    colorlog.basicConfig(
        level=level,
        format="[%(log_color)s%(levelname)-8s%(reset)s][%(asctime)s]\t%(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        force=True,
    )

    # Configure uvicorn loggers to use the same format
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True

    # In default mode, set most loggers to WARNING+ except specific modules
    if mode == "default":
        # Set root logger to WARNING
        logging.getLogger().setLevel(logging.WARNING)
        # Allow INFO for pixie modules
        logging.getLogger("pixie").setLevel(logging.INFO)
        # Suppress uvicorn access logs in default mode
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def enable_instrumentations():
    """Enable all available instrumentations for AI frameworks.

    Attempts to enable instrumentation for pydantic-ai, OpenAI Agents, Google ADK,
    CrewAI, and DSpy. Only logs in verbose/debug modes.
    """
    try:
        from pydantic_ai import Agent  # type: ignore

        Agent.instrument_all()
        if _logging_mode in ("verbose", "debug"):
            logger.info("Enabled pydantic-ai Agent instrumentation.")
    except ImportError:
        if _logging_mode in ("verbose", "debug"):
            logger.warning("pydantic-ai is not installed; instrumentation is disabled.")

    try:
        from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor  # type: ignore

        OpenAIAgentsInstrumentor().instrument()
        if _logging_mode in ("verbose", "debug"):
            logger.info("Enabled OpenAI Agents instrumentation.")
    except ImportError:
        if _logging_mode in ("verbose", "debug"):
            logger.warning(
                "openinference-instrumentation-openai-agents is not installed."
            )

    try:
        from openinference.instrumentation.google_adk import GoogleADKInstrumentor  # type: ignore

        GoogleADKInstrumentor().instrument()
        if _logging_mode in ("verbose", "debug"):
            logger.info("Enabled Google ADK instrumentation.")
    except ImportError:
        if _logging_mode in ("verbose", "debug"):
            logger.warning("openinference-instrumentation-google-adk is not installed.")

    try:
        from openinference.instrumentation.crewai import CrewaiInstrumentor  # type: ignore

        CrewaiInstrumentor().instrument()
        if _logging_mode in ("verbose", "debug"):
            logger.info("Enabled CrewAI instrumentation.")
    except ImportError:
        if _logging_mode in ("verbose", "debug"):
            logger.warning("openinference-instrumentation-crewai is not installed.")

    try:
        from openinference.instrumentation.dspy import DSpyInstrumentor  # type: ignore

        DSpyInstrumentor().instrument()
        if _logging_mode in ("verbose", "debug"):
            logger.info("Enabled DSpy instrumentation.")
    except ImportError:
        if _logging_mode in ("verbose", "debug"):
            logger.warning("openinference-instrumentation-dspy is not installed.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance with GraphQL router.
    """
    # Setup logging first (use global logging mode)
    setup_logging(_logging_mode)

    discover_and_load_modules()

    enable_instrumentations()

    dotenv.load_dotenv(os.getcwd() + "/.env")
    lifespan = init_prompt_storage()

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

    @app.get("/")
    async def root():
        return {
            "message": "Pixie SDK Server",
            "graphiql": "/graphql",
            "version": "0.1.0",
        }

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

    global _logging_mode
    _logging_mode = log_mode

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

    # Display gopixie.ai web link
    encoded_url = quote(f"{server_url}/graphql", safe="")
    pixie_web_url = f"https://gopixie.ai?url={encoded_url}"
    logger.info("")
    logger.info("=" * 60)
    logger.info("")
    logger.info("🎨 Open Pixie Web UI:")
    logger.info("")
    logger.info("   %s", pixie_web_url)
    logger.info("")
    logger.info("=" * 60)
    logger.info("")

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
