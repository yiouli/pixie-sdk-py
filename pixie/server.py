"""FastAPI server for SDK."""

import colorlog
import logging
import re
import sys
import importlib.util
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter
import nest_asyncio

from pixie.schema import schema


nest_asyncio.apply()


logger = logging.getLogger(__name__)


def discover_and_load_applications():
    """Discover and load all Python files that use register_application."""
    from pixie.registry import clear_registry

    # Clear registry before reloading to avoid duplicate registration errors
    clear_registry()

    cwd = Path.cwd()
    # Recursively find all Python files
    python_files = list(cwd.rglob("*.py"))

    if not python_files:
        return

    # Add current directory to Python path if not already there
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))

    loaded_count = 0
    for py_file in python_files:
        # Skip __init__.py, private files, and anything in site-packages/venv
        if py_file.name.startswith("_") or any(
            part in py_file.parts
            for part in ["site-packages", ".venv", "venv", "__pycache__"]
        ):
            continue

        # Quick check if file imports register_application
        content = py_file.read_text()
        if not re.search(r"\bfrom\s+pixie(?:\.[\w]+)*\s+import\s+pixie_app\b", content):
            continue

        # Load the module with a unique name based on path
        relative_path = py_file.relative_to(cwd)
        module_name = str(relative_path.with_suffix("")).replace("/", ".")
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            loaded_count += 1


def setup_logging():
    """Configure logging for the entire application."""
    colorlog.basicConfig(
        level=logging.INFO,
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


def enable_instrumentations():
    try:
        from pydantic_ai import Agent  # type: ignore

        Agent.instrument_all()
        logger.info("Enabled pydantic-ai Agent instrumentation.")
    except ImportError:
        logger.warning(
            "pydantic-ai is not installed; instrumentation is disabled. "
            "Install pydantic-ai to enable agent instrumentation."
        )

    try:
        from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor  # type: ignore

        OpenAIAgentsInstrumentor().instrument()
        logger.info("Enabled OpenAI Agents instrumentation.")
    except ImportError:
        logger.warning(
            "openinference-instrumentation-openai-agents is not installed; "
            "OpenAI Agents instrumentation is disabled. "
            "Install openinference-instrumentation-openai-agents to enable it."
        )

    try:
        from openinference.instrumentation.google_adk import GoogleADKInstrumentor  # type: ignore

        GoogleADKInstrumentor().instrument()
        logger.info("Enabled Google ADK instrumentation.")
    except ImportError:
        logger.warning(
            "openinference-instrumentation-google-adk is not installed; "
            "Google ADK instrumentation is disabled. "
            "Install openinference-instrumentation-google-adk to enable it."
        )

    try:
        from openinference.instrumentation.crewai import CrewaiInstrumentor  # type: ignore

        CrewaiInstrumentor().instrument()
        logger.info("Enabled CrewAI instrumentation.")
    except ImportError:
        logger.warning(
            "openinference-instrumentation-crewai is not installed; "
            "CrewAI instrumentation is disabled. "
            "Install openinference-instrumentation-crewai to enable it."
        )

    try:
        from openinference.instrumentation.dspy import DSpyInstrumentor  # type: ignore

        DSpyInstrumentor().instrument()
        logger.info("Enabled DSpy instrumentation.")
    except ImportError:
        logger.warning(
            "openinference-instrumentation-dspy is not installed; "
            "DSpy instrumentation is disabled. "
            "Install openinference-instrumentation-dspy to enable it."
        )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Setup logging first
    setup_logging()

    # Discover and load applications on every app creation (including reloads)
    discover_and_load_applications()

    enable_instrumentations()

    app = FastAPI(
        title="Pixie SDK Server",
        description="Server for running AI applications and agents",
        version="0.1.0",
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
    host: str = "127.0.0.1", port: int = 8000, reload: bool = False
) -> None:
    """Start the SDK server.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
    """
    from pixie.registry import list_applications

    # Setup logging (will be called again in create_app for reload scenarios)
    setup_logging()

    logger.info("Starting Pixie SDK Server on http://%s:%s", host, port)
    logger.info("GraphiQL interface: http://%s:%s/graphql", host, port)

    apps = list_applications()
    if apps:
        logger.info("Registered applications:")
        for app_id in apps:
            logger.info("  - %s", app_id)
    else:
        logger.warning("No applications registered yet")

    uvicorn.run(
        "pixie.server:create_app",
        host=host,
        port=port,
        loop="asyncio",
        reload=reload,
        factory=True,
        log_config=None,
    )
