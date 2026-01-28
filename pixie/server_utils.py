import logging

import colorlog


logger = logging.getLogger(__name__)

# Global logging mode
_logging_mode: str = "default"


def setup_logging(mode: str | None = None):
    """Configure logging for the entire application.

    Sets up colored logging with consistent formatting for all loggers.

    Args:
        mode: Logging mode - "default", "verbose", or "debug"
            - default: INFO for server events, WARNING+ for all modules
            - verbose: INFO+ for all modules
            - debug: DEBUG+ for all modules
    """
    global _logging_mode
    if mode is not None:
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
