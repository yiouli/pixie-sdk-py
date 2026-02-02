"""Type definitions for session module."""

from pydantic import BaseModel

from pixie.types import InputMixin, OutputMixin, RunStatus


class SessionInfo(BaseModel):
    """Information about a registered session.

    Captures metadata about the decorated function similar to how @app
    decorator captures information in the registry.

    Attributes:
        session_id: Unique identifier of the session.
        name: The function name.
        module: The module where the function is defined.
        qualname: The qualified name of the function.
        description: Optional description extracted from docstring.
    """

    session_id: str
    name: str
    module: str
    qualname: str
    description: str | None = None


class SessionUpdate(OutputMixin, InputMixin):
    """Status update for a session.

    Inherits user_input and user_input_schema from InputMixin,
    and data, trace, prompt_for_span from OutputMixin.

    Attributes:
        session_id: Unique identifier of the session.
        status: Current status of the session.
        time_unix_nano: Timestamp of the update in Unix nanoseconds.
    """

    session_id: str
    status: RunStatus
    time_unix_nano: str
