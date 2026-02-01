"""Type definitions for session module."""

from pixie.types import InputMixin, OutputMixin, RunStatus


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
