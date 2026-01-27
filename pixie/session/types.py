from pixie.types import InputMixin, OutputMixin, RunStatus


class SessionUpdate(OutputMixin, InputMixin):
    session_id: str
    status: RunStatus
