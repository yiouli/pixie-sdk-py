"""Database tables for run and LLM call records using Piccolo ORM."""

from piccolo.table import Table
from piccolo.columns import (
    Varchar,
    JSON,
    Timestamptz,
    BigInt,
)
from pixie.piccolo_conf import DB


class RunRecord(Table, db=DB):
    """Table for storing run records (app runs and session runs)."""

    id = Varchar(length=255, primary_key=True)
    source = Varchar(length=20)  # 'apps' or 'sessions'
    app_info = JSON(null=True)  # Serialized AppInfoRecord
    session_info = JSON(null=True)  # Serialized SessionInfoRecord
    messages = JSON(default="[]")  # Serialized list of Message
    prompt_ids = JSON(default="[]")  # List of prompt IDs
    start_time = BigInt(null=True)  # Unix timestamp in milliseconds
    end_time = BigInt(null=True)  # Unix timestamp in milliseconds
    rating = JSON(null=True)  # Serialized RatingDetails
    metadata = JSON(default="{}")
    created_at = Timestamptz()
    updated_at = Timestamptz()


class LlmCallRecord(Table, db=DB):
    """Table for storing LLM call records."""

    span_id = Varchar(length=255, primary_key=True)
    trace_id = Varchar(length=255)
    run_id = Varchar(length=255, null=True, index=True)  # Foreign key to RunRecord
    run_source = Varchar(length=20)  # 'apps' or 'sessions'
    app_info = JSON(null=True)  # Serialized AppInfoRecord
    session_info = JSON(null=True)  # Serialized SessionInfoRecord
    prompt_info = JSON(null=True)  # Serialized PromptInfoRecord (includes prompt_id)
    llm_input = JSON(null=True)
    llm_output = JSON(null=True)
    model_name = Varchar(length=255, default="unknown")
    model_parameters = JSON(null=True)
    start_time = BigInt(null=True)  # Unix timestamp in nanoseconds
    end_time = BigInt(null=True)  # Unix timestamp in nanoseconds
    internal_logs_after = JSON(default="[]")  # Logs after LLM call for evaluation
    rating = JSON(null=True)  # Serialized RatingDetails
    metadata = JSON(default="{}")
    created_at = Timestamptz()
    updated_at = Timestamptz()


async def create_tables():
    """Create tables in the database."""
    await RunRecord.create_table(if_not_exists=True)
    await LlmCallRecord.create_table(if_not_exists=True)
