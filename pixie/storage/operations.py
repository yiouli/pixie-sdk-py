"""Database operations for run and LLM call records."""

import json
from datetime import datetime, timezone
from typing import Any, Optional, TypeVar, cast, overload

from pydantic import BaseModel
from piccolo.columns import Column

from .tables import RunRecord as RunRecordTable, LlmCallRecord as LlmCallRecordTable
from .types import (
    RunRecord,
    RunRecordInput,
    RunRecordUpdate,
    LlmCallRecord,
    LlmCallRecordInput,
    LlmCallRecordUpdate,
    RecordFilters,
    Message,
    RatingDetails,
    AppInfoRecord,
    SessionInfoRecord,
    PromptInfoRecord,
)


# Type variable for Pydantic model classes
T = TypeVar("T", bound=BaseModel)


def _now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def _serialize_json(obj: Any) -> str | None:
    """Serialize a Pydantic model or dict to JSON string."""
    if obj is None:
        return None
    if isinstance(obj, BaseModel):
        return json.dumps(obj.model_dump())
    return json.dumps(obj)


# Overload 1: When model_class is provided, return that type or None
@overload
def _parse_json(data: str | dict | None, model_class: type[T]) -> T | None: ...


# Overload 2: When model_class is None, return dict or None
@overload
def _parse_json(data: str | dict | None, model_class: None = None) -> dict | None: ...


def _parse_json(
    data: str | dict | None, model_class: type[T] | None = None
) -> T | dict | None:
    """Parse JSON string or dict to a Pydantic model or dict.

    Args:
        data: JSON string or dict to parse
        model_class: Optional Pydantic model class to validate against

    Returns:
        - If data is None: returns None
        - If model_class is provided: returns validated model instance or None
        - If model_class is None: returns parsed dict or None
    """
    if data is None:
        return None
    parsed: dict | None
    if isinstance(data, str):
        parsed = json.loads(data)
    else:
        parsed = data
    if model_class and parsed:
        return model_class.model_validate(parsed)
    return parsed


def _parse_messages(data: str | list | None) -> list[Message]:
    """Parse messages from JSON string or list."""
    if data is None:
        return []
    parsed: list
    if isinstance(data, str):
        parsed = json.loads(data)
    else:
        parsed = data if data else []
    return [Message.model_validate(m) for m in parsed]


def _parse_rating(data: str | dict | None) -> RatingDetails | None:
    """Parse rating details from JSON string or dict."""
    return _parse_json(data, RatingDetails)


def _parse_prompt_ids(data: str | list | None) -> list[str]:
    """Parse prompt IDs from JSON string or list."""
    if data is None:
        return []
    if isinstance(data, str):
        return json.loads(data)
    return data


def _row_to_run_record(row: dict) -> RunRecord:
    """Convert a database row dict to a RunRecord."""
    return RunRecord(
        id=row["id"],
        source=row["source"],
        app_info=_parse_json(row["app_info"], AppInfoRecord),
        session_info=_parse_json(row["session_info"], SessionInfoRecord),
        messages=_parse_messages(row["messages"]),
        prompt_ids=_parse_prompt_ids(row["prompt_ids"]),
        start_time=row["start_time"],
        end_time=row["end_time"],
        rating=_parse_rating(row["rating"]),
        metadata=_parse_json(row["metadata"]) or {},
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_llm_call_record(row: dict) -> LlmCallRecord:
    """Convert a database row dict to an LlmCallRecord."""
    return LlmCallRecord(
        span_id=row["span_id"],
        trace_id=row["trace_id"],
        run_id=row["run_id"],
        run_source=row["run_source"],
        app_info=_parse_json(row["app_info"], AppInfoRecord),
        session_info=_parse_json(row["session_info"], SessionInfoRecord),
        prompt_info=_parse_json(row["prompt_info"], PromptInfoRecord),
        llm_input=_parse_json(row["llm_input"]),
        llm_output=_parse_json(row["llm_output"]),
        model_name=row["model_name"],
        model_parameters=_parse_json(row["model_parameters"]),
        start_time=row["start_time"],
        end_time=row["end_time"],
        internal_logs_after=(
            logs
            if isinstance((logs := _parse_json(row["internal_logs_after"])), list)
            else []
        ),
        rating=_parse_rating(row["rating"]),
        metadata=_parse_json(row["metadata"]) or {},
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# ============================================================================
# Run Record Operations
# ============================================================================


async def create_run_record(input_data: RunRecordInput) -> RunRecord:
    """Create a new run record."""
    now = _now()

    record = RunRecordTable(
        id=input_data.id,
        source=input_data.source,
        app_info=_serialize_json(input_data.app_info),
        session_info=_serialize_json(input_data.session_info),
        messages=json.dumps([m.model_dump() for m in input_data.messages]),
        prompt_ids=json.dumps(input_data.prompt_ids),
        start_time=input_data.start_time,
        end_time=None,
        rating=None,
        metadata=json.dumps(input_data.metadata),
        created_at=now,
        updated_at=now,
    )

    await record.save()
    # Convert ORM object to dict for processing
    saved_rows = await RunRecordTable.select().where(RunRecordTable.id == input_data.id)
    return _row_to_run_record(saved_rows[0])


async def get_run_record(run_id: str) -> Optional[RunRecord]:
    """Get a run record by ID."""
    rows = await RunRecordTable.select().where(RunRecordTable.id == run_id)
    if not rows:
        return None
    return _row_to_run_record(rows[0])


async def update_run_record(
    run_id: str, update_data: RunRecordUpdate
) -> Optional[RunRecord]:
    """Update an existing run record with partial data."""
    rows = await RunRecordTable.select().where(RunRecordTable.id == run_id)
    if not rows:
        return None

    record = rows[0]
    now = _now()

    # Build update dict with only provided fields
    updates: dict[str, Any] = {"updated_at": now}

    if update_data.messages is not None:
        # Append new messages to existing
        existing_messages = _parse_messages(record["messages"])
        all_messages = existing_messages + update_data.messages
        updates["messages"] = json.dumps([m.model_dump() for m in all_messages])

    if update_data.prompt_ids is not None:
        # Merge prompt IDs (union)
        existing_ids = set(_parse_prompt_ids(record["prompt_ids"]))
        new_ids = existing_ids | set(update_data.prompt_ids)
        updates["prompt_ids"] = json.dumps(list(new_ids))

    if update_data.end_time is not None:
        updates["end_time"] = update_data.end_time

    if update_data.rating is not None:
        updates["rating"] = _serialize_json(update_data.rating)

    if update_data.metadata is not None:
        # Merge metadata
        existing_meta = _parse_json(record["metadata"]) or {}
        existing_meta.update(update_data.metadata)
        updates["metadata"] = json.dumps(existing_meta)

    await RunRecordTable.update(cast(dict[Column | str, Any], updates)).where(
        RunRecordTable.id == run_id
    )

    # Return updated record
    return await get_run_record(run_id)


async def get_run_records(filters: RecordFilters) -> list[RunRecord]:
    """Get run records with optional filters."""
    query = RunRecordTable.select()

    if filters.app_id:
        # Filter by app_id in app_info JSON
        query = query.where(RunRecordTable.app_info.like(f'%"id": "{filters.app_id}"%'))

    if filters.run_source:
        query = query.where(RunRecordTable.source == filters.run_source)

    if filters.has_rating is not None:
        if filters.has_rating:
            query = query.where(RunRecordTable.rating.is_not_null())
        else:
            query = query.where(RunRecordTable.rating.is_null())

    if filters.rating_value:
        query = query.where(
            RunRecordTable.rating.like(f'%"value": "{filters.rating_value}"%')
        )

    if filters.created_after:
        query = query.where(RunRecordTable.created_at >= filters.created_after)

    if filters.created_before:
        query = query.where(RunRecordTable.created_at <= filters.created_before)

    query = query.order_by(RunRecordTable.created_at, ascending=False)
    query = query.limit(filters.limit).offset(filters.offset)

    rows = await query
    return [_row_to_run_record(row) for row in rows]


# ============================================================================
# LLM Call Record Operations
# ============================================================================


async def create_llm_call_record(input_data: LlmCallRecordInput) -> LlmCallRecord:
    """Create a new LLM call record."""
    now = _now()

    record = LlmCallRecordTable(
        span_id=input_data.span_id,
        trace_id=input_data.trace_id,
        run_id=input_data.run_id,
        run_source=input_data.run_source,
        app_info=_serialize_json(input_data.app_info),
        session_info=_serialize_json(input_data.session_info),
        prompt_info=_serialize_json(input_data.prompt_info),
        llm_input=_serialize_json(input_data.llm_input),
        llm_output=_serialize_json(input_data.llm_output),
        model_name=input_data.model_name,
        model_parameters=_serialize_json(input_data.model_parameters),
        start_time=input_data.start_time,
        end_time=input_data.end_time,
        internal_logs_after=json.dumps([]),
        rating=None,
        metadata=json.dumps(input_data.metadata),
        created_at=now,
        updated_at=now,
    )

    await record.save()
    # Convert ORM object to dict for processing
    saved_rows = await LlmCallRecordTable.select().where(
        LlmCallRecordTable.span_id == input_data.span_id
    )
    return _row_to_llm_call_record(saved_rows[0])


async def get_llm_call_record(span_id: str) -> Optional[LlmCallRecord]:
    """Get an LLM call record by span ID."""
    rows = await LlmCallRecordTable.select().where(
        LlmCallRecordTable.span_id == span_id
    )
    if not rows:
        return None
    return _row_to_llm_call_record(rows[0])


async def update_llm_call_record(
    span_id: str, update_data: LlmCallRecordUpdate
) -> Optional[LlmCallRecord]:
    """Update an existing LLM call record with partial data."""
    rows = await LlmCallRecordTable.select().where(
        LlmCallRecordTable.span_id == span_id
    )
    if not rows:
        return None

    record = rows[0]
    now = _now()

    updates: dict[str, Any] = {"updated_at": now}

    if update_data.prompt_info is not None:
        updates["prompt_info"] = _serialize_json(update_data.prompt_info)

    if update_data.llm_input is not None:
        updates["llm_input"] = _serialize_json(update_data.llm_input)

    if update_data.llm_output is not None:
        updates["llm_output"] = _serialize_json(update_data.llm_output)

    if update_data.model_name is not None:
        updates["model_name"] = update_data.model_name

    if update_data.model_parameters is not None:
        updates["model_parameters"] = _serialize_json(update_data.model_parameters)

    if update_data.end_time is not None:
        updates["end_time"] = update_data.end_time

    if update_data.internal_logs_after is not None:
        # Append new logs to existing
        existing_logs_raw = _parse_json(record["internal_logs_after"])
        existing_logs: list = (
            existing_logs_raw if isinstance(existing_logs_raw, list) else []
        )
        all_logs = existing_logs + update_data.internal_logs_after
        updates["internal_logs_after"] = json.dumps(all_logs)

    if update_data.rating is not None:
        updates["rating"] = _serialize_json(update_data.rating)

    if update_data.metadata is not None:
        existing_meta = _parse_json(record["metadata"]) or {}
        existing_meta.update(update_data.metadata)
        updates["metadata"] = json.dumps(existing_meta)

    await LlmCallRecordTable.update(cast(dict[Column | str, Any], updates)).where(
        LlmCallRecordTable.span_id == span_id
    )

    return await get_llm_call_record(span_id)


async def get_llm_call_records(filters: RecordFilters) -> list[LlmCallRecord]:
    """Get LLM call records with optional filters."""
    query = LlmCallRecordTable.select()

    if filters.prompt_id:
        # Filter by prompt_id in prompt_info JSON
        query = query.where(
            LlmCallRecordTable.prompt_info.like(f'%"prompt_id": "{filters.prompt_id}"%')
        )

    if filters.app_id:
        query = query.where(
            LlmCallRecordTable.app_info.like(f'%"id": "{filters.app_id}"%')
        )

    if filters.run_source:
        query = query.where(LlmCallRecordTable.run_source == filters.run_source)

    if filters.has_rating is not None:
        if filters.has_rating:
            query = query.where(LlmCallRecordTable.rating.is_not_null())
        else:
            query = query.where(LlmCallRecordTable.rating.is_null())

    if filters.rating_value:
        query = query.where(
            LlmCallRecordTable.rating.like(f'%"value": "{filters.rating_value}"%')
        )

    if filters.created_after:
        query = query.where(LlmCallRecordTable.created_at >= filters.created_after)

    if filters.created_before:
        query = query.where(LlmCallRecordTable.created_at <= filters.created_before)

    query = query.order_by(LlmCallRecordTable.created_at, ascending=False)
    query = query.limit(filters.limit).offset(filters.offset)

    rows = await query
    return [_row_to_llm_call_record(row) for row in rows]


async def get_llm_call_records_by_run(run_id: str) -> list[LlmCallRecord]:
    """Get all LLM call records for a specific run."""
    rows = await (
        LlmCallRecordTable.select()
        .where(LlmCallRecordTable.run_id == run_id)
        .order_by(LlmCallRecordTable.start_time, ascending=True)
    )
    return [_row_to_llm_call_record(row) for row in rows]
