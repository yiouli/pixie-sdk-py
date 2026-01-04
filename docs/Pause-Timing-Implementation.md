# Pause Timing Implementation

## Overview

This implementation ensures that pause periods during span execution are NOT counted in the span's duration, and that pause-related trace data is emitted to both Langfuse and the GraphQL subscription queue.

## Key Features

### 1. Pause Duration Exclusion from Spans

#### BEFORE Pauses (pause before span starts)
- When a breakpoint triggers BEFORE a span starts, the pause occurs before the span begins actual work
- The span's `start_time` is automatically adjusted forward by the pause duration
- This ensures the pause time is completely excluded from the span's duration
- The adjustment is tracked via span attributes:
  - `pixie.pause.duration_ns`: The pause duration in nanoseconds
  - `pixie.pause.adjusted`: Boolean flag indicating the span timing was adjusted

#### AFTER Pauses (pause after span ends)
- When a breakpoint triggers AFTER a span ends, the pause occurs after the span has completed
- The span's `end_time` is already set, so the pause naturally doesn't affect the span duration
- Pause events are still emitted for tracking purposes

### 2. Pause Event Emission

The implementation emits detailed pause events to two destinations:

#### A. GraphQL Subscription Queue (via execution_context)
Pause events are emitted as status updates containing:
```python
{
    "event": "pause_start" | "pause_end",
    "timestamp_ns": <nanosecond timestamp>,
    "span_name": <name of the span where pause occurred>,
    "trace_id": <OpenTelemetry trace ID>,
    "span_id": <OpenTelemetry span ID>,
    "parent_span_id": <parent span ID if available>,
    "pause_duration_ns": <duration in nanoseconds> (for pause_end only),
    "pause_duration_seconds": <duration in seconds> (for pause_end only),
    "span_attributes": <dict of span attributes>
}
```

#### B. Langfuse (via OpenTelemetry span events)
Pause events are added as OpenTelemetry span events with attributes:
- Event name: `pixie.pause_start` or `pixie.pause_end`
- Attributes:
  - `pixie.event.type`: "pause_start" or "pause_end"
  - `pixie.pause.timestamp_ns`: Event timestamp
  - `pixie.pause.duration_ns`: Pause duration (for end events)
  - `pixie.pause.duration_seconds`: Pause duration in seconds (for end events)

These events are automatically exported to Langfuse along with the span data.

## Implementation Details

### Modified Methods

#### `_check_breakpoint(span, is_before)`
Enhanced to:
1. Record pause start time before pausing
2. Emit `pause_start` event
3. Wait for resume (blocks execution)
4. Record pause end time after resume
5. Calculate pause duration
6. Adjust span timing for BEFORE pauses
7. Emit `pause_end` event with duration

#### `_emit_pause_event(span, event_type, event_time_ns, pause_duration_ns)`
New method that:
1. Creates pause event data structure
2. Emits to execution context queue for GraphQL subscription
3. Adds OpenTelemetry span event (exported to Langfuse)

### Technical Approach

#### Span Start Time Adjustment
For BEFORE pauses, we access the OpenTelemetry span's internal `_start_time` attribute:
```python
# pylint: disable=protected-access
if hasattr(span, "_start_time"):
    original_start = span._start_time
    span._start_time = original_start + pause_duration_ns
# pylint: enable=protected-access
```

This is necessary because OpenTelemetry's public API doesn't provide a way to adjust span timing after creation. The `_start_time` is stored in nanoseconds since epoch.

#### Time Measurement
All time measurements use `time.time_ns()` to match OpenTelemetry's nanosecond precision.

## Testing

A comprehensive test suite in `tests/pixie/test_pause_timing.py` verifies:
- Pause BEFORE span adjusts start time correctly
- Pause duration is recorded accurately
- Span attributes are set properly
- Pause events are emitted with correct data

## Example Usage

```python
from pixie import execution_context as exec_ctx

# Initialize execution context
run_id = "my-run-123"
ctx = exec_ctx.init_run(run_id)

# Set breakpoint to pause BEFORE LLM calls
exec_ctx.set_breakpoint(run_id, timing="BEFORE", types=["LLM"])

# When an LLM span starts, execution will pause
# The pause duration will NOT be counted in the span's duration
# Pause events will be emitted to both Langfuse and the subscription queue

# Resume execution
exec_ctx.resume_run(run_id)
```

## Benefits

1. **Accurate Performance Metrics**: Span durations reflect actual work time, not debug/pause time
2. **Debug Transparency**: Pause events are fully tracked and visible in traces
3. **Dual Emission**: Pause data is available both in real-time (subscription) and in Langfuse traces
4. **Minimal Overhead**: Pause tracking adds negligible overhead when not paused
