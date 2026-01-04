# OTel Trace Emission Implementation Documentation

## 1. Current Langfuse OTel Data Sending Mechanism

### Code Location
The OTel trace sending mechanism is implemented in:
- **Primary File**: `langfuse/_client/span_processor.py`
- **Class**: `LangfuseSpanProcessor` (extends `BatchSpanProcessor`)
- **Key Methods**:
  - `on_start()` - Called when a span starts
  - `on_end()` - Called when a span ends (line 155-181)
  - Inherited `export()` from `BatchSpanProcessor` which batches and sends spans

### Data Format
The data sent to Langfuse server is in **OTLP (OpenTelemetry Protocol) Protobuf format**:

1. **Encoding Process**:
   - Uses `OTLPSpanExporter` from OpenTelemetry SDK
   - Calls `encode_spans(spans)` which returns a `ExportTraceServiceRequest` protobuf message
   - The protobuf structure is:
     ```
     ExportTraceServiceRequest
       └── resource_spans: List[ResourceSpans]
             └── resource: Resource (SDK info, env, etc.)
             └── scope_spans: List[ScopeSpans]
                   └── scope: InstrumentationScope (library info)
                   └── spans: List[Span]
                         ├── trace_id
                         ├── span_id
                         ├── parent_span_id
                         ├── name
                         ├── kind (INTERNAL, SERVER, CLIENT, etc.)
                         ├── start_time_unix_nano
                         ├── end_time_unix_nano
                         ├── attributes (key-value pairs)
                         ├── events
                         ├── links
                         ├── status
                         └── dropped_*_count fields
     ```

2. **Serialization**:
   - The protobuf message is serialized to bytes: `serialized_data = encode_spans(spans).SerializePartialToString()`
   - Optionally compressed (gzip/deflate)
   - Sent via HTTP POST to endpoint: `{base_url}/api/public/otel/v1/traces`

3. **JSON Format**:
   - To convert to JSON (for GraphQL subscription), we need to use:
     ```python
     from google.protobuf.json_format import MessageToDict
     proto_message = encode_spans(spans)
     json_data = MessageToDict(proto_message, preserving_proto_field_name=True)
     ```
   - This produces a JSON object matching the protobuf structure above

### Timing
Spans are sent to the Langfuse server:
1. **Batching**: Spans are collected in batches (controlled by `max_export_batch_size` / `flush_at`)
2. **Scheduled Export**: Batches are exported at regular intervals (controlled by `schedule_delay_millis` / `flush_interval`)
3. **On Flush**: When `flush()` is called on the tracer or client
4. **On Shutdown**: When the span processor is shutdown

The actual export happens asynchronously in a background thread managed by `BatchSpanProcessor`.

### HTTP Request Details
- **Endpoint**: `{base_url}/api/public/otel/v1/traces`
- **Method**: POST
- **Headers**:
  - `Authorization: Basic {base64(public_key:secret_key)}`
  - `x-langfuse-sdk-name: python`
  - `x-langfuse-sdk-version: {version}`
  - `x-langfuse-public-key: {public_key}`
  - `Content-Type: application/x-protobuf` (implied by OTLP)
- **Body**: Protobuf serialized `ExportTraceServiceRequest`

---

## 2. Implementation Plan

### Architecture Overview

We will modify the `LangfuseSpanProcessor` to emit trace data to the GraphQL subscription via the existing execution context queue, while maintaining the original functionality of sending to Langfuse server.

### Key Components

#### A. Modified `LangfuseSpanProcessor` (`langfuse/_client/span_processor.py`)
**Goal**: Capture trace data and emit to execution context queue

**Approach**: Override the `export()` method or create a wrapper around span export to:
1. Let the original export to Langfuse server proceed unchanged
2. Before/after the export, convert the span batch to JSON format
3. Emit the JSON trace data to the execution context queue

**Design Choice**: We'll override the `on_end()` method to capture individual spans and batch them ourselves for emission, since:
- Individual span emission provides real-time updates
- We can serialize each span as it completes
- We maintain backward compatibility with existing batch export to server

#### B. Updated `ExecutionContext` and `AppRunUpdate` Types (`pixie/types.py`)
**Goal**: Support trace data in status updates

**Changes**:
- `AppRunUpdate.trace` field already exists (line 82) - we'll use this
- Type should be `Optional[dict]` or `Any` to hold JSON trace data

#### C. Execution Context Queue Integration (`pixie/execution_context.py`)
**Goal**: Provide method to emit trace updates

**Changes**:
- Potentially add a helper method `emit_trace_update()` or use existing `emit_status_update()` with trace data
- The existing `emit_status_update()` already supports optional parameters

#### D. GraphQL Schema Updates (`pixie/schema.py`)
**Goal**: Yield trace data in subscription

**Changes**:
- The `AppRunUpdate` already has a `trace` field (line 85)
- The subscription already yields `AppRunUpdate` objects
- Minimal or no changes needed - traces will automatically flow through

### Implementation Steps

1. **Add Trace Emission to LangfuseSpanProcessor**:
   - Import execution context module
   - In `on_end()` method, after span processing:
     - Check if execution context exists
     - Convert span to JSON format using protobuf JSON encoder
     - Emit to status queue with trace data
   - Handle cases where no execution context exists (normal operation)

2. **Update Type Annotations**:
   - Ensure `AppRunUpdate.trace` is properly typed as `Optional[dict]`
   - Update JSON type in GraphQL schema if needed

3. **Test the Flow**:
   - Verify traces are captured during app execution
   - Verify traces are sent to Langfuse server (existing functionality)
   - Verify traces are emitted to GraphQL subscription
   - Verify JSON format matches OTLP structure

### Design Decisions

1. **Where to Hook**: `on_end()` method
   - ✅ Called for every span completion
   - ✅ Has full span data including timing
   - ✅ Doesn't interfere with batching for server export
   - ❌ May create more queue items (acceptable for real-time updates)

2. **Data Format**: OTLP JSON (via MessageToDict)
   - ✅ Standard OpenTelemetry format
   - ✅ Same as what Langfuse server receives
   - ✅ Well-documented and structured
   - ✅ Easy to convert: protobuf -> JSON

3. **Span vs Batch Emission**:
   - **Choice**: Emit individual spans as they complete
   - ✅ Real-time visibility into execution
   - ✅ Simpler implementation
   - ✅ Natural fit with streaming GraphQL subscription
   - ❌ More queue messages (acceptable trade-off)

4. **Error Handling**:
   - Wrap emission in try-except to prevent trace emission failures from breaking span export to server
   - Log errors but continue normal operation

5. **Performance Considerations**:
   - JSON serialization is relatively fast for individual spans
   - Queue operations are async and non-blocking
   - No significant performance impact expected

### Potential Issues and Mitigations

1. **Issue**: Execution context not available
   - **Mitigation**: Check for context existence before emitting

2. **Issue**: Large spans causing queue backlog
   - **Mitigation**: Queue is async and should handle this; monitor in testing

3. **Issue**: JSON serialization failures
   - **Mitigation**: Wrap in try-except, log and continue

4. **Issue**: Breaking existing Langfuse export
   - **Mitigation**: Only add new code, don't modify existing export logic

### Success Criteria

1. ✅ Traces appear in GraphQL subscription in real-time
2. ✅ Trace format matches OTLP JSON structure
3. ✅ Existing Langfuse server export continues working
4. ✅ No performance degradation
5. ✅ Graceful handling when execution context is not available
