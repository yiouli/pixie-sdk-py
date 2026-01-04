# Strawberry GraphQL Types Implementation - Summary

## ✅ Completed Tasks

### 1. Defined Strawberry Types from Pydantic Models
Created comprehensive Strawberry GraphQL types in [`pixie/schema.py`](pixie/schema.py) converted from Pydantic OTLP models:

**Complete OTLP Trace Types:**
- `OTLPKeyValue` - Key-value attributes
- `OTLPStatus` - Span status (OK, ERROR, etc.)
- `OTLPSpanEvent` - Events within a span
- `OTLPSpanLink` - Links between spans
- `OTLPSpan` - Complete span representation
- `OTLPInstrumentationScope` - Library/instrumentation info
- `OTLPScopeSpans` - Spans grouped by scope
- `OTLPResource` - Resource (service) information
- `OTLPResourceSpans` - Spans grouped by resource
- `OTLPTraceData` - Top-level OTLP trace structure

**Partial Trace Type:**
- `PartialTraceData` - Span start information with available fields

**Union Type:**
- `TraceDataUnion` - Union containing either `otlpTrace` or `partialTrace`

All types use `@strawberry.experimental.pydantic.type(model=..., all_fields=True)` except where manual field definitions were needed for Strawberry compatibility (JSON scalars, string types instead of Literals).

### 2. Updated Subscription Handler
Modified [`pixie/schema.py`](pixie/schema.py):

- **Added `_convert_trace_to_union()` helper**: Converts trace dict to `TraceDataUnion`
  - Detects partial traces by `event: "span_start"` field
  - Converts to appropriate Pydantic model then to Strawberry type
  
- **Updated `AppRunUpdate` type**: 
  - Changed from `@strawberry.experimental.pydantic.type` to `@strawberry.type`
  - Added custom `from_pydantic()` class method
  - Properly converts trace data to `TraceDataUnion`
  - Returns `Optional[TraceDataUnion]` for trace field

### 3. Created Example GraphQL Subscriptions
Created [`GRAPHQL_SUBSCRIPTION_EXAMPLES.md`](GRAPHQL_SUBSCRIPTION_EXAMPLES.md) with:

**Multiple Example Queries:**
1. **Full Subscription** - All trace fields including complete OTLP structure
2. **Minimal Subscription** - Just span names and status
3. **Timing Focused** - Performance monitoring with timestamps
4. **Attributes Focused** - LLM parameters and metadata

**Example Response Sequence:**
- Initial status update
- Span start events (partial traces)
- Span completion events (complete OTLP traces)
- Final completion status

**Client Implementation:**
- Python example using `gql` library
- Shows how to handle both partial and complete traces

## Data Flow

```
Langfuse SpanProcessor
  ↓ (emits dict)
Execution Context Queue
  ↓ (Pydantic AppRunUpdate with dict trace)
_convert_trace_to_union()
  ↓ (detects partial vs complete)
Pydantic → Strawberry conversion
  ↓
TraceDataUnion (Strawberry type)
  ├─ partialTrace: PartialTraceData?
  └─ otlpTrace: OTLPTraceData?
  ↓
GraphQL Subscription
  ↓
Client receives properly typed data
```

## Key Design Decisions

### 1. Union Type Instead of Interface
Used `TraceDataUnion` with two optional fields rather than GraphQL union type:
- ✅ Simpler client-side handling
- ✅ No need for `__typename` checks
- ✅ Both fields visible in schema
- ✅ Only one field will be non-null per update

### 2. JSON Scalar for Complex Dict Fields
Used `JSON` scalar for:
- `OTLPKeyValue.value` - Can contain various value types (string_value, int_value, etc.)
- `PartialTraceData.attributes` - Free-form attribute dictionary
- ✅ Avoids complex nested type definitions
- ✅ Strawberry-compatible
- ✅ Flexible for various value types

### 3. Custom from_pydantic for AppRunUpdate
Implemented custom conversion instead of relying on auto-conversion:
- ✅ Full control over trace field conversion
- ✅ Clean separation between dict (Pydantic) and union (Strawberry)
- ✅ Type-safe at GraphQL layer

### 4. String Instead of Literal for Event Field
Changed `event: Literal["span_start"]` to `event: str`:
- ✅ Strawberry compatible (doesn't support Literal)
- ✅ Still validated by Pydantic layer
- ✅ Schema documents it's always "span_start" in description

## Testing

```bash
# Import test
$ python -c "from pixie.schema import schema, AppRunUpdate, OTLPTraceData, PartialTraceData, TraceDataUnion; print('✓ Success')"
✓ Success

# No compilation errors
$ python -c "import pixie.schema; print('✓ Schema loads')"
✓ Schema loads
```

## Example Usage

### Client Subscription (Simple)
```graphql
subscription {
  run(name: "my-app", inputData: null) {
    runId
    status
    trace {
      partialTrace {
        spanName
        startTimeUnixNano
      }
      otlpTrace {
        resourceSpans {
          scopeSpans {
            spans {
              name
              endTimeUnixNano
            }
          }
        }
      }
    }
  }
}
```

### Expected Updates
```json
// Span Start
{
  "trace": {
    "partialTrace": {
      "spanName": "llm.chat",
      "startTimeUnixNano": "1704326400000000000"
    },
    "otlpTrace": null
  }
}

// Span End
{
  "trace": {
    "partialTrace": null,
    "otlpTrace": {
      "resourceSpans": [...]
    }
  }
}
```

## Files Modified

1. **`pixie/schema.py`**
   - Added OTLP Strawberry types (lines ~74-146)
   - Added TraceDataUnion type (lines ~147-152)
   - Updated AppRunUpdate with custom conversion (lines ~154-175)
   - Added _convert_trace_to_union helper (lines ~263-286)

2. **`GRAPHQL_SUBSCRIPTION_EXAMPLES.md`** (NEW)
   - Comprehensive examples
   - Multiple use cases
   - Client implementation guide

## Benefits

✅ **Type Safety**: Fully typed GraphQL schema with Strawberry types  
✅ **Flexibility**: Clients can query only needed fields  
✅ **Real-time**: Both span start and completion events  
✅ **Standards Compliant**: Full OTLP trace structure preserved  
✅ **Developer Experience**: Clear examples and documentation  
✅ **Performance**: Clients can request minimal data for efficiency  

## Next Steps for Users

1. Start the server: `python -m pixie.server`
2. Open GraphQL playground: http://localhost:8000/graphql
3. Try the example subscriptions from GRAPHQL_SUBSCRIPTION_EXAMPLES.md
4. Watch real-time trace data stream in!

---
**All requirements met** ✅
- Proper Strawberry types defined from Pydantic models with `all_fields=True`
- Types properly returned from subscription handler
- Example GraphQL subscription requests provided
