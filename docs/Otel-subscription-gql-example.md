# Example GraphQL Subscription for Trace Data

## Basic Subscription with Full Trace Data

This subscription receives all status updates including complete OTLP trace data and partial span start events:

```graphql
subscription RunWithTraces {
  run(name: "my-app", inputData: null) {
    runId
    status
    data

    # Trace data - union type with either complete OTLP or partial trace
    trace {
      # For partial trace (span start events)
      partialTrace {
        event
        spanName
        traceId
        spanId
        parentSpanId
        startTimeUnixNano
        kind
        attributes
      }

      # For complete OTLP trace (span completion)
      otlpTrace {
        resourceSpans {
          resource {
            attributes {
              key
              value
            }
            droppedAttributesCount
          }
          scopeSpans {
            scope {
              name
              version
              attributes {
                key
                value
              }
              droppedAttributesCount
            }
            spans {
              traceId
              spanId
              traceState
              parentSpanId
              name
              kind
              startTimeUnixNano
              endTimeUnixNano
              attributes {
                key
                value
              }
              events {
                name
                timeUnixNano
                attributes {
                  key
                  value
                }
                droppedAttributesCount
              }
              links {
                traceId
                spanId
                traceState
                attributes {
                  key
                  value
                }
                droppedAttributesCount
                flags
              }
              status {
                code
                message
              }
              droppedAttributesCount
              droppedEventsCount
              droppedLinksCount
              flags
            }
            schemaUrl
          }
          schemaUrl
        }
      }
    }

    # Breakpoint information (for pause/resume)
    breakpoint {
      spanName
      breakpointType
      breakpointTiming
      spanAttributes
    }
  }
}
```

## Minimal Subscription (Just Status and Span Names)

For a lighter subscription that only tracks span names and status:

```graphql
subscription RunMinimal {
  run(name: "my-app", inputData: null) {
    runId
    status
    trace {
      partialTrace {
        spanName
      }
      otlpTrace {
        resourceSpans {
          scopeSpans {
            spans {
              name
              kind
            }
          }
        }
      }
    }
  }
}
```

## Focused on Timing Data

To track just the timing information for performance monitoring:

```graphql
subscription RunTiming {
  run(name: "my-app", inputData: { "input": "test" }) {
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
              startTimeUnixNano
              endTimeUnixNano
              kind
            }
          }
        }
      }
    }
  }
}
```

## Attribute-Focused Subscription

To monitor span attributes (e.g., LLM model parameters, API calls):

```graphql
subscription RunAttributes {
  run(name: "my-app", inputData: null) {
    runId
    status
    trace {
      partialTrace {
        spanName
        attributes
      }
      otlpTrace {
        resourceSpans {
          scopeSpans {
            spans {
              name
              attributes {
                key
                value
              }
              status {
                code
                message
              }
            }
          }
        }
      }
    }
  }
}
```

## Example Response Sequence

When you run a subscription, you'll receive a sequence of updates:

### 1. Initial Status (No Trace)
```json
{
  "data": {
    "run": {
      "runId": "550e8400-e29b-41d4-a716-446655440000",
      "status": "RUNNING",
      "data": "{\"run_id\": \"550e8400-e29b-41d4-a716-446655440000\"}",
      "trace": null,
      "breakpoint": null
    }
  }
}
```

### 2. Span Start (Partial Trace)
```json
{
  "data": {
    "run": {
      "runId": "550e8400-e29b-41d4-a716-446655440000",
      "status": "RUNNING",
      "data": null,
      "trace": {
        "partialTrace": {
          "event": "span_start",
          "spanName": "llm.chat.openai",
          "traceId": "0a1b2c3d4e5f67890123456789abcdef",
          "spanId": "1234567890abcdef",
          "parentSpanId": null,
          "startTimeUnixNano": "1704326400000000000",
          "kind": "CLIENT",
          "attributes": {
            "model": "gpt-4",
            "temperature": 0.7
          }
        },
        "otlpTrace": null
      },
      "breakpoint": null
    }
  }
}
```

### 3. Span End (Complete OTLP Trace)
```json
{
  "data": {
    "run": {
      "runId": "550e8400-e29b-41d4-a716-446655440000",
      "status": "RUNNING",
      "data": null,
      "trace": {
        "partialTrace": null,
        "otlpTrace": {
          "resourceSpans": [
            {
              "resource": {
                "attributes": [
                  {"key": "service.name", "value": {"stringValue": "my-app"}},
                  {"key": "telemetry.sdk.name", "value": {"stringValue": "opentelemetry"}}
                ],
                "droppedAttributesCount": 0
              },
              "scopeSpans": [
                {
                  "scope": {
                    "name": "langfuse-tracer",
                    "version": "1.0.0"
                  },
                  "spans": [
                    {
                      "traceId": "0a1b2c3d4e5f67890123456789abcdef",
                      "spanId": "1234567890abcdef",
                      "name": "llm.chat.openai",
                      "kind": 3,
                      "startTimeUnixNano": "1704326400000000000",
                      "endTimeUnixNano": "1704326402500000000",
                      "attributes": [
                        {"key": "model", "value": {"stringValue": "gpt-4"}},
                        {"key": "temperature", "value": {"doubleValue": 0.7}}
                      ],
                      "status": {"code": 0},
                      "events": [],
                      "links": [],
                      "droppedAttributesCount": 0,
                      "droppedEventsCount": 0,
                      "droppedLinksCount": 0
                    }
                  ]
                }
              ]
            }
          ]
        }
      },
      "breakpoint": null
    }
  }
}
```

### 4. Completion
```json
{
  "data": {
    "run": {
      "runId": "550e8400-e29b-41d4-a716-446655440000",
      "status": "COMPLETED",
      "data": null,
      "trace": null,
      "breakpoint": null
    }
  }
}
```

## Testing with GraphQL Playground

1. Start your server:
   ```bash
   python -m pixie.server
   ```

2. Navigate to http://localhost:8000/graphql (or your configured endpoint)

3. Paste one of the subscription queries above

4. Click "Execute" to start the subscription

5. Watch real-time trace data stream in as your application executes!

## Notes

- **Partial Trace**: Emitted when a span **starts** - contains minimal info available at that moment
- **OTLP Trace**: Emitted when a span **ends** - contains complete span data including timing and status
- **Union Type**: The `trace` field is a union - only one of `partialTrace` or `otlpTrace` will be non-null in each update
- **SpanKind Values**:
  - 0 = UNSPECIFIED
  - 1 = INTERNAL
  - 2 = SERVER
  - 3 = CLIENT
  - 4 = PRODUCER
  - 5 = CONSUMER

## Client Implementation Example (Python)

```python
import asyncio
from gql import gql, Client
from gql.transport.websockets import WebsocketsTransport

async def subscribe_to_run():
    transport = WebsocketsTransport(url="ws://localhost:8000/graphql")
    async with Client(transport=transport) as session:
        subscription = gql('''
            subscription {
              run(name: "my-app", inputData: null) {
                runId
                status
                trace {
                  partialTrace { spanName }
                  otlpTrace {
                    resourceSpans {
                      scopeSpans {
                        spans { name }
                      }
                    }
                  }
                }
              }
            }
        ''')

        async for result in session.subscribe(subscription):
            print(f"Update: {result}")

            trace = result["run"]["trace"]
            if trace:
                if trace["partialTrace"]:
                    print(f"→ Span started: {trace['partialTrace']['spanName']}")
                elif trace["otlpTrace"]:
                    spans = trace["otlpTrace"]["resourceSpans"][0]["scopeSpans"][0]["spans"]
                    for span in spans:
                        print(f"✓ Span completed: {span['name']}")

asyncio.run(subscribe_to_run())
```
