# Distributed Tracing, W3C TraceContext, and Latency Telemetry

## W3C TraceContext Propagation Standard
To diagnose latency tail degradation and isolate inter-service bottlenecks across microservices and partition shards, AetherDB adopts the W3C TraceContext specification. Every incoming client RPC is stamped with a standard HTTP/gRPC binary or text header:
```text
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
```
The header components encode:
- **Version**: 2 hexadecimal characters (currently `00`).
- **Trace ID**: 32 hexadecimal characters representing a globally unique 16-byte identifier shared across all causally related operations in the distributed call tree.
- **Parent ID (Span ID)**: 16 hexadecimal characters representing the immediate caller span identifier.
- **Trace Flags**: 2 hexadecimal characters encoding sampling directives (e.g., `01` denotes `recorded/sampled`).

## Head-Based vs Tail-Based Sampling
Capturing 100% of telemetry traces in high-throughput clusters generating millions of queries per second creates unsustainable storage and network costs. AetherDB implements dual sampling strategies:
1. **Head-Based Sampling**: The ingress gateway decides whether to record a trace at the moment of request origination based on a deterministic hash of the Trace ID and a sampling ratio `head_sample_rate = 0.01` (1%). The sampling decision is stamped into the trace flags and propagated downstream. While computationally inexpensive, head-based sampling frequently misses rare transient errors and outlier tail-latency spikes that manifest deep within internal storage layers.
2. **Tail-Based Sampling**: All intermediate spans are buffered in local ring buffers across cluster nodes. When an RPC terminates with an error code (e.g., `ERR_CLOCK_UNCERTAINTY_EXCEEDED` or `AetherErr_ServerOverloaded`) or when its total duration exceeds the p99 SLA threshold `tail_sample_latency_threshold_ms = 250`, the tail-sampler forces an emergency flush of the entire distributed trace tree to the telemetry collector.

## Baggage Items and Cross-Cutting Context
In addition to the immutable traceparent header, AetherDB supports the `tracestate` and `baggage` headers. Baggage carries key-value pairs (such as `tenant_id`, `priority_tier`, or `client_datacenter`) across asynchronous thread transitions and network boundaries. Storage schedulers inspect baggage attributes to assign CPU and disk I/O priorities, granting highest priority to VIP tenants while throttling bulk background batch tasks.
