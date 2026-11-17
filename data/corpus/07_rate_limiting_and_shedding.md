# Adaptive Rate Limiting, Load Shedding, and Backoff Strategies

## Token Bucket and Leaky Bucket Algorithms
To defend cluster nodes against cascading overload and resource exhaustion, AetherDB applies multi-tier traffic policing at the ingress proxy and RPC dispatcher levels. The primary algorithm employed is the Token Bucket algorithm with burst capacity. Tokens are continuously added to a bucket of capacity $B$ at a constant refill rate of $R$ tokens per second. An incoming request consuming $c$ tokens is immediately admitted if the bucket holds at least $c$ tokens; otherwise, the request is rejected with `HTTP 429 Too Many Requests` or placed in a bounded queue. In contrast, the Leaky Bucket algorithm enforces a strictly constant egress flow rate, smoothing out spiky traffic at the cost of queuing delay.

## Adaptive Load Shedding and Little's Law
When server memory, CPU saturation, or GC pause times spike, static rate limiting is insufficient. AetherDB employs an adaptive load shedding mechanism grounded in Little's Law:
$$L = \lambda W$$
where $L$ represents average concurrent requests in flight, $\lambda$ is arrival throughput, and $W$ is average request latency. Under server saturation, latency $W$ increases non-linearly, causing in-flight concurrency $L$ to balloon and queue buffers to overflow. The AetherDB load shedder continuously measures the queue residency time of incoming requests. If an RPC has spent longer than `max_acceptable_queue_latency_ms = 50` milliseconds waiting in the thread pool before execution begins, the engine immediately drops the request with `AetherErr_ServerOverloaded`, preserving CPU cycles for requests that still have a chance of satisfying client p99 latency SLA deadlines.

## Client-Side Exponential Backoff with Full Jitter
When client requests receive rate limit rejections or transient RPC timeouts, naive retries trigger catastrophic thundering herd problems that prevent cluster recovery. AetherDB client SDKs mandate exponential backoff with full jitter. For retry attempt $i$, the backoff duration is computed as:
$$\text{temp} = \min(T_{\text{max}}, T_{\text{base}} \times 2^i)$$
$$\text{sleep\_duration} = \text{UniformRandom}(0, \text{temp})$$
where $T_{\text{base}} = 50\text{ ms}$ and $T_{\text{max}} = 10000\text{ ms}$. Full jitter decorrelates retry pulses across thousands of concurrent clients, transforming synchronized retry spikes into a uniformly distributed, manageable background trickle.

