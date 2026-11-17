# Tunable Quorum Consistency and Read Repair Mechanics

## Quorum Equations and Strong Consistency Invariants
AetherDB implements Dynamo-style tunable quorum consistency for non-transactional key-value workloads. The cluster configuration defines a replication factor $N$ (the total number of replica nodes designated to store a given key), a write quorum $W$ (the minimum number of replica acknowledgments required for a successful write), and a read quorum $R$ (the minimum number of replica responses required for a successful read). Strong linearizable consistency is formally guaranteed when the quorum equation satisfies:
$$W + R > N$$
When $W + R > N$, the Pigeonhole Principle dictates that the set of nodes acknowledging a write and the set of nodes participating in a read must overlap by at least one node. This overlap guarantees that at least one responding node holds the most up-to-date version of the requested key. Conversely, if $W + R \le N$, the system operates in weak or eventual consistency mode.

## Read Repair and Anti-Entropy
When servicing a quorum read with $R > 1$, the coordinating node requests full data payloads from one replica and cryptographic SHA-256 hashes from the remaining $R - 1$ replicas to conserve internal network bandwidth. If all received version hashes match, the coordinator immediately returns the result to the client. If a digest mismatch is detected, the coordinator initiates **Read Repair**:
1. It requests the full data payload from all $R$ replicas.
2. It identifies the most recent value by comparing timestamp versions or vector clocks.
3. In **Synchronous Read Repair**, the coordinator writes the authoritative newest value to all lagging replicas before responding to the client, trading read latency for immediate replica synchronization.
4. In **Asynchronous Read Repair**, the coordinator returns the freshest version to the client immediately and submits background write tasks to update stale replicas asynchronously.

## Sloppy Quorums and Hinted Handoff
Under severe network partitioning where fewer than $W$ designated primary replicas are reachable, AetherDB can be configured to degrade into a **Sloppy Quorum** mode if `allow_sloppy_quorum = true`. Instead of rejecting the write outright, the coordinator routes the write to temporary surrogate nodes outside the standard replica set. The surrogate node accepts the record and writes it into a dedicated local storage table known as a **Hinted Handoff** buffer. Each hinted handoff entry stores the original destination node ID, target key, and value. Background worker threads continuously probe the partitioned node; once connectivity is restored, all accumulated hints are replayed to the primary replica and cleared.

