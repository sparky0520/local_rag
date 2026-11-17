# Distributed Transactions and Two-Phase Commit Protocol

## Multi-Version Concurrency Control (MVCC) and Timestamp Ordering
AetherDB provides Snapshot Isolation (SI) and Serializable Snapshot Isolation (SSI) for multi-partition distributed transactions using Multi-Version Concurrency Control (MVCC). Every transaction is assigned a monotonically increasing logical start timestamp `read_ts` by a centralized or distributed timestamp oracle. Reads observe the latest committed record version whose `commit_ts <= read_ts`, guaranteeing non-blocking repeatable reads without holding shared read locks. Each record stores a version chain containing `(value, created_ts, expired_ts)` to support point-in-time snapshot reconstruction.

## Two-Phase Commit (2PC) Execution Model
For cross-shard transactions spanning multiple independent partition leaders, atomic commit is governed by the Two-Phase Commit (2PC) protocol. The client or an appointed coordinator node initiates the transaction lifecycle:
1. **Prepare Phase**: The coordinator sends `PREPARE` messages along with proposed write mutations to all participating shard leaders. Each participant acquires local write intents in its `LockTable`, logs a durable `PREPARED` record to its local WAL, and replies with either `PREPARED_OK` or `ABORT`.
2. **Commit Phase**: If all participants reply with `PREPARED_OK`, the coordinator logs a durable `COMMITTED` record to its transaction log and broadcasts `COMMIT` messages with the allocated `commit_ts`. If any participant votes `ABORT` or times out, the coordinator broadcasts `ROLLBACK`.

## Coordinator Failure and Presumed Abort Protocol
The primary vulnerability of classical 2PC is the blocking coordinator failure scenario: if the coordinator crashes after participants have prepared but before broadcasting the decision, participants remain locked indefinitely. To alleviate this, AetherDB adopts the Presumed Abort optimization combined with a decentralized termination protocol. Under Presumed Abort, if a participant queries the coordinator regarding an unknown or unlogged transaction, the coordinator safely assumes the transaction was aborted. Furthermore, participants execute an inquiry peer-protocol: if coordinator is unreachable, participants poll other cohort members; if any cohort has aborted or committed, the remaining cohort members adopt that outcome.

## Distributed Deadlock Detection via Wait-For Graphs
When concurrent transactions attempt to acquire conflicting row locks across different shards, circular dependencies can emerge, resulting in distributed deadlocks. AetherDB employs an asynchronous distributed Wait-For Graph (WFG) analyzer. Every shard periodically exports its local lock dependency edges `(T_holder -> T_waiter)` to a background deadlock detection worker. The analyzer aggregates edges into a global dependency digraph and executes Tarjan's cycle-finding algorithm. Upon detecting a directed cycle, the analyzer aborts the youngest transaction in the cycle using the `TXN_DEADLOCK_RESOLVER_KILL_YOUNGEST` heuristic, minimizing the cost of rolled-back computational work.

