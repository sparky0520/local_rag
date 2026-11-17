# Globally Distributed Linearizability and TrueTime Synchronization

## The TrueTime API and Uncertainty Bounds
In globally distributed multi-region deployments, establishing linearizable serializability without a centralized synchronization bottleneck requires high-precision timekeeping. AetherDB integrates an atomic clock and GPS time synchronization API modeled on Google Spanner's TrueTime architecture. The `TrueTime.now()` primitive does not return a single discrete timestamp; rather, it yields an interval of bounded uncertainty:
$$TT.now() = [t_{\text{earliest}}, t_{\text{latest}}] = [t - \epsilon, t + \epsilon]$$
where $t$ represents absolute physical UTC time and $\epsilon$ represents the dynamic clock drift uncertainty bound, typically maintained below $\epsilon \le 4\text{ ms}$ under normal network conditions using synchronized GPS receivers and atomic rubidium oscillators in each datacenter.

## The Commit Wait Rule for External Consistency
The foundational requirement of linearizability (external consistency) is that if transaction $T_2$ begins after transaction $T_1$ commits in real absolute time, then $T_2$'s commit timestamp must be strictly greater than $T_1$'s commit timestamp:
$$T_1 \text{ commits before } T_2 \text{ starts} \implies t_{\text{commit}}(T_1) < t_{\text{commit}}(T_2)$$
To enforce this invariant without cross-region locks or coordinator handshakes, AetherDB applies the **Commit Wait Rule**:
1. When transaction $T_1$ prepares to commit, the coordinator picks a commit timestamp $s \ge TT.now().\text{latest}$.
2. The coordinator delays releasing locks and exposing $T_1$'s writes to external clients until it is guaranteed that absolute physical time has passed $s$, which is satisfied once $TT.now().\text{earliest} > s$.
3. This mandatory delay (equal to $2\epsilon$) ensures that any subsequent transaction $T_2$ initiated anywhere on Earth is guaranteed to observe $TT.now().\text{earliest} > s$, thereby preserving linearizability.

## TrueTime Anomaly Handling: Asymmetric Clock Drift
If a datacenter experiences severe network jitter or GPS antenna failure, local oscillator drift increases $\epsilon$. If $\epsilon$ exceeds `MAX_TRUETIME_DRIFT_THRESHOLD_MS = 100`, the affected storage node enters an emergency read-only isolation mode `ERR_CLOCK_UNCERTAINTY_EXCEEDED` to prevent publishing corrupted or non-linearizable transaction commit timestamps.

