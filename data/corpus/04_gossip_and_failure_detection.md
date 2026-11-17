# Cluster Membership Gossip and Phi Accrual Failure Detection

## SWIM-Based Gossip Protocol Architecture
Cluster membership and node topology in AetherDB are maintained using a decentralized gossip protocol inspired by SWIM (Structured Weakly-Consistent Infection-Style Process Group Membership). Every node periodically selects a random peer from its local member directory and transmits a direct `Ping` probe over UDP every `GOSSIP_PING_PERIOD_MS = 200` milliseconds. If the probed peer replies with `Ack` within `GOSSIP_PING_TIMEOUT_MS = 100` milliseconds, the node is confirmed healthy.

## Indirect Probing via Ping-Req
If a target node does not respond within the direct timeout window, the initiating node does not immediately declare the target dead, as transient packet drops or local network congestion can cause spurious timeouts. Instead, the initiator selects $k=3$ random auxiliary nodes and sends them a `Ping-Req(target)` message. Each auxiliary node attempts to ping the target directly. If any of the auxiliary nodes receives an `Ack` from the target, it relays the confirmation back to the initiator. This indirect probing mechanism reduces false-positive node failure declarations by more than 98% in multi-tenant cloud environments.

## Suspicion Mechanism and State Dissemination
If indirect probing fails to elicit any response within `GOSSIP_SUSPECT_TIMEOUT_MS = 1500` milliseconds, the initiator marks the target as `SUSPECT` rather than immediately dead. The suspicion status is disseminated to all cluster nodes piggybacked onto regular gossip heartbeats. During the suspicion period, the suspected node can refute its own demise by broadcasting a higher incarnation number (`incarnation_seq + 1`). If the suspicion timer expires without a valid rebuttal, the node is officially classified as `FAULTY` and removed from the active routing topology.

## Phi Accrual Failure Detector Algorithm
For high-performance inter-datacenter links, AetherDB supplements the gossip protocol with the Hayashibara Phi Accrual Failure Detector. Rather than rendering a binary alive/dead verdict, the detector continuously tracks the historical distribution of inter-arrival heartbeat intervals. By fitting an exponential or normal probability distribution over the recent sliding window of $N=1000$ heartbeat intervals, the detector computes a dynamic continuous metric $\Phi$:
$$\Phi = -\log_{10}(P_{\text{later}}(t - t_{\text{last}}))$$
where $P_{\text{later}}(t)$ represents the probability that a heartbeat will arrive more than $t$ time units after the previous heartbeat. The application triggers alert or failover actions when $\Phi$ crosses a configurable threshold: `phi_accrual_threshold = 8.0` for conservative failover or `phi_accrual_threshold = 12.0` for flaky WAN links.

