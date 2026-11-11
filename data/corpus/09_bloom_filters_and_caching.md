# Probabilistic Bloom Filters and Multi-Tiered Cache Architecture

## Bloom Filter Mathematics and False Positive Rates
Because LSM storage engines distribute keys across multiple SSTable levels on disk, querying a nonexistent key can trigger multiple disk seeks across every SSTable. To circumvent this read amplification penalty, AetherDB equips each SSTable with an in-memory Bloom filter. A Bloom filter is a bit array of size $m$ with $k$ independent uniform hash functions. Given $n$ keys to be inserted into an SSTable, the optimal number of hash functions minimizing the false positive probability $p$ is derived as:
$$k = \frac{m}{n} \ln 2 \approx 0.693 \frac{m}{n}$$
The required bit array size $m$ for an intended false positive rate $p$ is calculated using:
$$m = -\frac{n \ln p}{(\ln 2)^2}$$
In AetherDB, the default configuration sets `bloom_filter_bits_per_key = 10`, which yields $k=7$ hash functions and a false positive probability $p \approx 1\%$. If a Bloom filter query returns negative, the key is definitively absent from the SSTable, allowing the query planner to bypass expensive disk I/O with 100% confidence.

## Block-Based Filter Layout
To avoid allocating large contiguous memory buffers for multi-gigabyte SSTables, AetherDB uses Block-Based Bloom Filters. The bit array is partitioned into discrete 2 KB filter blocks aligned with the underlying filesystem block boundaries. When a key is searched, only the corresponding 2 KB filter block is retrieved into the CPU L1/L2 cache, maximizing cache locality and eliminating branch misprediction overheads during high-frequency lookups.

## Multi-Tier Cache: Block Cache vs Row Cache
To accelerate data access paths, AetherDB implements a two-tier caching hierarchy:
1. **Block Cache (LRU/2Q)**: Caches uncompressed 4 KB SSTable data blocks in memory. When multiple keys located in the same SSTable block are accessed in temporal proximity, only the initial read incurs disk I/O. AetherDB employs the 2Q eviction algorithm to prevent one-off sequential range scans from polluting the active working set cache.
2. **Row Cache (ARC - Adaptive Replacement Cache)**: Caches fully deserialized, in-memory representations of individual hot key-value rows. Lookups hitting the Row Cache bypass SSTable block parsing, index traversal, and decompression entirely, achieving sub-microsecond query latencies.
