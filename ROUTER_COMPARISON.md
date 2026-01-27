# Router Comparison Guide

This document shows how to use and compare the three routing strategies available in vLLM.

## Three Routing Strategies

### 1. **Default Router** (Load-Balanced)
**Class:** `DPLBAsyncMPClient`
**Algorithm:** Least-loaded engine selection
**Formula:** `score = waiting × 4 + running` (pick minimum)

```bash
# Default - no flags needed
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B \
    --data-parallel-size 2
```

**Characteristics:**
- ✅ Fair load distribution
- ✅ Adapts to engine load dynamically
- ❌ No request affinity
- ❌ No cache awareness

---

### 2. **Random Router** (Baseline)
**Class:** `RandomDPLBAsyncMPClient`
**Algorithm:** Random engine selection
**Formula:** `eng_index = random.randint(0, num_engines - 1)`

```bash
# Random routing
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B \
    --data-parallel-size 2 \
    --enable-random-routing
```

**Characteristics:**
- ✅ Simple baseline for comparison
- ✅ No overhead beyond randomness
- ⚠️ May cause load imbalance
- ❌ No cache awareness
- ❌ No load consideration

---

### 3. **Prefix-Aware Router** (Smart Caching)
**Class:** `PrefixAwareDPLBAsyncMPClient`
**Algorithm:** Prefix-based affinity + load balancing
**Logic:**
```python
if prefix in cache:
    route to cached engine  # Cache hit!
else:
    route to least-loaded   # Fall back to default
```

```bash
# Prefix-aware routing
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B \
    --data-parallel-size 2 \
    --enable-prefix-aware-routing \
    --prefix-routing-length 16
```

**Characteristics:**
- ✅ Cache-aware routing
- ✅ Request affinity for same prefixes
- ✅ Falls back to load balancing for new prefixes
- ⚠️ Memory grows with unique prefixes
- ⚠️ Best for pattern-heavy workloads

---

## Quick Comparison

| Feature | Default | Random | Prefix-Aware |
|---------|---------|--------|--------------|
| **Decision basis** | Engine load | None (random) | Prefix history + load |
| **Cache aware** | ❌ | ❌ | ✅ |
| **Load balancing** | ✅ Best | ⚠️ May imbalance | ✅ For new prefixes |
| **Overhead** | Low | Minimal | Low + O(1) lookup |
| **Memory** | Constant | Constant | Grows with prefixes |
| **Best for** | Varied workloads | Baseline testing | Repeated patterns |

---

## Benchmarking All Three

### Setup

Start three server instances on different ports:

```bash
# Terminal 1: Default (load-balanced)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B \
    --data-parallel-size 2 \
    --port 8000

# Terminal 2: Random
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B \
    --data-parallel-size 2 \
    --enable-random-routing \
    --port 8001

# Terminal 3: Prefix-aware
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B \
    --data-parallel-size 2 \
    --enable-prefix-aware-routing \
    --port 8002
```

### Run Benchmarks

```bash
# Terminal 4: Compare all three
echo "=== Default Router ==="
python examples/online_serving/benchmark_prefix_routing.py \
    --port 8000 --num-requests 100 --rps 2.0 > results_default.txt

echo "=== Random Router ==="
python examples/online_serving/benchmark_prefix_routing.py \
    --port 8001 --num-requests 100 --rps 2.0 > results_random.txt

echo "=== Prefix-Aware Router ==="
python examples/online_serving/benchmark_prefix_routing.py \
    --port 8002 --num-requests 100 --rps 2.0 > results_prefix.txt

# Compare results
echo "Default:" && grep "Overall Avg Latency" results_default.txt
echo "Random:" && grep "Overall Avg Latency" results_random.txt
echo "Prefix-Aware:" && grep "Overall Avg Latency" results_prefix.txt
```

### Expected Results

For workloads with **repeated prefixes**:

```
Default:       Overall Avg Latency: 0.215s
Random:        Overall Avg Latency: 0.230s  (worst - no cache, imbalanced)
Prefix-Aware:  Overall Avg Latency: 0.180s  (best - cache hits)
```

For workloads with **all unique prompts**:

```
Default:       Overall Avg Latency: 0.220s  (best - good balancing)
Random:        Overall Avg Latency: 0.235s  (worst - imbalanced)
Prefix-Aware:  Overall Avg Latency: 0.222s  (similar to default)
```

---

## Priority Order

If multiple flags are enabled, the priority is:

1. **Prefix-aware** (highest) - `--enable-prefix-aware-routing`
2. **Random** - `--enable-random-routing`
3. **Default** (lowest) - automatic if DP size > 1

Example:
```bash
# This will use prefix-aware routing (highest priority)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B \
    --data-parallel-size 2 \
    --enable-prefix-aware-routing \
    --enable-random-routing
```

---

## Verification Logs

### Default Router
```
INFO: Initialized DPLBAsyncMPClient with num_engines=2
```

### Random Router
```
INFO: Initialized RandomDPLBAsyncMPClient with num_engines=2
```

### Prefix-Aware Router
```
INFO: Prefix-aware routing enabled with prefix_length=16
INFO: Initialized PrefixAwareDPLBAsyncMPClient with prefix_length=16, num_engines=2
```

With `VLLM_LOGGING_LEVEL=DEBUG`:
```
# Random
DEBUG: Random routing: request req_001 to engine 1

# Prefix-Aware
DEBUG: Prefix cache MISS: routing new prefix to engine 0
DEBUG: Prefix cache HIT: routing request req_002 to engine 0
```

---

## When to Use Each Router

### Use Default (Load-Balanced)
- General purpose serving
- Highly varied prompts
- Unknown workload patterns
- Want proven stable performance

### Use Random (Baseline)
- Performance testing and comparison
- Research and benchmarking
- Understanding impact of smart routing
- **Not recommended for production**

### Use Prefix-Aware
- Chat applications with system prompts
- RAG systems with shared context
- Template-based generation
- Few-shot learning prompts
- Any workload with repeated prefix patterns

---

## Implementation Details

**File:** `vllm/v1/engine/core_client.py`

**Classes:**
- `DPLBAsyncMPClient` (lines 1108-1192) - Default
- `RandomDPLBAsyncMPClient` (lines 1429-1472) - Random
- `PrefixAwareDPLBAsyncMPClient` (lines 1197-1426) - Prefix-aware

**Selection Logic:** `make_async_mp_client()` (lines 85-105)

```python
if parallel_config.data_parallel_size > 1:
    if parallel_config.data_parallel_external_lb:
        return DPAsyncMPClient(*client_args)
    # Choose routing strategy
    if parallel_config.enable_prefix_aware_routing:
        return PrefixAwareDPLBAsyncMPClient(*client_args)
    if parallel_config.enable_random_routing:
        return RandomDPLBAsyncMPClient(*client_args)
    return DPLBAsyncMPClient(*client_args)  # Default
```

---

## Summary

Three routing strategies now available:

1. **Default** - Smart load balancing (no flags)
2. **Random** - Baseline for testing (`--enable-random-routing`)
3. **Prefix-Aware** - Cache-optimized (`--enable-prefix-aware-routing`)

Use the benchmark script to compare them on your specific workload!
