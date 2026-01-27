# Prefix-Aware Routing Testing Scripts

This directory contains scripts to test and benchmark the prefix-aware routing feature.

## Scripts Overview

### 1. `benchmark_prefix_routing.py` - Simple Benchmark (Recommended)

Quick and configurable benchmark script with command-line options.

**Basic Usage:**
```bash
# Start server
PYTHONPATH=/workspace/vllm python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B \
    --data-parallel-size 2 \
    --enable-prefix-aware-routing \
    --port 8000

# Run benchmark
python benchmark_prefix_routing.py
```

**Custom Parameters:**
```bash
# More requests, faster rate
python benchmark_prefix_routing.py --num-requests 200 --rps 4.0

# Different model and port
python benchmark_prefix_routing.py --model meta-llama/Llama-3.2-3B-Instruct --port 8001

# Adjust pattern
python benchmark_prefix_routing.py --num-engines 4 --prefix-length 1000
```

**Options:**
- `--num-requests`: Number of requests to send (default: 100)
- `--rps`: Requests per second (default: 2.0)
- `--num-engines`: Number of DP engines (default: 2)
- `--max-tokens`: Tokens to generate (default: 20)
- `--prefix-length`: Prefix length in chars (default: 500)
- `--port`: Server port (default: 8000)

### 2. `test_prefix_routing_workload.py` - Detailed Workload Test

More detailed test that closely mimics `workload_mert.py` pattern.

**Usage:**
```bash
python test_prefix_routing_workload.py
```

Provides detailed analysis including:
- Per-request latency tracking
- Latency distribution (P50, P90, P99)
- Cache hit rate estimation
- Speedup calculations

### 3. `prefix_routing_demo.py` - Interactive Demo

User-friendly demonstration with predefined scenarios.

**Usage:**
```bash
python prefix_routing_demo.py
```

Shows three scenarios:
1. Repeated prefixes (cache hits)
2. Load balancing new prefixes
3. Mixed workload

## Expected Output

### Successful Test Output

```
================================================================================
Prefix-Aware Routing Benchmark
================================================================================
Model: Qwen/Qwen2.5-0.5B
Requests: 100
Rate: 2.0 req/sec
Engines: 2
Pattern: Every 3 requests shares same prefix

Running benchmark...
[  1/100] NEW 0.234s
[  2/100] NEW 0.198s
[  3/100] DUP 0.145s  ← Lower latency (cache hit)
[  4/100] NEW 0.223s
...

================================================================================
Results
================================================================================

Unique Requests (Cache MISS):
  Count: 67
  Avg: 0.215s
  P50: 0.210s
  P90: 0.245s

Duplicate Requests (Cache HIT):
  Count: 33
  Avg: 0.142s  ← Lower than unique
  P50: 0.138s
  P90: 0.158s

Cache Benefit:
  Speedup: 1.51x
  Latency reduction: 33.8%
  ✅ Prefix caching working!
```

## Understanding Results

### Good Signs ✅

1. **Speedup > 1.1x**: Cache is working
2. **Duplicate latency < Unique latency**: Routing is effective
3. **Consistent pattern**: Cache hits show lower latency

### What to Check if Results Look Wrong

1. **Speedup ≈ 1.0x**:
   - Check if prefix-aware routing is enabled in server logs
   - Verify `--enable-prefix-aware-routing` flag
   - Check if `--data-parallel-size > 1`

2. **High variance in duplicate latencies**:
   - Normal for small sample sizes
   - Run more requests (`--num-requests 500`)

3. **Server errors**:
   - Check server is running: `curl http://localhost:8000/health`
   - Verify model name matches server
   - Check port number

## Server Logs to Monitor

When running these scripts, watch the server logs for:

```
INFO: Prefix-aware routing enabled with prefix_length=16
INFO: Initialized PrefixAwareDPLBAsyncMPClient with prefix_length=16, num_engines=2
```

During requests (with DEBUG logging):
```
DEBUG: Prefix cache MISS: routing new prefix to engine 0
DEBUG: Prefix cache HIT: routing request req_001 to engine 0
```

Every 100 requests:
```
INFO: Prefix routing stats: 100 requests, 75 cache hits (75.0%), 25 unique prefixes
```

## Tips for Best Results

1. **Use consistent prefix length**: Default 500 chars works well
2. **Reasonable RPS**: Start with 2-4 req/sec
3. **Enough requests**: At least 100 to see patterns
4. **Check server logs**: Verify cache hits are happening
5. **Multiple runs**: Run 2-3 times and average results

## Troubleshooting

### Connection Errors
```bash
# Check server is running
curl http://localhost:8000/health

# Check correct port
netstat -an | grep 8000
```

### No Cache Benefit
```bash
# Enable debug logging on server
VLLM_LOGGING_LEVEL=DEBUG python -m vllm.entrypoints.openai.api_server ...

# Check logs for "Prefix cache HIT/MISS" messages
```

### Import Errors
```bash
# Install OpenAI client
pip install openai

# Use PYTHONPATH if running from source
PYTHONPATH=/workspace/vllm python benchmark_prefix_routing.py
```

## Advanced Usage

### Compare With/Without Prefix Routing

```bash
# Terminal 1: With prefix routing
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B \
    --data-parallel-size 2 \
    --enable-prefix-aware-routing \
    --port 8000

# Terminal 2: Without prefix routing
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B \
    --data-parallel-size 2 \
    --port 8001

# Terminal 3: Compare
python benchmark_prefix_routing.py --port 8000 > with_routing.txt
python benchmark_prefix_routing.py --port 8001 > without_routing.txt
diff with_routing.txt without_routing.txt
```

### Vary Prefix Length

```bash
# Test different prefix lengths
for length in 100 500 1000 2000; do
    echo "Testing prefix length: $length"
    python benchmark_prefix_routing.py --prefix-length $length
done
```

### Load Testing

```bash
# High throughput test
python benchmark_prefix_routing.py \
    --num-requests 1000 \
    --rps 10.0 \
    --num-engines 4
```
