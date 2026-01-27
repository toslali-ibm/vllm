# Prefix-Aware Routing Verification Guide

This guide shows you how to verify that prefix-aware routing is properly enabled and working when you start vLLM.

## Quick Verification Steps

### 1. Check Startup Logs

When you start vLLM with prefix-aware routing enabled, look for these log messages:

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --data-parallel-size 2 \
    --enable-prefix-aware-routing \
    --prefix-routing-length 16
```

**Expected logs at startup:**

```
INFO: Prefix-aware routing enabled with prefix_length=16
INFO: Initialized PrefixAwareDPLBAsyncMPClient with prefix_length=16, num_engines=2
```

These logs confirm:
- ✅ Prefix-aware routing is enabled
- ✅ The prefix length is configured correctly
- ✅ The correct client type is being used
- ✅ The number of engines is correct

### 2. Enable Debug Logging (Optional)

For detailed routing decisions, start vLLM with debug logging:

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --data-parallel-size 2 \
    --enable-prefix-aware-routing \
    --prefix-routing-length 16
```

**Expected debug logs during request handling:**

```
DEBUG: Prefix cache MISS: routing new prefix to engine 0 (prefix: (1, 2, 3, 4)..., load score: 0)
DEBUG: Prefix cache HIT: routing request abc123 to engine 0 (prefix: (1, 2, 3, 4)...)
```

These debug logs show:
- 🔍 Whether each request hit the cache or is a new prefix
- 🔍 Which engine was selected for each request
- 🔍 The first 4 tokens of the prefix
- 🔍 The load score when assigning new prefixes

### 3. Monitor Statistics

Every 100 requests, you'll see statistics:

```
INFO: Prefix routing stats: 100 requests, 75 cache hits (75.0%), 25 unique prefixes
INFO: Prefix routing stats: 200 requests, 160 cache hits (80.0%), 30 unique prefixes
```

This tells you:
- 📊 Total requests processed
- 📊 Number of cache hits
- 📊 Cache hit rate percentage
- 📊 Number of unique prefixes seen

### 4. Verify Configuration

You can verify the configuration is loaded correctly by checking:

```bash
# Start server with config display
vllm serve <model> \
    --data-parallel-size 2 \
    --enable-prefix-aware-routing \
    --prefix-routing-length 16 2>&1 | grep -i "prefix"
```

Expected output includes:
```
INFO: Prefix-aware routing enabled with prefix_length=16
INFO: Initialized PrefixAwareDPLBAsyncMPClient with prefix_length=16, num_engines=2
```

## Log Levels Explained

### INFO Level (Default)
- ✅ Feature enablement confirmation
- ✅ Client initialization
- ✅ Statistics every 100 requests

### DEBUG Level
- 🔍 Every routing decision (cache hit/miss)
- 🔍 Engine selection for each request
- 🔍 Prefix information
- 🔍 Load scores

## What to Look For

### ✅ Good Signs

1. **At startup:**
   - "Prefix-aware routing enabled with prefix_length=X"
   - "Initialized PrefixAwareDPLBAsyncMPClient"

2. **During operation:**
   - Periodic statistics showing cache hits
   - Debug logs showing routing decisions (if enabled)
   - Increasing cache hit rate over time

3. **Performance:**
   - Improved latency for repeated prefixes
   - Even load distribution across engines

### ❌ Warning Signs

1. **Missing logs:**
   - No "Prefix-aware routing enabled" message → Feature not enabled
   - No "PrefixAwareDPLBAsyncMPClient" message → Wrong client type

2. **Performance issues:**
   - Cache hit rate stays at 0% → Check if requests have varied prefixes
   - All requests go to one engine → Check load balancing

3. **Errors:**
   - "enable_prefix_aware_routing requires data_parallel_size > 1"
   - Configuration errors in startup logs

## Testing the Feature

### Quick Test with curl

```bash
# Start server
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --data-parallel-size 2 \
    --enable-prefix-aware-routing \
    --port 8000

# In another terminal, send requests with same prefix
for i in {1..10}; do
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "prompt": "Once upon a time in a magical kingdom, chapter '$i'",
            "max_tokens": 10
        }'
done
```

Watch the logs - you should see:
1. First request: Cache MISS
2. Subsequent requests with same prefix: Cache HIT
3. After 10 requests: Statistics showing cache hit rate

### Using the Demo Script

```bash
# Run the demo script
python examples/online_serving/prefix_routing_demo.py
```

This will:
- Send requests with repeated prefixes
- Show routing behavior
- Display cache hit statistics

## Troubleshooting

### Issue: No logs about prefix routing

**Solution:**
- Check that `--enable-prefix-aware-routing` flag is present
- Verify `--data-parallel-size` is > 1
- Check that you're not using external load balancing (`--data-parallel-rank` not set)

### Issue: Cache hit rate is 0%

**Possible causes:**
- All requests have unique prefixes (expected for varied workload)
- Prefix length too long (try reducing `--prefix-routing-length`)
- Requests are very short (< prefix length)

**Solution:**
- Use debug logs to see what prefixes are being extracted
- Try shorter prefix length (e.g., `--prefix-routing-length 8`)

### Issue: All requests go to one engine

**Possible causes:**
- All requests have the same prefix (expected)
- Load balancing not working for new prefixes

**Solution:**
- Check debug logs to see if new prefixes are being detected
- Verify engine load stats are updating correctly
- Send requests with varied prefixes to test load balancing

## Advanced: Monitoring with Prometheus

If you have Prometheus metrics enabled, you can monitor:

```bash
# Check request distribution across engines
curl http://localhost:8000/metrics | grep vllm_request_count
```

Expected: Requests should be distributed across engines based on prefix patterns.

## Example Log Output

Here's what a successful startup and operation looks like:

```
INFO 01-27 12:00:00 parallel.py:285] Prefix-aware routing enabled with prefix_length=16
INFO 01-27 12:00:01 core_client.py:1215] Initialized PrefixAwareDPLBAsyncMPClient with prefix_length=16, num_engines=2

# During operation with DEBUG enabled:
DEBUG 01-27 12:00:05 core_client.py:1245] Prefix cache MISS: routing new prefix to engine 0 (prefix: (1, 2, 3, 4)..., load score: 0)
DEBUG 01-27 12:00:06 core_client.py:1238] Prefix cache HIT: routing request req_001 to engine 0 (prefix: (1, 2, 3, 4)...)
DEBUG 01-27 12:00:07 core_client.py:1238] Prefix cache HIT: routing request req_002 to engine 0 (prefix: (1, 2, 3, 4)...)

# Statistics every 100 requests:
INFO 01-27 12:00:30 core_client.py:1257] Prefix routing stats: 100 requests, 82 cache hits (82.0%), 18 unique prefixes
```

## Summary

To verify prefix-aware routing is working:

1. ✅ Look for "Prefix-aware routing enabled" at startup
2. ✅ Look for "PrefixAwareDPLBAsyncMPClient" initialization
3. ✅ Check periodic statistics showing cache hits
4. ✅ (Optional) Enable DEBUG logs to see individual routing decisions
5. ✅ Run demo script to see feature in action

The feature is working correctly if you see:
- Initialization logs at startup
- Increasing cache hit rates for repeated prefixes
- Load balancing for new prefixes
- Statistics showing reasonable hit rates
