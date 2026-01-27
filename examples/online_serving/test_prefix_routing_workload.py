#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Minimal workload script to test prefix-aware routing with realistic patterns.

This script mimics the workload pattern from workload_mert.py:
- Sends requests with repeated prefix patterns
- Records end-to-end latencies
- Shows statistics comparing cache hits vs misses

Usage:
    # Start vLLM server first:
    PYTHONPATH=/workspace/vllm python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-0.5B \
        --data-parallel-size 2 \
        --enable-prefix-aware-routing \
        --port 8000

    # Run this script:
    python test_prefix_routing_workload.py
"""

import asyncio
import time
from dataclasses import dataclass
from typing import List

from openai import AsyncOpenAI

# Configuration
API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "Qwen/Qwen2.5-0.5B"

# Workload parameters
NUM_REQUESTS = 100
REQUESTS_PER_SECOND = 2.0
NUM_ENGINES = 2

# Prefix pattern: every (NUM_ENGINES + 1)-th request duplicates the previous one
# This tests prefix caching and routing stickiness


@dataclass
class RequestResult:
    """Store results for a single request."""
    request_id: int
    is_duplicate: bool
    latency: float
    prompt_prefix: str
    tokens_generated: int


def generate_shared_prefix(length: int = 500) -> str:
    """Generate a long shared prefix to simulate real workload."""
    return ("The quick brown fox jumps over the lazy dog. " * 50)[:length]


def create_prompt(request_id: int, num_engines: int = 2) -> tuple[str, bool]:
    """
    Create a prompt following the workload pattern.

    Returns:
        (prompt, is_duplicate): The prompt string and whether it's a duplicate
    """
    shared_prefix = generate_shared_prefix(500)

    # Every (num_engines + 1)-th request is a duplicate
    is_duplicate = (request_id % (num_engines + 1)) == 0

    if is_duplicate and request_id > 0:
        # Use the same counter as previous request
        counter = request_id - 1
    else:
        counter = request_id

    # Create prompt with shared prefix + unique suffix
    prompt = f"Request {counter}: {shared_prefix} Continue the story:"

    return prompt, is_duplicate


async def send_request(
    client: AsyncOpenAI,
    request_id: int,
    prompt: str,
    is_duplicate: bool,
) -> RequestResult:
    """Send a single request and measure latency."""
    start_time = time.time()

    try:
        response = await client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            max_tokens=20,
            temperature=0.0,  # Deterministic for testing
        )

        latency = time.time() - start_time
        tokens = response.usage.completion_tokens if response.usage else 0

        return RequestResult(
            request_id=request_id,
            is_duplicate=is_duplicate,
            latency=latency,
            prompt_prefix=prompt[:50] + "...",
            tokens_generated=tokens,
        )
    except Exception as e:
        print(f"Request {request_id} failed: {e}")
        return RequestResult(
            request_id=request_id,
            is_duplicate=is_duplicate,
            latency=-1,
            prompt_prefix=prompt[:50] + "...",
            tokens_generated=0,
        )


async def run_workload(num_requests: int, rps: float, num_engines: int):
    """
    Run the workload with timing control.

    Args:
        num_requests: Total number of requests to send
        rps: Requests per second rate
        num_engines: Number of data parallel engines
    """
    print("=" * 80)
    print("Prefix-Aware Routing Workload Test")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Requests: {num_requests}")
    print(f"  - Rate: {rps} req/sec")
    print(f"  - Engines: {num_engines}")
    print(f"  - Duplicate pattern: Every {num_engines + 1} requests")
    print(f"  - Model: {MODEL_NAME}")
    print(f"  - API: {API_BASE}")
    print()

    client = AsyncOpenAI(api_key="EMPTY", base_url=API_BASE)

    results: List[RequestResult] = []
    delta_t = 1.0 / rps

    print(f"Starting workload... (ETA: {num_requests * delta_t:.1f}s)")
    print()

    start_time = time.time()

    for i in range(num_requests):
        # Calculate target send time
        target_time = start_time + (i * delta_t)

        # Wait until target time
        sleep_time = target_time - time.time()
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

        # Generate prompt
        prompt, is_duplicate = create_prompt(i, num_engines)

        # Send request
        result = await send_request(client, i, prompt, is_duplicate)
        results.append(result)

        # Print progress
        status = "DUP " if is_duplicate else "NEW "
        if result.latency > 0:
            print(f"[{i+1:3d}/{num_requests}] {status} "
                  f"Latency: {result.latency:.3f}s | {result.prompt_prefix}")
        else:
            print(f"[{i+1:3d}/{num_requests}] {status} FAILED")

    total_time = time.time() - start_time

    # Analyze results
    print()
    print("=" * 80)
    print("Results Analysis")
    print("=" * 80)

    successful = [r for r in results if r.latency > 0]

    if not successful:
        print("No successful requests!")
        return

    # Separate by duplicate status
    duplicates = [r for r in successful if r.is_duplicate]
    uniques = [r for r in successful if not r.is_duplicate]

    # Calculate statistics
    total_requests = len(successful)
    dup_count = len(duplicates)
    unique_count = len(uniques)

    avg_latency_all = sum(r.latency for r in successful) / len(successful)

    if duplicates:
        avg_latency_dup = sum(r.latency for r in duplicates) / len(duplicates)
        min_latency_dup = min(r.latency for r in duplicates)
        max_latency_dup = max(r.latency for r in duplicates)
    else:
        avg_latency_dup = 0
        min_latency_dup = 0
        max_latency_dup = 0

    if uniques:
        avg_latency_unique = sum(r.latency for r in uniques) / len(uniques)
        min_latency_unique = min(r.latency for r in uniques)
        max_latency_unique = max(r.latency for r in uniques)
    else:
        avg_latency_unique = 0
        min_latency_unique = 0
        max_latency_unique = 0

    # Print statistics
    print(f"\nOverall Statistics:")
    print(f"  - Total requests: {total_requests}")
    print(f"  - Successful: {len(successful)}")
    print(f"  - Failed: {len(results) - len(successful)}")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"  - Actual RPS: {len(successful) / total_time:.2f}")
    print(f"  - Average latency: {avg_latency_all:.3f}s")

    print(f"\nUnique Requests (Cache MISS expected):")
    print(f"  - Count: {unique_count}")
    print(f"  - Avg latency: {avg_latency_unique:.3f}s")
    print(f"  - Min latency: {min_latency_unique:.3f}s")
    print(f"  - Max latency: {max_latency_unique:.3f}s")

    print(f"\nDuplicate Requests (Cache HIT expected):")
    print(f"  - Count: {dup_count}")
    print(f"  - Avg latency: {avg_latency_dup:.3f}s")
    print(f"  - Min latency: {min_latency_dup:.3f}s")
    print(f"  - Max latency: {max_latency_dup:.3f}s")

    if duplicates and uniques:
        speedup = avg_latency_unique / avg_latency_dup
        print(f"\nPrefix Cache Benefit:")
        print(f"  - Speedup: {speedup:.2f}x")
        print(f"  - Latency reduction: {(1 - 1/speedup) * 100:.1f}%")

        if speedup > 1.1:
            print(f"  ✅ Prefix caching is working! (speedup > 1.1x)")
        elif speedup > 1.0:
            print(f"  ⚠️  Minimal benefit (speedup {speedup:.2f}x)")
        else:
            print(f"  ❌ No cache benefit detected (speedup {speedup:.2f}x)")

    # Show latency distribution
    print(f"\nLatency Distribution (all requests):")
    latencies = sorted([r.latency for r in successful])
    if latencies:
        p50 = latencies[len(latencies) // 2]
        p90 = latencies[int(len(latencies) * 0.9)]
        p99 = latencies[int(len(latencies) * 0.99)]
        print(f"  - P50: {p50:.3f}s")
        print(f"  - P90: {p90:.3f}s")
        print(f"  - P99: {p99:.3f}s")

    print()
    print("=" * 80)


async def main():
    """Run the workload test."""
    try:
        await run_workload(
            num_requests=NUM_REQUESTS,
            rps=REQUESTS_PER_SECOND,
            num_engines=NUM_ENGINES,
        )

        print("\n💡 Tips:")
        print("  - Check server logs for 'Prefix cache HIT/MISS' messages")
        print("  - Every 100 requests shows cache hit statistics")
        print("  - Enable DEBUG logging: VLLM_LOGGING_LEVEL=DEBUG")
        print("  - Duplicate requests should show lower latency (cache hits)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the vLLM server is running:")
        print("  PYTHONPATH=/workspace/vllm python -m vllm.entrypoints.openai.api_server \\")
        print("      --model Qwen/Qwen2.5-0.5B \\")
        print("      --data-parallel-size 2 \\")
        print("      --enable-prefix-aware-routing \\")
        print("      --port 8000")


if __name__ == "__main__":
    asyncio.run(main())
