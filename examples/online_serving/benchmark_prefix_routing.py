#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark script for prefix-aware routing with configurable parameters.

Usage:
    # Basic usage
    python benchmark_prefix_routing.py

    # Custom parameters
    python benchmark_prefix_routing.py --num-requests 200 --rps 4.0 --num-engines 2

    # Different model
    python benchmark_prefix_routing.py --model meta-llama/Llama-3.2-3B-Instruct --port 8001
"""

import argparse
import asyncio
import random
import time
from dataclasses import dataclass
from typing import List

from openai import AsyncOpenAI


@dataclass
class LatencyStats:
    """Latency statistics."""
    count: int
    avg: float
    min: float
    max: float
    p50: float
    p90: float
    p99: float


def calculate_stats(latencies: List[float]) -> LatencyStats:
    """Calculate statistics from latency list."""
    if not latencies:
        return LatencyStats(0, 0, 0, 0, 0, 0, 0)

    sorted_lat = sorted(latencies)
    return LatencyStats(
        count=len(latencies),
        avg=sum(latencies) / len(latencies),
        min=sorted_lat[0],
        max=sorted_lat[-1],
        p50=sorted_lat[len(sorted_lat) // 2],
        p90=sorted_lat[int(len(sorted_lat) * 0.9)],
        p99=sorted_lat[int(len(sorted_lat) * 0.99)],
    )


async def send_request(client: AsyncOpenAI, model: str, prompt: str,
                       max_tokens: int) -> float:
    """Send request and return latency in seconds."""
    start = time.time()
    try:
        await client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return time.time() - start
    except Exception as e:
        print(f"Request failed: {e}")
        return -1

def random_word(min_len=2, max_len=3):
    import string
    letters = string.ascii_lowercase
    return "".join(random.choices(letters, k=random.randint(min_len, max_len)))

def random_words(n):
    return " ".join(random_word() for _ in range(n))

async def run_benchmark(args):
    """Run the benchmark."""
    print("=" * 80)
    print("Prefix-Aware Routing Benchmark")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"API: {args.api_base}")
    print(f"Requests: {args.num_requests}")
    print(f"Rate: {args.rps} req/sec")
    print(f"Engines: {args.num_engines}")
    print(
        f"Pattern: Every {args.num_engines + 1} requests shares same prefix")
    print(f"Prefix length: ~{args.prefix_length} chars")
    print()

    client = AsyncOpenAI(api_key="EMPTY", base_url=args.api_base)

    # Generate base prefix
    shared_body = random_word() + " "
    # base_prefix = ("The quick brown fox jumps over the lazy dog. " *
    #                20)[:args.prefix_length]

    latencies_unique = []
    latencies_duplicate = []
    delta_t = 1.0 / args.rps

    print("Running benchmark...")
    start_time = time.time()

    i = 1
    while i < args.num_requests:
        
        # Target send time for rate limiting
        target_time = start_time + (i * delta_t)
        await asyncio.sleep(max(0, target_time - time.time()))

        # Determine if this is a duplicate
        is_dup = (i % (args.num_engines + 1)) == 0 and i > 0

        # Create prompt
        if is_dup:
            counter = i - random.choice([1, 2])  # Same as previous
        else:
            counter = i
        
        base_prefix = shared_body * args.prefix_length

        prompt = f"Request {counter} {base_prefix} {random_words(random.randint(40, 100))}"

        print(prompt[:300])
        print(len(prompt.split()))
        i = i + 1

        # Send request
        latency = await send_request(client, args.model, prompt,
                                     args.max_tokens)

        if latency > 0:
            if is_dup:
                latencies_duplicate.append(latency)
                marker = "DUP"
            else:
                latencies_unique.append(latency)
                marker = "NEW"

            print(f"[{i+1:3d}/{args.num_requests}] {marker} {latency:.3f}s")

    total_time = time.time() - start_time

    # Calculate statistics
    print()
    print("=" * 80)
    print("Results")
    print("=" * 80)

    stats_unique = calculate_stats(latencies_unique)
    stats_dup = calculate_stats(latencies_duplicate)

    total_success = stats_unique.count + stats_dup.count

    # Calculate overall average latency
    all_latencies = latencies_unique + latencies_duplicate
    overall_avg_latency = (sum(all_latencies) /
                           len(all_latencies)) if all_latencies else 0

    print(f"\nTotal:")
    print(f"  Requests: {args.num_requests}")
    print(f"  Successful: {total_success}")
    print(f"  Failed: {args.num_requests - total_success}")
    print(f"  Time: {total_time:.2f}s")
    print(f"  Actual RPS: {total_success / total_time:.2f}")
    print(f"  Overall Avg Latency: {overall_avg_latency:.3f}s")

    if stats_unique.count > 0:
        print(f"\nUnique Requests (Cache MISS):")
        print(f"  Count: {stats_unique.count}")
        print(f"  Avg: {stats_unique.avg:.3f}s")
        print(f"  Min: {stats_unique.min:.3f}s")
        print(f"  Max: {stats_unique.max:.3f}s")
        print(f"  P50: {stats_unique.p50:.3f}s")
        print(f"  P90: {stats_unique.p90:.3f}s")

    if stats_dup.count > 0:
        print(f"\nDuplicate Requests (Cache HIT):")
        print(f"  Count: {stats_dup.count}")
        print(f"  Avg: {stats_dup.avg:.3f}s")
        print(f"  Min: {stats_dup.min:.3f}s")
        print(f"  Max: {stats_dup.max:.3f}s")
        print(f"  P50: {stats_dup.p50:.3f}s")
        print(f"  P90: {stats_dup.p90:.3f}s")

    if stats_dup.count > 0 and stats_unique.count > 0:
        speedup = stats_unique.avg / stats_dup.avg
        reduction = (1 - 1 / speedup) * 100

        print(f"\nCache Benefit:")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Latency reduction: {reduction:.1f}%")

        if speedup > 1.1:
            print(f"  ✅ Prefix caching working!")
        else:
            print(f"  ⚠️  Limited cache benefit")

    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark prefix-aware routing")
    parser.add_argument("--model",
                        default="Qwen/Qwen2.5-0.5B",
                        help="Model name")
    parser.add_argument("--api-base",
                        default="http://localhost:8000/v1",
                        help="API base URL")
    parser.add_argument("--port",
                        type=int,
                        help="Port (overrides api-base)")
    parser.add_argument("--num-requests",
                        type=int,
                        default=100,
                        help="Number of requests")
    parser.add_argument("--rps",
                        type=float,
                        default=1.0,
                        help="Requests per second")
    parser.add_argument("--num-engines",
                        type=int,
                        default=2,
                        help="Number of data parallel engines")
    parser.add_argument("--max-tokens",
                        type=int,
                        default=2,
                        help="Max tokens to generate")
    parser.add_argument("--prefix-length",
                        type=int,
                        default=2000,
                        help="Approximate prefix length in words")

    args = parser.parse_args()

    # Override api_base if port is specified
    if args.port:
        args.api_base = f"http://localhost:{args.port}/v1"

    try:
        asyncio.run(run_benchmark(args))
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure vLLM server is running:")
        print(
            f"  python -m vllm.entrypoints.openai.api_server --model {args.model} \\")
        print("      --data-parallel-size 2 --enable-prefix-aware-routing")


if __name__ == "__main__":
    main()
