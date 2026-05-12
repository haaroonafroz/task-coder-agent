#!/usr/bin/env python3
"""
Standalone benchmark: Baseline vs Speculative Decoding (MTP)
Tests single long-form generation (3-4k tokens) outside agentic pipeline.
"""

import os
import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple

# Add pipeline to path to reuse the existing client
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from llm_client import call_llm, ENDPOINTS, TARGET_MODEL


@dataclass
class BenchmarkResult:
    mode: str
    tokens_generated: int
    prefill_ms: float
    decode_ms: float
    total_ms: float
    tokens_per_second: float
    text_preview: str


def get_long_prompt() -> str:
    """
    A prompt designed to elicit a very long (3-4k token) response.
    Technical deep-dive format maximizes token generation.
    """
    return """You are a senior software architect and technical educator. Write a comprehensive, in-depth guide on "Advanced Design Patterns in Distributed Systems" for senior engineers.

Your response must include:

1. **Introduction to Distributed Systems Fundamentals** (2-3 paragraphs)
   - CAP theorem deep dive
   - Consistency models (linearizable, sequential, causal, eventual)
   - Partition tolerance and network failures

2. **Core Design Patterns** - For EACH pattern provide:
   - Detailed problem statement
   - Solution architecture with component diagrams (describe textually)
   - Implementation considerations
   - Trade-offs and when to use
   - Code example in Python or pseudocode

   Patterns to cover:
   - Circuit Breaker
   - Bulkhead
   - Retry with Exponential Backoff and Jitter
   - Saga Pattern (choreography vs orchestration)
   - CQRS (Command Query Responsibility Segregation)
   - Event Sourcing
   - Sharding and Data Partitioning
   - Leader Election and Consensus

3. **Anti-Patterns and Pitfalls** (2-3 paragraphs each)
   - Distributed monoliths
   - Shared database fallacy
   - Synchronous communication chains
   - Ignoring backpressure

4. **Real-World Case Studies** - Describe 2 scenarios:
   - High-frequency trading system
   - Global video streaming platform

5. **Monitoring and Observability** (2-3 paragraphs)
   - Distributed tracing
   - Metrics aggregation
   - Alerting strategies

Be extremely detailed and thorough. This is educational content for experienced engineers. Minimum 3000 words. Include specific implementation details, configuration parameters, and architectural reasoning throughout.

Begin:"""


def benchmark_mode(mode: str, max_tokens: int = 4000) -> Optional[BenchmarkResult]:
    """
    Run a single benchmark for the specified mode.
    
    Args:
        mode: "baseline" or "speculative"
        max_tokens: Maximum tokens to generate
    
    Returns:
        BenchmarkResult or None if error
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking: {mode.upper()}")
    print(f"{'='*70}")
    print(f"Endpoint: {ENDPOINTS[mode]}")
    print(f"Model: {TARGET_MODEL}")
    print(f"Max tokens: {max_tokens}")
    print("-" * 70)
    
    prompt = get_long_prompt()
    print(f"Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")
    print("Generating long-form response...")
    
    try:
        result = call_llm(
            prompt=prompt,
            mode=mode,
            max_tokens=max_tokens,
        )
        
        tps = result.tokens_generated / (result.total_ms / 1000.0)
        
        benchmark = BenchmarkResult(
            mode=mode,
            tokens_generated=result.tokens_generated,
            prefill_ms=result.prefill_ms,
            decode_ms=result.decode_ms,
            total_ms=result.total_ms,
            tokens_per_second=tps,
            text_preview=result.text[:200].replace('\n', ' ') + "..."
        )
        
        print(f"\n✅ Generation complete!")
        print(f"   Tokens generated: {benchmark.tokens_generated}")
        print(f"   Prefill time: {benchmark.prefill_ms:.2f} ms")
        print(f"   Decode time: {benchmark.decode_ms:.2f} ms")
        print(f"   Total time: {benchmark.total_ms:.2f} ms")
        print(f"   Tokens/sec: {benchmark.tokens_per_second:.2f}")
        print(f"\n   Preview: {benchmark.text_preview}")
        
        return benchmark
        
    except Exception as e:
        print(f"\n❌ Error in {mode} mode: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_comparison(num_runs: int = 3, max_tokens: int = 4000) -> Tuple[List[BenchmarkResult], List[BenchmarkResult]]:
    """
    Run multiple benchmarks for both modes and collect results.
    
    Args:
        num_runs: Number of runs per mode (for averaging)
        max_tokens: Max tokens per generation
    
    Returns:
        Tuple of (baseline_results, speculative_results)
    """
    baseline_results: List[BenchmarkResult] = []
    speculative_results: List[BenchmarkResult] = []
    
    print("\n" + "="*70)
    print("SPECULATIVE DECODING BENCHMARK")
    print("Single Long-Form Generation (3-4K tokens)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Runs per mode: {num_runs}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Target model: {TARGET_MODEL}")
    
    # Run baseline first
    print("\n" + "="*70)
    print("PHASE 1: BASELINE (Autoregressive)")
    print("="*70)
    
    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")
        result = benchmark_mode("baseline", max_tokens)
        if result:
            baseline_results.append(result)
        # Small delay between runs
        time.sleep(1)
    
    # Run speculative
    print("\n" + "="*70)
    print("PHASE 2: SPECULATIVE (MTP)")
    print("="*70)
    
    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")
        result = benchmark_mode("speculative", max_tokens)
        if result:
            speculative_results.append(result)
        # Small delay between runs
        time.sleep(1)
    
    return baseline_results, speculative_results


def analyze_results(baseline_results: List[BenchmarkResult], speculative_results: List[BenchmarkResult]):
    """
    Print comparative analysis of benchmark results.
    """
    print("\n" + "="*70)
    print("BENCHMARK ANALYSIS")
    print("="*70)
    
    if not baseline_results or not speculative_results:
        print("\n❌ Insufficient data for analysis")
        return
    
    # Calculate averages
    def avg(values: List[float]) -> float:
        return sum(values) / len(values)
    
    # Baseline stats
    base_tokens = avg([r.tokens_generated for r in baseline_results])
    base_prefill = avg([r.prefill_ms for r in baseline_results])
    base_decode = avg([r.decode_ms for r in baseline_results])
    base_total = avg([r.total_ms for r in baseline_results])
    base_tps = avg([r.tokens_per_second for r in baseline_results])
    
    # Speculative stats
    spec_tokens = avg([r.tokens_generated for r in speculative_results])
    spec_prefill = avg([r.prefill_ms for r in speculative_results])
    spec_decode = avg([r.decode_ms for r in speculative_results])
    spec_total = avg([r.total_ms for r in speculative_results])
    spec_tps = avg([r.tokens_per_second for r in speculative_results])
    
    # Speedup calculations
    prefill_speedup = base_prefill / spec_prefill if spec_prefill > 0 else 0
    decode_speedup = base_decode / spec_decode if spec_decode > 0 else 0
    total_speedup = base_total / spec_total if spec_total > 0 else 0
    tps_ratio = spec_tps / base_tps if base_tps > 0 else 0
    
    print(f"\n{'Metric':<25} {'Baseline':>15} {'Speculative':>15} {'Speedup':>12}")
    print("-" * 70)
    print(f"{'Avg Tokens Generated':<25} {base_tokens:>15.1f} {spec_tokens:>15.1f} {'--':>12}")
    print(f"{'Avg Prefill (ms)':<25} {base_prefill:>15.2f} {spec_prefill:>15.2f} {prefill_speedup:>11.2f}x")
    print(f"{'Avg Decode (ms)':<25} {base_decode:>15.2f} {spec_decode:>15.2f} {decode_speedup:>11.2f}x")
    print(f"{'Avg Total (ms)':<25} {base_total:>15.2f} {spec_total:>15.2f} {total_speedup:>11.2f}x")
    print(f"{'Tokens/Second':<25} {base_tps:>15.2f} {spec_tps:>15.2f} {tps_ratio:>11.2f}x")
    
    print("\n" + "-" * 70)
    if total_speedup > 1.0:
        print(f"✅ SPECULATIVE IS {total_speedup:.2f}x FASTER")
    elif total_speedup < 1.0:
        print(f"⚠️  SPECULATIVE IS {1/total_speedup:.2f}x SLOWER")
    else:
        print(f"➡️  PERFORMANCE IS EQUIVALENT")
    
    # Raw data export
    print("\n" + "="*70)
    print("RAW RESULTS (JSON)")
    print("="*70)
    data = {
        "baseline": [
            {
                "tokens": r.tokens_generated,
                "prefill_ms": r.prefill_ms,
                "decode_ms": r.decode_ms,
                "total_ms": r.total_ms,
                "tps": r.tokens_per_second
            }
            for r in baseline_results
        ],
        "speculative": [
            {
                "tokens": r.tokens_generated,
                "prefill_ms": r.prefill_ms,
                "decode_ms": r.decode_ms,
                "total_ms": r.total_ms,
                "tps": r.tokens_per_second
            }
            for r in speculative_results
        ],
        "summary": {
            "baseline_avg_tps": base_tps,
            "speculative_avg_tps": spec_tps,
            "speedup": total_speedup,
            "decode_speedup": decode_speedup
        }
    }
    print(json.dumps(data, indent=2))


def test_simple():
    """Quick smoke test."""
    print("\n" + "="*70)
    print("SMOKE TEST: Simple completion")
    print("="*70)
    
    for mode in ["baseline", "speculative"]:
        print(f"\nTesting {mode}...")
        try:
            result = call_llm(
                prompt="Write a 500-word essay about the importance of testing in software engineering.",
                mode=mode,
                max_tokens=800
            )
            tps = result.tokens_generated / (result.total_ms / 1000.0)
            print(f"  ✅ {mode}: {result.tokens_generated} tokens @ {tps:.2f} tps")
        except Exception as e:
            print(f"  ❌ {mode}: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark speculative vs baseline decoding")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per mode (default: 3)")
    parser.add_argument("--max-tokens", type=int, default=4000, help="Max tokens to generate (default: 4000)")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke test only")
    args = parser.parse_args()
    
    if args.quick:
        test_simple()
    else:
        # Full benchmark
        baseline_results, speculative_results = run_comparison(
            num_runs=args.runs,
            max_tokens=args.max_tokens
        )
        analyze_results(baseline_results, speculative_results)