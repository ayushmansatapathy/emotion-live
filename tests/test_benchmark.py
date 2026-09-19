import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarks.benchmark_pipeline import run_pipeline_benchmark

def test_benchmark_speed_and_latency():
    # Run a 30-frame benchmark on CPU
    results = run_pipeline_benchmark(num_frames=30, use_onnx=True)
    
    # Assert criteria
    assert results['fps'] >= 20.0, f"Benchmark FPS {results['fps']:.1f} was below required threshold of 20 FPS"
    assert results['avg_latency'] < 100.0, f"Benchmark latency {results['avg_latency']:.1f}ms exceeded 100ms threshold"
    assert results['passed'], "Benchmark did not meet target speed criteria"
