#!/bin/bash
# Run GPU benchmark with pixi environment
# Usage: ./scripts/run_benchmark.sh [config] [output]

CONFIG=${1:-quick}
OUTPUT=${2:-reports/gpu_benchmark_results.json}

cd /mnt/c/Users/jcwen/Projects/aoty_pred_pub

# Ensure we're using pixi
export PATH="$HOME/.pixi/bin:$PATH"

echo "Running GPU benchmark with config: $CONFIG"
echo "Output: $OUTPUT"

pixi run python scripts/benchmark_gpu.py --config "$CONFIG" --output "$OUTPUT"
