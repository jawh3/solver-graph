#!/bin/bash
# Qwen3-32B-AWQ - Reduced memory footprint
# Lowered gpu_memory_utilization to 0.85 and context to 24k to fit

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "Starting vLLM Server (Qwen3.5-27B-FP8, 32k Context)..."

vllm serve Qwen/Qwen3.5-27B-FP8 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.80 \
  --disable-custom-all-reduce \
  --max-num-seqs 8 \
  --trust-remote-code \
  --port 8000
