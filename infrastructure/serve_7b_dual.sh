#!/bin/bash
# Serve TWO Qwen2.5-7B-Instruct TP=1 servers (one per GPU) from the grader env (vllm 0.19.1,
# intact; ctf_solver's vllm got broken by a dep upgrade). Reliable + fast; the campaign
# round-robins shards across :8000 and :8001 for throughput.
VLLM=${VLLM:-vllm}
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
  --max-model-len 8192 --gpu-memory-utilization 0.90 --max-num-seqs 48 \
  --enable-prefix-caching --port 8000 > logs/vllm7b_g0.log 2>&1 < /dev/null &
echo "GPU0:8000 pid $!"
CUDA_VISIBLE_DEVICES=1 setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct \
  --max-model-len 8192 --gpu-memory-utilization 0.90 --max-num-seqs 48 \
  --enable-prefix-caching --port 8001 > logs/vllm7b_g1.log 2>&1 < /dev/null &
echo "GPU1:8001 pid $!"
