#!/bin/bash
# Serve Qwen2.5-32B-Instruct-AWQ (TP=2) from the GRADER env (vllm 0.19.1, intact).
# AWQ is well-supported on Ampere; enforce-eager avoids the cudagraph-capture OOM.
# (ctf_solver's vllm 0.7.3 was broken by a sentence-transformers dep upgrade; we only
#  serve from grader and orchestrate from ctf_solver — they talk over HTTP.)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1
exec ${VLLM:-vllm} serve Qwen/Qwen2.5-32B-Instruct-AWQ \
  --tensor-parallel-size 2 \
  --max-model-len 6144 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 24 \
  --enforce-eager \
  --enable-prefix-caching \
  --port 8000
