#!/bin/bash
# Detached follow-on: wait for the 7B campaign's node_values.json, then run the H5 transfer
# experiment (uses the GPUs after the campaign finishes). Independent of the main pipeline —
# if it fails, the campaign result still stands.
cd ${REPO:-$(pwd)}
export PYTHONPATH=${REPO:-$(pwd)}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export VLLM_URL=http://127.0.0.1:8000/v1 MODEL_NAME=Qwen/Qwen2.5-7B-Instruct MAX_OUTPUT_TOKENS=256
NV=results/lab/campaign_7b/merged/node_values.json
echo "[h5-followon] waiting for $NV ..."
until [ -f "$NV" ]; do sleep 120; done
sleep 60   # let analysis finish writing all artifacts
echo "[h5-followon] node_values ready; launching H5 at $(date)"
python -m experiments.lab_h5 \
  --values-file "$NV" \
  --families-file results/lab/probe_7b/selection_merged.json \
  --n-inst 8 --seeds 5 --workers 16 --slack 2 \
  --out results/lab/h5_7b
echo "[h5-followon] DONE at $(date)"
