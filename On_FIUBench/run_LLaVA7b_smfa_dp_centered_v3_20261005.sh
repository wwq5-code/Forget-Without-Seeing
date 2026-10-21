#!/usr/bin/env bash
set -euo pipefail

source /home/wwq/miniconda3/etc/profile.d/conda.sh
conda activate py313wwq
cd /home/wwq/VLM_unl_without_seeing

python On_FIUBench/unlearning_LLaVA7b_by_subliminal.py \
  --output-dir On_FIUBench/outputs/unlearning_LLaVA7b_smfa_dp_centered_v3_20261005 \
  --summary-path On_FIUBench/unlearning_LLaVA7b_smfa_dp_centered_v3_20261005.txt \
  --capacity-checkpoint-dir On_FIUBench/outputs/unlearning_LLaVA7b_smfa_dp_20261005 \
  --functional-projection-cache On_FIUBench/outputs/unlearning_LLaVA7b_smfa_smoke_v26_publicproj512_20261005/public_functional_projection_proxy_cache.pt \
  --functional-projection-dim 6144 \
  --functional-projection-records 24 \
  --functional-local-delta-scale 2.8 \
  --functional-clip-norm 6.3 \
  --functional-center-on-public-retain \
  --functional-postrelease-retain-blend 0.5 \
  --functional-postrelease-retain-repair-steps 0 \
  --dp-epsilons inf 8192 4096 2048 1024 512 \
  --dp-noise-repeats 5 \
  --eval-qa-per-identity 4 \
  --sample-answers 4 \
  --no-carrier-capacity-sweep \
  --local-files-only \
  --print-progress
