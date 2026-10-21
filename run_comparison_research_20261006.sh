#!/usr/bin/env bash
# Corrected ClipErase/Zero-Shot comparisons, sequentially on physical GPU 0.
# Each entry writes its own On_*/outputs/*_comparison_20261006.txt plus
# audit-only MIA scores.  The files are research audits, not DP releases.
set -uo pipefail
COMPARISON_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$COMPARISON_REPO" || exit 1
source /home/wwq/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate py313wwq || exit 1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
COMPARISON_LOG_ROOT="$COMPARISON_REPO/codex_logs/comparison_research_20261006/full"
mkdir -p "$COMPARISON_LOG_ROOT"
exec 9>"$COMPARISON_REPO/codex_logs/comparison_research_20261006/gpu0_queue.lock"
if ! flock -n 9; then
    echo "Another corrected comparison queue already holds the GPU 0 lock." >&2
    exit 1
fi
printf 'started_utc\t%s\npid\t%s\n' "$(date -u +%FT%TZ)" "$$" > "$COMPARISON_LOG_ROOT/queue_status_20261006.tsv"
COMPARISON_FAILURES=0
for COMPARISON_ENTRY in \
    On_CIFAR100/unlearning_clip_by_subliminal.py \
    On_TinyImageNet/unlearning_clip_by_subliminal.py \
    On_CIFAR100/unlearning_SigLIP_by_subliminal.py \
    On_TinyImageNet/unlearning_SigLIP_by_subliminal.py
do
    COMPARISON_LABEL="${COMPARISON_ENTRY//\//_}"
    COMPARISON_LOG="$COMPARISON_LOG_ROOT/${COMPARISON_LABEL%.py}_comparison_20261006.log"
    printf 'running\t%s\t%s\n' "$COMPARISON_ENTRY" "$(date -u +%FT%TZ)" >> "$COMPARISON_LOG_ROOT/queue_status_20261006.tsv"
    if python -u "$COMPARISON_ENTRY" --sweep comparison_research --gpu 0 --run-tag 20261006 \
        --seeds 43 44 45 --train-batch-size 128 --eval-batch-size 128 \
        --num-workers 4 --private-batch-size 16 \
        --comparison-methods cliperase zero_shot_lipschitz \
        --comparison-pareto-scales 0 0.0625 0.125 0.25 0.5 0.75 1 \
        --comparison-hard-search-steps 12 \
        --prompt-template 'a photo of a {}.' --print-progress > "$COMPARISON_LOG" 2>&1
    then
        printf 'complete\t%s\t%s\n' "$COMPARISON_ENTRY" "$(date -u +%FT%TZ)" >> "$COMPARISON_LOG_ROOT/queue_status_20261006.tsv"
    else
        COMPARISON_EXIT=$?
        COMPARISON_FAILURES=$((COMPARISON_FAILURES + 1))
        printf 'failed\t%s\texit=%s\t%s\n' "$COMPARISON_ENTRY" "$COMPARISON_EXIT" "$(date -u +%FT%TZ)" >> "$COMPARISON_LOG_ROOT/queue_status_20261006.tsv"
    fi
done
printf 'finished\tfailures=%s\t%s\n' "$COMPARISON_FAILURES" "$(date -u +%FT%TZ)" >> "$COMPARISON_LOG_ROOT/queue_status_20261006.tsv"
test "$COMPARISON_FAILURES" -eq 0
