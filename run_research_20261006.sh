#!/usr/bin/env bash
# Four sequential full experiments, all on physical GPU 0, in py313wwq.
# Each Python entry writes its OWN On_*/outputs/<script>_20261006.txt.
# Start once on wwq_server with: nohup bash run_research_20261006.sh > ... 2>&1 &
# Never launch four GPU jobs concurrently or overwrite an existing result.
set -uo pipefail
RESEARCH_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$RESEARCH_REPO" || exit 1
source /home/wwq/miniconda3/etc/profile.d/conda.sh || exit 1
conda activate py313wwq || exit 1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
RESEARCH_LOG_ROOT="$RESEARCH_REPO/codex_logs/research_20261006/full"
mkdir -p "$RESEARCH_LOG_ROOT"
exec 9>"$RESEARCH_REPO/codex_logs/research_20261006/gpu0_queue.lock"
if ! flock -n 9; then
    echo "Another 20261006 GPU 0 queue already holds the lock." >&2
    exit 1
fi
printf 'started_utc\t%s\npid\t%s\n' "$(date -u +%FT%TZ)" "$$" > "$RESEARCH_LOG_ROOT/queue_status_20261006.tsv"
RESEARCH_FAILURES=0
for RESEARCH_ENTRY in \
    On_CIFAR100/unlearning_clip_by_subliminal.py \
    On_TinyImageNet/unlearning_clip_by_subliminal.py \
    On_CIFAR100/unlearning_SigLIP_by_subliminal.py \
    On_TinyImageNet/unlearning_SigLIP_by_subliminal.py
do
    RESEARCH_LABEL="${RESEARCH_ENTRY//\//_}"
    RESEARCH_LOG="$RESEARCH_LOG_ROOT/${RESEARCH_LABEL%.py}_20261006.log"
    printf 'running\t%s\t%s\n' "$RESEARCH_ENTRY" "$(date -u +%FT%TZ)" >> "$RESEARCH_LOG_ROOT/queue_status_20261006.tsv"
    if python -u "$RESEARCH_ENTRY" --sweep research --gpu 0 --run-tag 20261006 \
        --seeds 43 44 45 --train-batch-size 128 --eval-batch-size 128 \
        --num-workers 4 --private-batch-size 16 --noise-repeats 3 \
        --prompt-template 'a photo of a {}.' --print-progress > "$RESEARCH_LOG" 2>&1
    then
        printf 'complete\t%s\t%s\n' "$RESEARCH_ENTRY" "$(date -u +%FT%TZ)" >> "$RESEARCH_LOG_ROOT/queue_status_20261006.tsv"
    else
        RESEARCH_EXIT=$?
        RESEARCH_FAILURES=$((RESEARCH_FAILURES + 1))
        printf 'failed\t%s\texit=%s\t%s\n' "$RESEARCH_ENTRY" "$RESEARCH_EXIT" "$(date -u +%FT%TZ)" >> "$RESEARCH_LOG_ROOT/queue_status_20261006.tsv"
    fi
done
printf 'finished\tfailures=%s\t%s\n' "$RESEARCH_FAILURES" "$(date -u +%FT%TZ)" >> "$RESEARCH_LOG_ROOT/queue_status_20261006.tsv"
test "$RESEARCH_FAILURES" -eq 0
