#!/usr/bin/env bash
# @brief Launch the FARSI_WHAM training runner inside a detached tmux
# session, so it survives SSH disconnects. Recording basic run metadata
# (stdout tee, PID, args) for the companion status script:
#   sudo_rm_rf/utils/monitor_training.py
#
# Usage examples (paths/dirs created if missing):
#   ./sudo_rm_rf/utils/start_training.sh
#   ./sudo_rm_rf/utils/start_training.sh \
#     -- ~/extra_runner_args
# Env-overridable: SESSION, CHECKPOINTS, METRICS, LOGS, NVAL, NTEST, PYTHON_BIN
# Attach to the live output with:  tmux attach -t farsi_wham

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXPERIMENTS_ROOT="$REPO_ROOT/sudo_rm_rf/dnn/experiments"

SESSION=${SESSION:-farsi_wham}
CHECKPOINTS=${CHECKPOINTS:-./checkpoints}
METRICS=${METRICS:-./metrics}
LOGS=${LOGS:-./experiment_logs}
NVAL=${NVAL:-1000}
NTEST=${NTEST:-1000}
PYTHON_BIN=${PYTHON_BIN:-python}

extra_args=()
if [[ $# -gt 0 ]]; then
    if [[ "$1" != "--" ]]; then
        echo "Usage: [SESSION=... CHECKPOINTS=... METRICS=... LOGS=... NVAL=... NTEST=... PYTHON_BIN=...] $0 -- [extra runner args]"
        exit 1
    fi
    shift
    extra_args=("$@")
fi

mkdir -p "$CHECKPOINTS" "$METRICS" "$LOGS"
RUN_INFO_PATH="${CHECKPOINTS}/run_info.json"

# resume automatically when a latest checkpoint already exists
RESUME=""
if [[ -f "${CHECKPOINTS}/latest_checkpoint.pt" ]]; then
    RESUME="-rfc latest"
    echo "Found checkpoint: ${CHECKPOINTS}/latest_checkpoint.pt -> resuming."
fi

cat > "${CHECKPOINTS}/train_cmd.sh" <<EOF
cd "${EXPERIMENTS_ROOT}"
${PYTHON_BIN} run_farsi_wham_separation.py \\
    --train FARSI_WHAM --val FARSI_WHAM --test FARSI_WHAM \\
    --separation_task sep_noisy \\
    --min_num_sources 1 --max_num_sources 3 --n_channels 1 \\
    --model_type groupcomm_v2 \\
    --enc_kernel_size 41 --enc_num_basis 512 --out_channels 256 \\
    --in_channels 512 --num_blocks 8 --group_size 16 --upsampling_depth 5 \\
    --audio_timelength 4.0 -fs 16000 -bs 8 -lr 0.002 --clip_grad_norm 5.0 \\
    --patience 20 \\
    --n_val ${NVAL} --n_test ${NTEST} \\
    -tags farsi_wham_gc \\
    --n_epochs 100 --n_train 20000 \\
    --checkpoints_path ${CHECKPOINTS} --save_checkpoint_every 2 \\
    --experiment_logs_path ${LOGS} \\
    --metrics_logs_path ${METRICS} \\
    --n_jobs 2 \\
    ${RESUME} \\
    ${extra_args[*]:-}
EOF

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists; not starting a new one."
    echo "Attach with: tmux attach -t $SESSION"
    exit 1
fi

cd "$EXPERIMENTS_ROOT"
tmux new-session -d -s "$SESSION" "bash ${CHECKPOINTS}/train_cmd.sh 2>&1" \
    || { echo "tmux launch failed."; exit 1; }

sleep 2
PID=$(pgrep -fo run_farsi_wham_separation.py || true)
cat > "${CHECKPOINTS}/run_info.json" <<EOF
{"started": "$(date -Is)", "tmux_session": "$SESSION",
 "runner_pid": "${PID:-unknown}", "checkpoints": "${CHECKPOINTS}",
 "metrics": "${METRICS}", "logs": "${LOGS}", "resumed": "$RESUME"}
EOF

echo "Training launched in tmux session: $SESSION"
echo "  attach live output:  tmux attach -t $SESSION"
echo "  status check:        python utils/monitor_training.py --checkpoints_path $CHECKPOINTS --metrics_path $METRICS"
