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
# Env-overridable: SESSION, CHECKPOINTS, METRICS, LOGS, NVAL, NTEST,
#   PYTHON_BIN, PATIENCE, LEARNING_RATE, RESUME
# RESUME semantics:
#   auto (default) -> resume from latest_checkpoint.pt when it exists
#   none           -> always start a fresh run (ignore checkpoints)
#   <path>         -> resume from the chosen checkpoint file (e.g. a
#                     dated farsi_wham_sudo_epoch_N checkpoint), the
#                     path is resolved to absolute automatically.
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
PATIENCE=${PATIENCE:-60}
LEARNING_RATE=${LEARNING_RATE:-0.001}

# resolve the three output dirs to absolute paths before any other
# cwd-changing step: the script later cd's into the experiments dir to
# launch tmux, and relative paths here would then resolve from there
abspath() {
    case "$1" in
    /*) printf '%s\n' "$1";;
    *) printf '%s\n' "$(pwd)/$1";;
    esac
}
CHECKPOINTS="$(abspath "$CHECKPOINTS")"
METRICS="$(abspath "$METRICS")"
LOGS="$(abspath "$LOGS")"

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

# resume policy: auto (default) -> latest checkpoint when it exists,
# none -> fresh start, otherwise the value is the path of a chosen
# checkpoint (dated epoch backups included), resolved to absolute
RFC_FLAG=""
RESUME_POLICY=${RESUME:-auto}
case "$RESUME_POLICY" in
auto)
    if [[ -f "${CHECKPOINTS}/latest_checkpoint.pt" ]]; then
        RFC_FLAG="-rfc ${CHECKPOINTS}/latest_checkpoint.pt"
        echo "Resuming from latest checkpoint."
    fi
    ;;
none|"")
    ;;
*)
    RESUME_PATH="$(abspath "$RESUME_POLICY")"
    if [[ ! -f "$RESUME_PATH" ]]; then
        echo "ERROR: RESUME=$RESUME_POLICY -> file not found: $RESUME_PATH"
        exit 1
    fi
    RFC_FLAG="-rfc ${RESUME_PATH}"
    echo "Resuming from chosen checkpoint: $RESUME_PATH"
    ;;
esac

cat > "${CHECKPOINTS}/train_cmd.sh" <<EOF
cd "${EXPERIMENTS_ROOT}"
${PYTHON_BIN} run_farsi_wham_separation.py \\
    --train FARSI_WHAM --val FARSI_WHAM --test FARSI_WHAM \\
    --separation_task sep_noisy \\
    --min_num_sources 1 --max_num_sources 3 --n_channels 1 \\
    --model_type groupcomm_v2 \\
    --enc_kernel_size 41 --enc_num_basis 512 --out_channels 256 \\
    --in_channels 512 --num_blocks 8 --group_size 16 --upsampling_depth 5 \\
    --audio_timelength 4.0 -fs 16000 -bs 8 -lr ${LEARNING_RATE} --clip_grad_norm 5.0 \\
    --patience ${PATIENCE} \\
    --n_val ${NVAL} --n_test ${NTEST} \\
    -tags farsi_wham_gc \\
    --n_epochs 100 --n_train 20000 \\
    --checkpoints_path ${CHECKPOINTS} --save_checkpoint_every 2 \\
    --experiment_logs_path ${LOGS} \\
    --metrics_logs_path ${METRICS} \\
    --n_jobs 2 \\
    ${RFC_FLAG} \\
    ${extra_args[*]:-}
EOF

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists; not starting a new one."
    echo "Attach with: tmux attach -t $SESSION"
    exit 1
fi

# ---- pre-flight checks (fail fast before the tmux pane dies silently) ----
MISSING=()
for f in "$REPO_ROOT/__config__.py" "$EXPERIMENTS_ROOT/run_farsi_wham_separation.py"; do
    [[ -f "$f" ]] || MISSING+=("$f")
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "ERROR: missing runner files: ${MISSING[*]} (repo layout unexpected?)"
    exit 1
fi

"$PYTHON_BIN" - <<'PYEOF'
import importlib, sys
missing = []
for mod in ('torch', 'scipy', 'soundfile', 'glob2', 'musdb'):
    try:
        mod = __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    print('PYTHON ENV ERROR: missing modules: {}.'.format(
        ' '.join(missing)))
    print('-> Activate the training venv (or set PYTHON_BIN), and install '
          'requirements: pip install -r requirements_gpu_training.txt')
    sys.exit(1)
PYEOF
if [[ $? -ne 0 ]]; then
    exit 1
fi

# the FARSI_WHAM loader requires the index jsons built by
# prepare_farsi_wham_cache: check the paths from __config__
if ! "$PYTHON_BIN" - <<PYEOF
import os, sys
sys.path.insert(0, '${REPO_ROOT}')
from __config__ import FARSI_WHAM_ROOT_PATH, WHAM_NOISE_ROOT_PATH
ok = True
for p in (os.path.join(FARSI_WHAM_ROOT_PATH, 'speech_index.json'),
          os.path.join(WHAM_NOISE_ROOT_PATH, 'noise_index.json')):
    if not os.path.lexists(p):
        print('CACHE MISSING: {} does not exist.'.format(p))
        ok = False
if not ok:
    print('-> Build the caches first with:')
    print('   python -m sudo_rm_rf.utils.prepare_farsi_wham_cache '
          '--speech_out {} --noise_out {}'.format(
              FARSI_WHAM_ROOT_PATH, WHAM_NOISE_ROOT_PATH))
    sys.exit(1)
PYEOF
then
    echo "ERROR: training data caches are missing (or __config__ failed to import); see the message above."
    exit 1
fi

# everything that tmux will print is also persisted here, so that
# instant-crashing panes leave a readable error behind
cd "$EXPERIMENTS_ROOT"
tmux new-session -d -s "$SESSION" \
    "bash \"${CHECKPOINTS}/train_cmd.sh\" 2>&1 | tee ${CHECKPOINTS}/train_out.log" \
    || { echo "tmux launch failed."; exit 1; }

sleep 2
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session died within 2s of launch. Last output lines:"
    tail -n 15 "${CHECKPOINTS}/train_out.log" 2>/dev/null || \
        echo "(no output captured)"
    echo "Full output: ${CHECKPOINTS}/train_out.log"
    exit 1
fi

PID=$(pgrep -fo run_farsi_wham_separation.py || true)
cat > "${CHECKPOINTS}/run_info.json" <<EOF
{"started": "$(date '+%Y-%m-%dT%H:%M:%S%z')", "tmux_session": "$SESSION",
 "runner_pid": "${PID:-unknown}", "checkpoints": "${CHECKPOINTS}",
 "metrics": "${METRICS}", "logs": "${LOGS}", "resumed": "${RESUME_POLICY}${RFC_FLAG:+ ${RFC_FLAG#-rfc }}"}
EOF

echo "Training launched in tmux session: $SESSION"
echo "  attach live output:  tmux attach -t $SESSION"
echo "  status check:        python ${REPO_ROOT}/sudo_rm_rf/utils/monitor_training.py \\
    --checkpoints_path $CHECKPOINTS --metrics_path $METRICS"
