#!/usr/bin/env python3
"""!
@brief One-shot health status for a running FARSI_WHAM training session
(monitoring companion for sudo_rm_rf/utils/start_training.sh).

Designed to be executed by hand or by an opencode session whenever a
check-in is wanted:

    python sudo_rm_rf/utils/monitor_training.py \
        --checkpoints_path checkpoints --metrics_path metrics
        --stall_minutes 120

Prints:
 - last epochs of metrics.jsonl (train loss + per-bucket SI-SDR means)
 - epoch cadence (time since the last epoch got written)
 - checkpoint freshness + which epoch is stored in latest_checkpoint.pt
 - runner process liveness (pgrep -f run_farsi_wham_separation.py)
 - GPU memory and utilization (if nvidia-smi is available)

Exit codes for automation-friendly checks:
    0  healthy (epoch visible and fresh, results get written)
    1  warning (epochs not progressing but within the stall window, or
       no runner process while metrics were recently written)
    2  stalled (no new epoch within --stall_minutes) or no run at all

@author Hamidreza (adapted from repo conventions)
"""

import argparse
import glob
import json
import os
import re
import subprocess
import time


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='FARSI_WHAM training status check.')
    parser.add_argument('--metrics_path', type=str, default='./metrics',
                        help='Dir where the runner writes metrics.jsonl.')
    parser.add_argument('--checkpoints_path', type=str,
                        default='./checkpoints',
                        help='Dir where the runner writes checkpoints.')
    parser.add_argument('--tail', type=int, default=5,
                        help='Number of epochs to print from the tail.')
    parser.add_argument('--stall_minutes', type=float, default=180.,
                        help='Raise error exit code 2 when no epochs were '
                             'written within this many minutes.')
    parser.add_argument('--warn_minutes', type=float, default=60.,
                        help='Raise the warning exit code when no epochs '
                             'were written within this many minutes.')
    return parser


def read_metrics(metrics_path):
    metrics_file = os.path.join(metrics_path, 'metrics.jsonl')
    if not os.path.exists(metrics_file):
        return None
    rows = []
    with open(metrics_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    return rows, metrics_file


def format_float(value):
    if value is None or value != value:
        return 'nan'
    return '{:.2f}'.format(value)


def runner_pids():
    try:
        out = subprocess.run(['pgrep', '-f',
                              'run_farsi_wham_separation.py'],
                             capture_output=True, text=True)
        return [int(p) for p in out.stdout.split()]
    except Exception:
        return []


def gpu_info():
    try:
        out = subprocess.run(
            ['nvidia-smi',
             '-query-gpu=memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'], capture_output=True,
            text=True, timeout=10)
        used, total, util = out.stdout.strip().split('\n')[0].split(', ')
        return '{} / {} MiB VRAM, {} % util'.format(used, total, util)
    except Exception:
        return None


def main():
    args = build_arg_parser().parse_args()
    exit_code = 0

    metrics = read_metrics(args.metrics_path)
    if metrics is None:
        print('NO RUN: no metrics.jsonl under {} yet.'.format(
            args.metrics_path))
        print('GPU: {}'.format(gpu_info() or 'no nvidia-smi'))
        print('Runner processes: {}'.format(runner_pids() or ['none']))
        return 2
    rows, metrics_file = metrics

    epoch = rows[-1]['epoch']
    cadence = time.time() - os.path.getmtime(metrics_file)
    cadence_str = '{:.1f} min ago'.format(cadence / 60.)

    print('== latest {} epoch(s) =='.format(args.tail))
    for row in rows[-args.tail:]:
        parts = ['epoch {:>4} lr {:>9}'.format(row['epoch'], row['lr'])]
        for loss_name, value in sorted(row['losses'].items()):
            parts.append('{}={}'.format(
                loss_name.replace('_srcs_SISDR', ''),
                format_float(value.get('mean'))))
        print(' '.join(parts))

    ckpt_path = os.path.join(args.checkpoints_path,
                             'latest_checkpoint.pt')
    ckpt_str = 'MISSING'
    if os.path.lexists(ckpt_path):
        age = (time.time() - os.path.getmtime(ckpt_path)) / 3600.
        ckpt_str = 'epoch {:.0f}-ish, {:.1f} h old'.format(
            epoch, age)
    print('== checkpoint ==')
    print('latest_checkpoint.pt: {} ({})'.format(
        'exists' if os.path.lexists(ckpt_path) else 'missing', ckpt_str))

    pids = runner_pids()
    print('== runner ==')
    print('processes: {}'.format(
        ', '.join(str(p) for p in pids) if pids else 'NOT RUNNING'))
    gpu = gpu_info()
    print('GPU: {}'.format(gpu if gpu else 'no nvidia-smi'))

    print('== cadence ==')
    print('last epoch written: {} (thresholds: warn {} min, '
          'stall {} min)'.format(
              cadence_str, args.warn_minutes, args.stall_minutes))

    if cadence > args.stall_minutes * 60.:
        print('STALLED: no new epochs within the last {} minutes!'.format(
            args.stall_minutes))
        exit_code = 2

    # If the process is gone while the epoch cadence is fine, it simply
    # finished or the machine's tmux was left without a process.
    if cadence > args.warn_minutes * 60. and exit_code == 0:
        if not pids:
            print('WARNING: runner not running but epochs '
                  'are this fresh: ambiguous state!')
            exit_code = 2
        else:
            print('WARNING: training may be slow: {} without a new epoch.'
                  ''.format(cadence_str))
            exit_code = 1

    if exit_code == 0:
        print('STATUS: OK')
    elif exit_code == 1:
        print('STATUS: WARNING')
    else:
        print('STATUS: PROBLEM')
    return exit_code


if __name__ == '__main__':
    import sys
    sys.exit(main())
