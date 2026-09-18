"""!
@brief Plot the FARSI_WHAM training trajectory from a run's
metrics.jsonl (companion to run_farsi_wham_separation.py).

Outputs a two-panel chart:
 - top:   per-epoch training loss (tr_back_loss_VAR_SPEAKER_SNR)
 - bottom: absolute SI-SDR for the 1/2/3-speaker validation buckets
plus markers of the LR at each epoch.

Usage:
    python sudo_rm_rf/utils/plot_farsi_training.py \
        --metrics_path runs/metrics --out train_history.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'tr': 'tab:gray',
          1: 'tab:blue',
          2: 'tab:orange',
          3: 'tab:green'}


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Plot FARSI_WHAM training history from metrics.jsonl.')
    parser.add_argument('--metrics_path', type=str, default='./metrics',
                        help='Dir containing metrics.jsonl.')
    parser.add_argument('--out', type=str, default='train_history.png',
                        help='Output png path.')
    return parser


def main():
    args = build_arg_parser().parse_args()
    metrics_file = os.path.join(args.metrics_path, 'metrics.jsonl')
    rows = []
    with open(metrics_file) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    epochs = [row['epoch'] for row in rows]
    lrs = [row['lr'] for row in rows]
    train_loss = [row['losses'].get('tr_back_loss_VAR_SPEAKER_SNR', {})
                  .get('mean') for row in rows]
    val = {n: [row['losses'].get('val_{}_srcs_SISDR'.format(n), {})
               .get('mean') for row in rows] for n in (1, 2, 3)}

    fig, (ax_tr, ax_val) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={'height_ratios': [1, 1.4]})
    fig.suptitle('FARSI_WHAM training history '
                 '({} epochs)'.format(len(rows)), fontsize=13)

    ax_tr.plot(epochs, train_loss, color=COLORS['tr'], marker='o', ms=3,
               lw=1.5, label='train loss (neg. SNR sum)')
    ax_tr.set_ylabel('train loss')
    ax_tr.grid(alpha=.3)
    ax_tr.legend(loc='best', fontsize=9)

    for n in (1, 2, 3):
        ax_val.plot(epochs, val[n], color=COLORS[n], marker='o', ms=3,
                    lw=1.8, label='val {}-speaker'.format(n))
    ax_val.set_xlabel('epoch')
    ax_val.set_ylabel('vi SI-SDR (dB)')
    ax_val.grid(alpha=.3)
    ax_val.legend(loc='best', fontsize=9)

    # lr milestones as vertical dashed lines on both panels
    seen = set()
    for ep, lr in zip(epochs, lrs):
        if lr not in seen:
            seen.add(lr)
            for ax in (ax_tr, ax_val):
                ax.axvline(ep, color='k', ls=':', alpha=.4, lw=1)
    ax_tr.text(epochs[0], ax_tr.get_ybound()[0], ' : lr changes',
               fontsize=8, va='bottom', color='k')

    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(args.out, dpi=150)
    print('saved: {}'.format(args.out))


if __name__ == '__main__':
    main()
