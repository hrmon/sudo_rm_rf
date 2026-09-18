"""!
@brief Offline evaluation of a FARSI_WHAM separation checkpoint.

Loads a saved checkpoint, rebuilds the model, runs the 1/2/3-speaker
FARSI_WHAM validation/test buckets (deterministic recipes from the
farsi_wham loader) and reports, per bucket and overall:

    SNR, SNRi, SI-SNR, SI-SNRi, SI-SDR, SI-SDRi

under PIT (best permutation of the estimated channels). Improvements
(i) are computed against the actual noisy input mixture as baseline,
which is the meaningful reference for noisy separation (x = s1+s2+s3+n).

Note: SI-SNR and SI-SDR are aliases of the same quantity (scale
invariant signal-to-noise ratio); both names are reported so external
tools can pick up either.

Usage example:
    python sudo_rm_rf/utils/eval_farsi_wham_checkpoint.py \
        --checkpoint runs/checkpoints/latest_checkpoint.pt \
        --split both --n_max_samples 1000 --out results.csv

Exit code 0 on success.

@author Hamidreza (adapted from repo conventions)
"""

import argparse
import csv
import itertools
import os
import sys

import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, repo_root)

import sudo_rm_rf.dnn.experiments.utils.dataset_setup as dataset_setup
import sudo_rm_rf.dnn.models.groupcomm_sudormrf_v2 as sudormrf_gc_v2


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate a FARSI_WHAM checkpoint.')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to a saved .pt checkpoint (e.g. '
                             'checkpoints/latest_checkpoint.pt or a dated '
                             'farsi_wham_sudo_epoch_N file).')
    parser.add_argument('--split', type=str, default='both',
                        choices=['val', 'test', 'both'],
                        help='Which split to evaluate.')
    parser.add_argument('--n_max_samples', type=int, default=1000,
                        help='Maximum samples per bucket (0 = loader '
                             'default of 10000).')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_jobs', type=int, default=2)
    parser.add_argument('--out', type=str, default=None,
                        help='Optional csv path for the result table.')
    parser.add_argument('--device', type=str, default=None,
                        help='cuda or cpu; defaults to cuda when '
                             'available.')
    parser.add_argument('--fs', type=int, default=16000)
    parser.add_argument('--audio_timelength', type=float, default=4.0)
    return parser


def build_model(hparams):
    if hparams['model_type'] != 'groupcomm_v2':
        raise ValueError('This eval script supports the groupcomm_v2 '
                         'model, got {}.'.format(hparams['model_type']))
    return sudormrf_gc_v2.GroupCommSudoRmRf(
        in_audio_channels=1,
        out_channels=hparams['out_channels'],
        in_channels=hparams['in_channels'],
        num_blocks=hparams['num_blocks'],
        upsampling_depth=hparams['upsampling_depth'],
        enc_kernel_size=hparams['enc_kernel_size'],
        enc_num_basis=hparams['enc_num_basis'],
        num_sources=hparams['max_num_sources'],
        group_size=16)


def load_state_dict(model, checkpoint):
    sd = checkpoint['model_state_dict']
    # tolerate DataParallel-wrapped checkpoints
    if any(k.startswith('module.') for k in sd):
        sd = dict([(k.split('.', 1)[1], v) for k, v in sd.items()])
    model.load_state_dict(sd)


def sdr_metrics(est, tgt, mix, si=True, eps=1e-8):
    """SI-SNR or plain SNR between one estimate and one target.
    est, tgt, mix: 1D numpy arrays of equal length."""
    est = est - est.mean()
    tgt = tgt - tgt.mean()
    if si:
        proj = (np.dot(est, tgt) / (np.dot(tgt, tgt) + eps)) * tgt
    else:
        proj = tgt
    noise = est - proj
    return 10.0 * np.log10(((proj ** 2).mean() + eps) /
                           ((noise ** 2).mean() + eps))


def eval_batch(estimates, targets, mixtures):
    """Compute PIT SNR / SI-SNR (+ improvements) for one batch.

    estimates: [B, n_est, T] torch tensor
    targets:   [B, n_active, T] torch tensor (only actually active rows)
    mixtures:  [B, 1, T]

    Returns a list of dicts (one per sample).
    """
    results = []
    B = estimates.shape[0]
    for b in range(B):
        est = estimates[b].cpu().numpy()
        tgts = targets[b].cpu().numpy()
        mix = mixtures[b, 0].cpu().numpy()
        n_act = tgts.shape[0]
        perms = list(itertools.permutations(range(est.shape[0]), n_act))
        # best permutation: maximum sum of per-source SI-SNR
        best = None
        for perm in perms:
            total = 0.
            for i in range(n_act):
                total += sdr_metrics(est[perm[i]], tgts[i], mix, si=True)
            if best is None or total > best[0]:
                best = (total, perm)
        _, best_perm = best
        si_snrs = [sdr_metrics(est[best_perm[i]], tgts[i], mix, si=True)
                   for i in range(n_act)]
        snrs = [sdr_metrics(est[best_perm[i]], tgts[i], mix, si=False)
                for i in range(n_act)]
        baselines_si = [sdr_metrics(mix, tgts[i], mix, si=True)
                        for i in range(n_act)]
        baselines_s = [sdr_metrics(mix, tgts[i], mix, si=False)
                       for i in range(n_act)]
        results.append({
            'n_active': n_act,
            'SNR': float(np.mean(snrs)),
            'SNRi': float(np.mean(snrs) - np.mean(baselines_s)),
            'SI-SNR': float(np.mean(si_snrs)),
            'SI-SNRi': float(np.mean(si_snrs) - np.mean(baselines_si)),
            'SI-SDR': float(np.mean(si_snrs)),
            'SI-SDRi': float(np.mean(si_snrs) - np.mean(baselines_si)),
        })
    return results


def main():
    args = build_arg_parser().parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    hparams = checkpoint['hparams']
    model = build_model(hparams)
    load_state_dict(model, checkpoint)
    model.to(device).eval()
    print('Loaded checkpoint: {} (epoch {}, tr_step {})'.format(
        args.checkpoint, checkpoint.get('epoch'),
        checkpoint.get('tr_step')))

    splits = ['val', 'test'] if args.split == 'both' else [args.split]
    metric_names = ['SNR', 'SNRi', 'SI-SNR', 'SI-SNRi',
                    'SI-SDR', 'SI-SDRi']

    all_rows = []
    for split in splits:
        for n_src in range(1, hparams['max_num_sources'] + 1):
            loader = dataset_setup.create_loader_for_simple_dataset(
                dataset_name='FARSI_WHAM',
                separation_task='sep_noisy',
                data_split=split, sample_rate=args.fs,
                n_channels=1, min_or_max='max', zero_pad=True,
                timelegth=args.audio_timelength, normalize_audio=False,
                n_samples=args.n_max_samples,
                min_num_sources=n_src, max_num_sources=n_src)
            gen = loader.get_generator(batch_size=args.batch_size,
                                       num_workers=args.n_jobs,
                                       shuffle=False)
            bucket_results = []
            with torch.no_grad():
                for data in gen:
                    mixtures = data['mixture'].to(device)
                    targets = data['targets'][:, :n_src].to(device)
                    # joint normalization: input and references by the
                    # same scale (as in the training/val runner)
                    input_mix_std = mixtures.std(-1, keepdim=True)
                    mixtures = mixtures / (input_mix_std + 1e-8)
                    targets = targets / (input_mix_std + 1e-8)
                    estimates = model(mixtures)
                    bucket_results += eval_batch(estimates, targets,
                                                 mixtures)
            row = {'split': split, 'n_src': n_src,
                   'n_samples': len(bucket_results)}
            for metric_name in metric_names:
                values = np.array([r[metric_name]
                                   for r in bucket_results])
                row[metric_name] = float(np.mean(values))
                row['{}_median'.format(metric_name)] = float(
                    np.median(values))
                row['{}_std'.format(metric_name)] = float(np.std(values))
            per_metric_means = {m: row[m] for m in metric_names}
            print('== {} {}-speaker samples: {} =='.format(
                split, n_src, row['n_samples']))
            for metric_name in metric_names:
                print('  {:<8} mean {:>7.3f} median {:>7.3f} std {:>6.3f}'
                      ''.format(metric_name, row[metric_name],
                                row['{}_median'.format(metric_name)],
                                row['{}_std'.format(metric_name)]))
            all_rows.append(row)

    if args.out is not None:
        fieldnames = list(all_rows[0].keys())
        with open(args.out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print('Wrote: {}'.format(args.out))

    print('\nOverall (mean over all buckets):')
    for split in splits:
        subset = [r for r in all_rows if r['split'] == split]
        for metric_name in metric_names:
            mean = float(np.mean([row[metric_name] for row in subset]))
            print('  {:<4} {:<8} mean {:>7.3f}'.format(
                split, metric_name, mean))


if __name__ == '__main__':
    main()
