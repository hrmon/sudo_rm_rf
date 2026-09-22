"""!
@brief Inference with a selected FARSI_WHAM checkpoint over the samples
of the farsi-youtube-wham-3spk dataset (LibriMix-style layout).

Expected dataset layout (found on the repo path data/...):
    <data_root>/mix/<sample_id>.wav          actual model input
    <data_root>/mix_clean/<sample_id>.wav    noiseless mixture (optional)
    <data_root>/s{1,2,3}_reverb/<sample_id>.wav   reference sources
    <data_root>/metadata.csv                 sample_id = first column

For every mixture file, the script runs the model, writes the three
estimated sources as

    <out>/<sample_id>_s1.wav .. _s3.wav

and (optionally, --with_metrics) computes per-sample SNR/SNRi/SI-SNR/
SI-SNRi against the s{1..3}_reverb references under PIT, with the noisy
mixture as the improvement baseline, and writes a summary CSV.

Usage:
    python sudo_rm_rf/utils/infer_farsi_youtube_wham.py \
        --checkpoint runs/checkpoints/latest_checkpoint.pt \
        --data_root data/farsi-youtube-wham-3spk --with_metrics \
        --out runs/inference/epoch_30

@author Hamidreza (adapted from repo conventions)
"""

import argparse
import csv
import glob
import os
import sys
import itertools

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, repo_root)

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

import sudo_rm_rf.dnn.models.groupcomm_sudormrf_v2 as sudormrf_gc_v2

EPS = 1e-8
METRIC_NAMES = ['SNR', 'SNRi', 'SI-SNR', 'SI-SNRi']


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Run inference over farsi-youtube-wham-3spk samples '
                    'with a FARSI_WHAM checkpoint.')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to a saved .pt checkpoint.')
    parser.add_argument('--data_root', type=str,
                        default='data/farsi-youtube-wham-3spk')
    parser.add_argument('--out', type=str, default=None,
                        help='Output dir for the estimated sources; '
                             'default: <data_root>/estimates/'
                             '<checkpoint_stem>')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Only process the first N mixtures (0=all).')
    parser.add_argument('--device', type=str, default=None,
                        help='cuda or cpu; defaults to cuda when '
                             'available.')
    parser.add_argument('--with_metrics', action='store_true',
                        help='Also compute SNR/SNRi/SI-SNR/SI-SNRi '
                             'against the s%d_reverb references (PIT).')
    parser.add_argument('--metrics_csv', type=str, default=None,
                        help='Optional csv output for the per-sample '
                             'metrics (implies --with_metrics).')
    return parser


def build_model(hparams):
    if hparams['model_type'] != 'groupcomm_v2':
        raise ValueError('This script supports the groupcomm_v2 model, '
                         'got {}.'.format(hparams['model_type']))
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
    if any(k.startswith('module.') for k in sd):
        sd = dict([(k.split('.', 1)[1], v) for k, v in sd.items()])
    model.load_state_dict(sd)


def sdr_metrics(est, tgt, si=True, eps=EPS):
    """SI-SNR (si=True) or plain SNR (si=False) between two 1D arrays.
    The target must be zero-meaned; the estimate is projected onto it."""
    est = est - est.mean()
    tgt = tgt - tgt.mean()
    if si:
        est = (np.dot(est, tgt) / (np.dot(tgt, tgt) + eps)) * tgt
    noise = est - tgt
    return 10. * np.log10(((est ** 2).mean() + eps) /
                          ((noise ** 2).mean() + eps))


def pit_metrics_for_sample(estimates, references, mixture):
    """SNR / SNRi / SI-SNR / SI-SNRi for one sample under the best
    permutation, with the noisy mixture as the improvement baseline
    (falls back to the sum of the references when the raw noisy
    mixture is not provided)."""
    n_ref = len(references)
    mixture_used = mixture if mixture is not None else np.sum(references, 0)
    # improvement baselines per reference (mixture vs source)
    baseline_s = [sdr_metrics(mixture_used, ref, si=False)
                  for ref in references]
    baseline_si = [sdr_metrics(mixture_used, ref, si=True)
                   for ref in references]
    best = None
    for perm in itertools.permutations(range(len(estimates)), n_ref):
        total = 0.
        for i in range(n_ref):
            total += sdr_metrics(estimates[perm[i]],
                                 references[i], si=True)
        if best is None or total > best[0]:
            best = (total, perm)
    _, best_perm = best
    si_snrs = [sdr_metrics(estimates[best_perm[i]], references[i],
                           si=True) for i in range(n_ref)]
    snrs = [sdr_metrics(estimates[best_perm[i]], references[i],
                        si=False) for i in range(n_ref)]
    return {
        'SNR': float(np.mean(snrs)),
        'SNRi': float(np.mean(snrs) - np.mean(baseline_s)),
        'SI-SNR': float(np.mean(si_snrs)),
        'SI-SNRi': float(np.mean(si_snrs) - np.mean(baseline_si)),
    }


def main():
    args = build_arg_parser().parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available()
                             else 'cpu')

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    hparams = checkpoint['hparams']
    model = build_model(hparams)
    load_state_dict(model, checkpoint)
    model.to(device).eval()
    print('Loaded checkpoint: {} (epoch {}, tr_step {})'.format(
        args.checkpoint, checkpoint.get('epoch'),
        checkpoint.get('tr_step')))

    mix_dir = os.path.join(args.data_root, 'mix')
    mixture_paths = sorted(glob.glob(os.path.join(mix_dir, '*.wav')))
    if args.max_samples > 0:
        mixture_paths = mixture_paths[:args.max_samples]
    if not mixture_paths:
        raise IOError('No mixtures found under {}'.format(mix_dir))

    ckpt_stem = os.path.splitext(
        os.path.basename(args.checkpoint))[0]
    out_dir = args.out or os.path.join(
        args.data_root, 'estimates', ckpt_stem)
    os.makedirs(out_dir, exist_ok=True)

    with_metrics = args.with_metrics or args.metrics_csv is not None
    metric_rows = []

    for path in tqdm(mixture_paths, unit='file',
                     desc='Inferring ({})'.format(ckpt_stem)):
        sample_id = os.path.basename(path).rsplit('.', 1)[0]
        wav, fs = sf.read(path, dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(-1)
        input_wav = torch.tensor(
            wav[np.newaxis], dtype=torch.float32,
            device=device).unsqueeze(0)
        # match the runner's input normalization
        input_std = input_wav.std(-1, keepdim=True)
        input_wav = input_wav / (input_std + EPS)

        with torch.no_grad():
            estimates = model(input_wav)[0]  # [n_src, T]
        estimates = estimates.cpu().numpy()

        for i, est in enumerate(estimates, 1):
            out_path = os.path.join(out_dir,
                                    '{}_s{}.wav'.format(sample_id, i))
            sf.write(out_path, est, samplerate=fs)

        if with_metrics:
            references = []
            for i in range(1, 4):
                ref_path = os.path.join(
                    args.data_root, 's{}_reverb'.format(i),
                    '{}.wav'.format(sample_id))
                if os.path.lexists(ref_path):
                    references.append(sf.read(ref_path,
                                              dtype='float32')[0])
            if references:
                # estimates carry the scale of the normalized input:
                # bring them back to the file scale for metrics
                ests_denorm = [est * float(input_std.item())
                               for est in estimates]
                metrics = pit_metrics_for_sample(
                    ests_denorm, references, wav)
                metric_rows.append(dict(
                    [('sample_id', sample_id)] +
                    [(name, metrics[name]) for name in METRIC_NAMES]))
                tqdm.write(
                    '{}: SNR {:.2f} SNRi {:.2f} SI-SNR {:.2f} '
                    'SI-SNRi {:.2f}'.format(
                        sample_id, metrics['SNR'], metrics['SNRi'],
                        metrics['SI-SNR'], metrics['SI-SNRi']))

    if metric_rows:
        csv_path = args.metrics_csv or os.path.join(
            out_dir, 'metrics.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['sample_id'] + METRIC_NAMES)
            writer.writeheader()
            writer.writerows(metric_rows)
        print('Wrote metrics: {}'.format(csv_path))
        for name in METRIC_NAMES:
            values = np.array([row[name] for row in metric_rows])
            print('{:<8} mean {:>7.3f} median {:>7.3f}'.format(
                name, float(np.mean(values)),
                float(np.median(values))))

    print('Done: {} mixtures -> {} (3 wavs each)'.format(
        len(mixture_paths), out_dir))


if __name__ == '__main__':
    main()
