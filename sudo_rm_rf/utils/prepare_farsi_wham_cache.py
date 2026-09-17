"""!
@brief Offline preprocessing to build local 16 kHz wav caches for the
FARSI_WHAM dataset.

Speech cache is built from the HuggingFace dataset
Thomcles/YodaLingua-Farsi (MP3, 24 kHz, mono, with speaker_id metadata).
Each clip is resampled to 16 kHz float32 and written to:

    <speech_out>/speech/<speaker_id>/<idx>.wav

Noise cache is built from the HuggingFace dataset
montaseri/wham-noise-subset-sharded. Attention: the per-row `label`
column (0=cv, 1=tr, 2=tt) is populated only in its `eval` (all cv) and
`test` (all tt) splits and is null in the HF `train` split, which only
contains original WHAM `tr` noise. So the WHAM split is derived from the
row label when present, otherwise from the HF split name:
train -> tr, eval -> cv, test -> tt. Clips are resampled to 16 kHz and
written to:

    <noise_out>/<tr|cv|tt>/<idx>.wav

Both caches include an index JSON (paths + durations) that
sudo_rm_rf.dnn.dataset_loader.farsi_wham reads at init, so the loader
never needs the HF datasets library.

@author Hamidreza (adapted from repo conventions)
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
from tqdm import tqdm


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Build 16kHz wav caches for FARSI_WHAM dataset.')
    parser.add_argument('--speech_out', type=str,
                        default='/data/farsi_wham',
                        help='Output root dir for the speech cache.')
    parser.add_argument('--noise_out', type=str,
                        default='/data/wham_noise_16k',
                        help='Output root dir for the WHAM noise cache.')
    parser.add_argument('--fs', type=int, default=16000,
                        help='Target sampling rate (default 16000).')
    parser.add_argument('--min_speech_sec', type=float, default=2.0,
                        help='Discard speech clips shorter than this.')
    parser.add_argument('--n_jobs', type=int, default=4)
    parser.add_argument('--chunk_size', type=int, default=128,
                        help='Number of clips decoded in flight at any '
                             'time; bounds the decode pipeline RAM usage '
                             '(the wav decoder pool holds one chunk of '
                             'result wavs max).')
    parser.add_argument('--speech_only', action='store_true')
    parser.add_argument('--noise_only', action='store_true')
    return parser


def resample_to(x, orig_fs, target_fs):
    if orig_fs == target_fs:
        return x
    from fractions import Fraction
    g = Fraction(target_fs, orig_fs)
    return resample_poly(x, g.numerator, g.denominator).astype(np.float32)


def write_wav_pair(task):
    out_path, wav, fs = task
    sf.write(out_path, wav, fs)
    return out_path


def iter_chunks(iterable, chunk_size):
    """Yield lists of at most chunk_size items from a (lazy) iterable."""
    iterator = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(chunk_size):
                chunk.append(next(iterator))
        except StopIteration:
            pass
        if chunk:
            yield chunk
        if len(chunk) < chunk_size:
            break


def prepare_speech(cache_root, fs, min_speech_sec, n_jobs, chunk_size=128,
                   cache_dir=None):
    import datasets
    speech_dir = os.path.join(cache_root, 'speech')
    if os.path.exists(os.path.join(cache_root, 'speech_index.json')):
        print('speech_index.json already exists, skipping speech cache.')
        return

    # The HF download cache can be reused/pointed to by cache_dir.
    ds_kwargs = dict()
    if cache_dir is not None:
        ds_kwargs['cache_dir'] = cache_dir
    ds = datasets.load_dataset('Thomcles/YodaLingua-Farsi',
                               split='train', **ds_kwargs)

    os.makedirs(speech_dir, exist_ok=True)
    index = {}
    n_total = len(ds)
    discarded = 0
    # speech cache: filenames are assigned in the main thread so that
    # idx ids cannot collide. The dataset is processed in bounded chunks;
    # a plain unbounded executor.map submits all 23k examples at once
    # and keeps every decoded waveform alive until the iterator gets to
    # it, OOM-ing machines with little RAM.
    with ThreadPoolExecutor(max_workers=n_jobs) as pool_d:
        def decode(example):
            audio = example['mp3']
            wav, fs_orig = audio['array'], audio['sampling_rate']
            if wav.ndim > 1:
                wav = wav.mean(-1)
            wav = resample_to(wav.astype(np.float32), fs_orig, fs)
            return example['speaker_id'], wav

        pbar = tqdm(total=n_total, unit='clip',
                    desc='Caching speech (resampling to {}Hz)'.format(fs))
        for chunk in iter_chunks(ds, chunk_size):
            for speaker_id, wav in pool_d.map(decode, chunk):
                dur = len(wav) / float(fs)
                if dur < min_speech_sec:
                    discarded += 1
                    pbar.update(1)
                    continue
                speaker_dir = os.path.join(speech_dir, speaker_id)
                os.makedirs(speaker_dir, exist_ok=True)
                idx = len(index.get(speaker_id, []))
                out_path = os.path.join(speaker_dir, '{}.wav'.format(
                    str(idx).zfill(6)))
                write_wav_pair((out_path, wav, fs))
                paths = index.setdefault(speaker_id, [])
                paths.append(os.path.abspath(out_path))
                pbar.update(1)
                pbar.set_postfix(speakers=len(index),
                                 discarded=discarded)
        pbar.close()

    for speaker_id in index:
        index[speaker_id].sort()

    with open(os.path.join(cache_root, 'speech_index.json'), 'w') as f:
        json.dump({'fs': fs, 'speakers': index}, f)
    print('Speech cache done: {} speakers, {} clips.'.format(
        len(index), sum(len(v) for v in index.values())))


def prepare_noise(cache_root, fs, n_jobs, chunk_size=128, cache_dir=None):
    import datasets
    if os.path.exists(os.path.join(cache_root, 'noise_index.json')):
        print('noise_index.json already exists, skipping noise cache.')
        return

    index = {k: [] for k in ('tr', 'cv', 'tt')}
    # Filenames are assigned in the main thread (decode runs on workers,
    # so counting/starting inside decode could collide on the same idx
    # filename). Chunked processing bounds the amount of decoded
    # waveforms alive in RAM (see prepare_speech comment).
    for hf_split in ('train', 'eval', 'test'):
        ds_kwargs = dict()
        if cache_dir is not None:
            ds_kwargs['cache_dir'] = cache_dir
        # Keep original WHAM splits: the per-row `label` column
        # (0=cv, 1=tr, 2=tt) exists in eval \& test but is null in
        # the HF `train` split (which is all original WHAM `tr`
        # noise anyway). Fall back to the HF split name when label
        # is missing. HF split -> WHAM split:
        #   train -> tr, eval -> cv, test -> tt
        ds = datasets.load_dataset(
            'montaseri/wham-noise-subset-sharded',
            split=hf_split, **ds_kwargs)
        label_map = {0: 'cv', 1: 'tr', 2: 'tt'}

        with ThreadPoolExecutor(max_workers=n_jobs) as pool_d:
            def decode(example, hf_split=hf_split):
                audio = example['audio']
                wav, fs_orig = audio['array'], audio['sampling_rate']
                if wav.ndim > 1:
                    wav = wav.mean(-1)
                wav = resample_to(wav.astype(np.float32), fs_orig, fs)
                if np.mean(wav ** 2) < 1e-10:
                    return None
                label = example.get('label', None)
                if label is not None:
                    split = label_map[int(label)]
                else:
                    split = {'train': 'tr', 'eval': 'cv',
                             'test': 'tt'}[hf_split]
                return split, wav

            pbar = tqdm(total=len(ds), unit='clip',
                        desc='Caching noise [{}] (resampling to {}Hz)'
                             ''.format(hf_split, fs))
            for chunk in iter_chunks(ds, chunk_size):
                for result in pool_d.map(decode, chunk):
                    if result is None:
                        pbar.update(1)
                        continue
                    split, wav = result
                    split_dir = os.path.join(cache_root, split)
                    os.makedirs(split_dir, exist_ok=True)
                    idx = len(index[split])
                    out_path = os.path.join(split_dir, '{}.wav'.format(
                        str(idx).zfill(6)))
                    write_wav_pair((out_path, wav, fs))
                    index[split].append(os.path.abspath(out_path))
                    pbar.update(1)
                    pbar.set_postfix(tr=len(index['tr']),
                                     cv=len(index['cv']),
                                     tt=len(index['tt']))
            pbar.close()

    with open(os.path.join(cache_root, 'noise_index.json'), 'w') as f:
        json.dump({'fs': fs, 'splits': index}, f)
    print('Noise cache done: ' + ', '.join(
        '{}: {}'.format(k, len(v)) for k, v in index.items()))


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    if not args.noise_only:
        prepare_speech(args.speech_out, args.fs, args.min_speech_sec,
                       args.n_jobs, chunk_size=args.chunk_size)
    if not args.speech_only:
        prepare_noise(args.noise_out, args.fs, args.n_jobs,
                      chunk_size=args.chunk_size)
