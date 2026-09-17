"""!
@brief Pytorch dataloader for the FARSI_WHAM dataset: variable 1-3 Farsi
speakers (YodaLingua) mixed with optional WHAM noise.

Each sample is generated as:
    x = s1 + s2 + s3 + n   (noise is interference, not a target)

and returns the noisy mixture plus three speech targets with zero rows
for missing speakers:

    {
        'mixture':   [1, T],
        'targets':   [3, T],  active sources first, zeros afterwards
        'n_speakers': int,
    }

Training mixtures are generated on the fly with random speaker IDs,
temporal placement, relative gains and WHAM SNR. Validation/test mixtures
use fixed precomputed recipes so they are reproducible.

Requires the caches built by sudo_rm_rf.utils.prepare_farsi_wham_cache
(speech_index.json / noise_index.json).

@author Hamidreza (adapted from repo conventions)
"""

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.append(root_dir)

import json
import numpy as np
import torch
import soundfile as sf

import sudo_rm_rf.dnn.dataset_loader.abstract_dataset as abstract_dataset

EPS = 1e-8

SPEAKER_COUNT_PROBS = [0.20, 0.45, 0.35]  # 1, 2, 3 speakers
NOISE_PROB = 0.75
SNR_RANGE = (-5., 20.)
GAIN_RANGE = (0.5, 1.5)


class Dataset(torch.utils.data.Dataset, abstract_dataset.Dataset):
    """ Dataset class for the FARSI_WHAM variable-speaker separation task.

    Example of kwargs:
        root_dirpath='/data/farsi_wham', noise_dirpath='/data/wham_noise_16k',
        split='train', sample_rate=16000, timelength=4.0,
        n_samples=0, min_num_sources=1, max_num_sources=3
    """

    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.kwargs = kwargs

        self.task = self.get_arg_and_check_validness(
            'task', known_type=str, choices=['sep_noisy'])

        self.max_num_sources = self.get_arg_and_check_validness(
            'max_num_sources', known_type=int,
            extra_lambda_checks=[lambda x: (x >= 1) and (x <= 3)])
        self.min_num_sources = self.get_arg_and_check_validness(
            'min_num_sources', known_type=int,
            extra_lambda_checks=[
                lambda x: (x >= 1) and (x <= self.max_num_sources)])

        self.sample_rate = self.get_arg_and_check_validness(
            'sample_rate', known_type=int, choices=[16000])
        self.timelength = self.get_arg_and_check_validness(
            'timelength', known_type=float)
        self.time_samples = int(self.sample_rate * self.timelength)

        self.split = self.get_arg_and_check_validness(
            'split', known_type=str, choices=['train', 'validation', 'eval'])
        self.augment = self.get_arg_and_check_validness(
            'augment', known_type=bool)

        self.root_path = self.get_arg_and_check_validness(
            'root_dirpath', known_type=str,
            extra_lambda_checks=[lambda y: os.path.lexists(y)])
        self.noise_root_path = self.get_arg_and_check_validness(
            'noise_dirpath', known_type=str,
            extra_lambda_checks=[lambda y: os.path.lexists(y)])

        # ---- read speech index ----
        with open(os.path.join(self.root_path, 'speech_index.json'),
                  'r') as f:
            speech_index = json.load(f)
        assert speech_index['fs'] == self.sample_rate, (
            'Speech cache was built at a different sampling rate.')
        speakers = speech_index['speakers']
        self.speaker_ids = sorted(speakers.keys())
        self.speech_files = dict(
            [(spk, sorted(speakers[spk])) for spk in self.speaker_ids])

        # ---- read noise index ----
        noise_split = {'train': 'tr', 'validation': 'cv', 'eval': 'tt'}[
            self.split]
        with open(os.path.join(self.noise_root_path, 'noise_index.json'),
                  'r') as f:
            noise_index = json.load(f)
        assert noise_index['fs'] == self.sample_rate, (
            'Noise cache was built at a different sampling rate.')
        self.noise_files = noise_index['splits'][noise_split]

        # ---- fixed mixing recipes for validation/test ----
        self.n_samples = self.get_arg_and_check_validness(
            'n_samples', known_type=int, extra_lambda_checks=[
                lambda x: x >= 0])

        self.epoch = 0
        if not self.augment:
            self.rng = np.random.RandomState(42)
            self.n_samples = 10000 if self.n_samples == 0 else self.n_samples
            self.mixtures_recipes = [
                self.get_mixing_recipe(self.rng, i)
                for i in range(self.n_samples)]
        else:
            self.n_samples = self.n_samples if self.n_samples > 0 \
                else 10000000
            self.mixtures_recipes = [None] * self.n_samples

        self.actual_n_samples = len(self.mixtures_recipes)

    def set_epoch(self, epoch):
        """Set the epoch so that __getitem__ can derive a per-idx seeded
        RNG (DataLoader worker processes never see in-place mutations of
        dataset attributes made on the fly)."""
        self.epoch = epoch

    def _augment_rng(self, idx):
        seed = (self.epoch * 7919 + idx) % (2 ** 31)
        return np.random.RandomState(seed)

    def get_mixing_recipe(self, rng, idx):
        # speaker count: 1-3 with prescribed probabilities, resampled
        # until the requested min/max number of sources is satisfied
        n_speakers = None
        while True:
            candidate = int(rng.choice([1, 2, 3],
                                       p=SPEAKER_COUNT_PROBS))
            if self.min_num_sources <= candidate <= self.max_num_sources:
                n_speakers = candidate
                break
        speaker_ids = rng.choice(self.speaker_ids, size=n_speakers,
                                 replace=False).tolist()
        speech_paths = [self.speech_files[spk][
                            rng.randint(len(self.speech_files[spk]))]
                        for spk in speaker_ids]
        gains = rng.uniform(GAIN_RANGE[0], GAIN_RANGE[1], n_speakers
                            ).tolist()
        with_noise = bool(rng.rand() < NOISE_PROB) if len(
            self.noise_files) else False
        noise_path = None
        snr = None
        if with_noise:
            noise_path = self.noise_files[rng.randint(len(self.noise_files))]
            snr = rng.uniform(SNR_RANGE[0], SNR_RANGE[1])
        recipe = {
            'n_speakers': n_speakers,
            'speech_paths': speech_paths,
            'gains': gains,
            'noise_path': noise_path,
            'snr': snr,
        }
        return recipe

    def load_and_crop(self, path, rng):
        """Crop/pad a wav and place it at a random offset inside the
        window. Long clips are cropped to a random sublength first, so
        multiple speakers naturally get partial overlap inside the same
        window."""
        wav, fs = sf.read(path, dtype='float32', always_2d=True)
        assert fs == self.sample_rate, (
            'File: {} had unexpected fs: {}'.format(path, fs))
        wav = wav.mean(-1)
        if wav.shape[0] > self.time_samples:
            keep = int(rng.randint(self.time_samples // 2,
                                   self.time_samples + 1))
            offset = int(rng.randint(0, wav.shape[0] - keep))
            wav = wav[offset:offset + keep]
        out = np.zeros(self.time_samples, dtype=np.float32)
        start = int(rng.randint(0, self.time_samples - wav.shape[0] + 1))
        out[start:start + wav.shape[0]] = wav
        # for a fully occupied window start will end up 0
        return out

    def generate_mix(self, recipe, rng):
        targets = np.zeros((self.max_num_sources, self.time_samples),
                           dtype=np.float32)
        for i, (path, gain) in enumerate(zip(recipe['speech_paths'],
                                             recipe['gains'])):
            wav = self.load_and_crop(path, rng)
            targets[i] = wav * gain

        speech_mix = targets[:recipe['n_speakers']].sum(0)

        mixture = speech_mix
        if recipe['noise_path'] is not None and recipe['snr'] is not None:
            noise = self.load_and_crop(recipe['noise_path'], rng)
            # skip silent noise crops
            noise_power = np.mean(noise ** 2)
            if noise_power > EPS:
                speech_power = np.mean(speech_mix ** 2)
                target_noise_power = speech_power * \
                    10 ** (-recipe['snr'] / 10.)
                noise = noise * np.sqrt(target_noise_power /
                                        (noise_power + EPS))
                mixture = mixture + noise

        # unused targets stay zero
        # joint normalization: keep relative levels meaningful
        scale = mixture.std() + EPS
        mixture = mixture / scale
        targets = targets / scale
        return mixture, targets

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.augment:
            rng = self._augment_rng(idx)
            recipe = self.get_mixing_recipe(rng, idx)
        else:
            rng = np.random.RandomState(idx)
            recipe = self.mixtures_recipes[idx]
        mixture, targets = self.generate_mix(recipe, rng)
        return {
            'mixture': torch.tensor(
                mixture[np.newaxis], dtype=torch.float32),
            'targets': torch.tensor(targets, dtype=torch.float32),
            'n_speakers': recipe['n_speakers'],
        }

    def get_generator(self, batch_size=4, shuffle=True, num_workers=4):
        generator_params = {'batch_size': batch_size,
                            'shuffle': shuffle,
                            'num_workers': num_workers,
                            'drop_last': True}
        return torch.utils.data.DataLoader(
            self, pin_memory=True, **generator_params)


def test_generator():
    batch_size = 3
    sample_rate = 16000
    timelength = 4.0
    time_samples = int(sample_rate * timelength)
    max_num_sources = 3
    min_num_sources = 1
    fake_root = '/tmp/opencode/farsi_wham_test/speech'
    fake_noise = '/tmp/opencode/farsi_wham_test/noise'
    build_fake_caches(fake_root, fake_noise, sample_rate)
    data_loader = Dataset(
        root_dirpath=fake_root, noise_dirpath=fake_noise,
        task='sep_noisy', max_num_sources=max_num_sources,
        min_num_sources=min_num_sources, split='train',
        sample_rate=sample_rate, timelength=timelength,
        n_samples=12, augment=True)
    generator = data_loader.get_generator(batch_size=batch_size,
                                          num_workers=1)
    for data in generator:
        assert data['mixture'].shape == (batch_size, 1, time_samples)
        assert data['targets'].shape == (batch_size, max_num_sources,
                                         time_samples)

    # reproducible validation style examples
    data_loader = Dataset(
        root_dirpath=fake_root, noise_dirpath=fake_noise,
        task='sep_noisy', max_num_sources=max_num_sources,
        min_num_sources=min_num_sources, split='validation',
        sample_rate=sample_rate, timelength=timelength,
        n_samples=6, augment=False)
    generator = data_loader.get_generator(batch_size=1, num_workers=1,
                                          shuffle=False)
    results = [tuple(data['mixture'][0, 0, :5].tolist())
               for data in generator]
    assert len(results) == 6
    print('Loader smoke test passed.')


def build_fake_caches(speech_root, noise_root, fs):
    if not os.path.lexists(os.path.join(noise_root, 'tr')):
        n_speakers = 4
        speakers = {}
        import os.path as osp
        for spk in range(n_speakers):
            spk_id = 'spk{}'.format(spk)
            spk_dir = os.path.join(speech_root, spk_id)
            os.makedirs(spk_dir, exist_ok=True)
            files = []
            for k in range(3):
                path = os.path.join(spk_dir, '{}.wav'.format(k))
                lens = [int(fs * 1.), int(fs * 3.), int(fs * 6.)][k]
                np.random.seed(spk * 10 + k)
                sf.write(path, np.random.randn(lens).astype(
                    np.float32) * 0.1, fs)
                files.append(os.path.abspath(path))
            speakers[spk_id] = files
        with open(os.path.join(speech_root, 'speech_index.json'),
                  'w') as f:
            json.dump({'fs': fs, 'speakers': speakers}, f)
    if not os.path.lexists(os.path.join(noise_root, 'noise_index.json')):
        os.makedirs(noise_root, exist_ok=True)
        index = {'tr': [], 'cv': [], 'tt': []}
        np.random.seed(0)
        for i in range(6):
            path = os.path.join(noise_root, 'tr', '{}.wav'.format(
                str(i).zfill(6)))
            os.makedirs(os.path.join(noise_root, 'tr'), exist_ok=True)
            sf.write(path, (np.random.randn(int(fs * 8.)) * 0.05
                            ).astype(np.float32), fs)
            index['tr'].append(path)
        with open(os.path.join(noise_root, 'noise_index.json'),
                  'w') as f:
            json.dump({'fs': fs, 'splits': index}, f)


if __name__ == '__main__':
    test_generator()
