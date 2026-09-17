"""!
@brief Local experiment tracking to replace comet_ml cloud logging.
       Metrics are appended to a CSV, hyperparams to a JSON file and
       audio samples are written as wavs, all under a per-experiment
       directory.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import os
import json
import contextlib
from time import time, asctime
from uuid import uuid4

import numpy as np
from scipy.io.wavfile import write as wavwrite


class LocalExperiment(object):
    """ Duck-typed substitute for a comet_ml Experiment.

    Exposes exactly the subset of the comet API used by the sudo_rm_rf
    experiment scripts:
        log_parameters, log_parameter, log_metric, log_audio,
        set_name, add_tag, get_key, get_tags,
        train() and validate() as context managers.
    """

    def __init__(self, logs_path='offline_exps',
                 project_name='local_experiment',
                 tags=None):
        self.tags = tags if tags is not None else []
        self.name = '_'.join(self.tags)
        self.key = asctime() + '_' + str(uuid4())[:8]
        self.experiment_dirpath = os.path.join(
            logs_path, project_name, self.key)
        if not os.path.exists(self.experiment_dirpath):
            os.makedirs(self.experiment_dirpath)
        self.audio_dirpath = os.path.join(self.experiment_dirpath, 'audio')
        if not os.path.exists(self.audio_dirpath):
            os.makedirs(self.audio_dirpath)
        self.metrics_path = os.path.join(
            self.experiment_dirpath, 'metrics.csv')
        self.hparams = {}
        self.info = dict(
            key=self.key,
            name=self.name,
            tags=self.tags,
            project_name=project_name,
            start_datetime=self.key,
            parameters_count=0)
        self._write_info()

    def _write_info(self):
        with open(os.path.join(self.experiment_dirpath,
                               'experiment.json'), 'w') as f:
            json.dump(self.info, f, indent=2)

    def set_name(self, name):
        self.name = name
        self.info['name'] = name
        self._write_info()

    def add_tag(self, tag):
        self.tags.append(tag)
        self.info['tags'] = self.tags
        self._write_info()

    def get_key(self):
        return self.key

    def get_tags(self):
        return self.tags

    def log_parameters(self, hparams):
        if not isinstance(hparams, dict):
            raise ValueError("log_parameters expects a dict, got {}".format(
                type(hparams)))
        self.hparams.update({str(k): v for k, v in hparams.items()})
        self._write_hparams()

    def log_parameter(self, name, value):
        self.hparams[str(name)] = value
        self._write_hparams()

    def _write_hparams(self):
        for key, val in self.hparams.items():
            try:
                json.dump({key: val}, open(os.devnull, 'w'))
            except TypeError:
                self.hparams[key] = str(val)
        with open(os.path.join(self.experiment_dirpath,
                               'hparams.json'), 'w') as f:
            json.dump(self.hparams, f, indent=2)

    def log_metric(self, name, value, step=None):
        with open(self.metrics_path, 'a') as f:
            f.write('"{}","{}","{}"\n'.format(name, value, step))

    def log_audio(self, wav, sample_rate, file_name, step=None, **kwargs):
        wav = np.array(wav, dtype=np.float32)
        max_val = np.max(np.abs(wav))
        if max_val > 0:
            wav = wav / max_val
        wav = wav.squeeze()
        fname = file_name.replace('/', '_')
        if fname.endswith('.wav'):
            fname = fname[:-4]
        if step is not None:
            fname += '_step_{}'.format(step)
        fname += '.wav'
        wavwrite(os.path.join(self.audio_dirpath, fname),
                 int(sample_rate), wav)

    @contextlib.contextmanager
    def train(self):
        yield self

    @contextlib.contextmanager
    def validate(self):
        yield self
