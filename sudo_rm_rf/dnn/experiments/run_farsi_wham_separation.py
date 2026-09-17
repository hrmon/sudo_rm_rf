"""!
@brief Training/validation of GroupComm SuDoRmRf for up-to-3-speaker
Farsi speech separation with optional WHAM noise interference.

Based on run_fuss_separation.py but adapted for the FARSI_WHAM dataset:
- The dataset loader returns dict batches with the noisy mixture and
  three speech targets (zero rows for missing speakers).
- No online source reshuffling/level augmentation (the loader controls
  speaker identity, count, overlap, gains and WHAM SNR).
- No mixture consistency projection (noise must not be forced back into
  the speech estimates).
- PIT loss on active sources + explicit inactive-output energy penalty.
- Validation uses n_estimated_sources = max_num_sources for all cases;
  the active speaker may appear in any output channel.

@author Hamidreza (adapted from repo conventions)
"""

import os
import sys
current_dir = os.path.dirname(os.path.abspath('__file__'))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
sys.path.append(root_dir)

import torch
from tqdm import tqdm
from pprint import pprint
import sudo_rm_rf.dnn.experiments.utils.improved_cmd_args_parser_v2 \
    as parser
import sudo_rm_rf.dnn.experiments.utils.dataset_setup as dataset_setup
import sudo_rm_rf.dnn.losses.sisdr as sisdr_lib
import sudo_rm_rf.dnn.losses.variable_speaker_snr as vsnr_lib
import sudo_rm_rf.dnn.models.improved_sudormrf as improved_sudormrf
import sudo_rm_rf.dnn.models.groupcomm_sudormrf_v2 as sudormrf_gc_v2
import sudo_rm_rf.dnn.models.causal_improved_sudormrf_v3 as \
    causal_improved_sudormrf
import sudo_rm_rf.dnn.models.sudormrf as initial_sudormrf
import sudo_rm_rf.dnn.utils.cometml_loss_report as cometml_report
import sudo_rm_rf.dnn.utils.cometml_log_audio as cometml_audio_logger
from sudo_rm_rf.dnn.utils.local_experiment import LocalExperiment

args = parser.get_args()
hparams = vars(args)
assert hparams['n_channels'] == 1, (
    'Mono source separation is available for now')

if hparams["checkpoints_path"] is not None:
    if hparams["save_checkpoint_every"] <= 0:
        raise ValueError("Expected a value greater than 0 for checkpoint "
                         "storing.")
    if not os.path.exists(hparams["checkpoints_path"]):
        os.makedirs(hparams["checkpoints_path"])

audio_loggers = dict(
    [(n_src,
      cometml_audio_logger.AudioLogger(fs=hparams["fs"],
                                       bs=1,
                                       n_sources=n_src))
     for n_src in range(1, hparams['max_num_sources'] + 1)])

generators = {}

# Generate the training mixtures online (random speakers, placement,
# gains, WHAM noise) from the FARSI_WHAM loader.
train_loader = dataset_setup.create_loader_for_simple_dataset(
    dataset_name='FARSI_WHAM',
    separation_task='sep_noisy',
    data_split='train', sample_rate=hparams['fs'],
    n_channels=hparams['n_channels'], min_or_max='max',
    zero_pad=hparams['zero_pad_audio'],
    timelegth=hparams['audio_timelength'],
    normalize_audio=hparams['normalize_audio'],
    n_samples=hparams['n_train'],
    min_num_sources=hparams['min_num_sources'],
    max_num_sources=hparams['max_num_sources'])
generators['train'] = train_loader.get_generator(
    batch_size=hparams['batch_size'], num_workers=hparams['n_jobs'])

# Hardcode separate val/test generators per one of the number of sources
for n_src in range(hparams['min_num_sources'],
                   hparams['max_num_sources'] + 1):
    for split_name in ['val', 'test']:
        loader = dataset_setup.create_loader_for_simple_dataset(
            dataset_name='FARSI_WHAM',
            separation_task='sep_noisy',
            data_split=split_name, sample_rate=hparams['fs'],
            n_channels=hparams['n_channels'], min_or_max='max',
            zero_pad=hparams['zero_pad_audio'],
            timelegth=hparams['audio_timelength'],
            normalize_audio=hparams['normalize_audio'],
            n_samples=0, min_num_sources=n_src, max_num_sources=n_src)

        gen_name = '{}_{}_srcs'.format(split_name, n_src)
        generators[gen_name] = loader.get_generator(
            batch_size=hparams['batch_size'], num_workers=hparams['n_jobs'])

experiment = LocalExperiment(
    logs_path=(hparams['experiment_logs_path']
               if hparams['experiment_logs_path'] is not None
               else 'offline_exps'),
    project_name=hparams['project_name'])
experiment.log_parameters(hparams)
experiment_name = '_'.join(hparams['cometml_tags'])
for tag in hparams['cometml_tags']:
    experiment.add_tag(tag)
if hparams['experiment_name'] is not None:
    experiment.set_name(hparams['experiment_name'])
else:
    experiment.set_name(experiment_name)

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
    [cad for cad in hparams['cuda_available_devices']])

# train loss: PIT-SNR on active sources + inactive outputs penalty
back_loss_tr_loss_name = 'tr_back_loss_VAR_SPEAKER_SNR'
back_loss_tr_loss = vsnr_lib.VariableSpeakerSNRwithZeroRefs(
    n_sources=hparams["max_num_sources"],
    zero_mean=False,
    backward_loss=True,
    inactive_threshold_dB=20.)

val_losses = {}
all_losses = []
for val_set in generators:
    if val_set == 'train':
        continue
    n_actual_sources = int(val_set.split('_')[1])
    # The active speaker may appear in any of the output channels:
    # always evaluate with the max number of estimated sources so that
    # permutations of the estimated channels are considered.
    n_estimated_sources = hparams['max_num_sources']
    improvement = False
    metric_name = 'SISDR'
    val_losses[val_set] = {}
    all_losses.append(val_set + '_{}'.format(metric_name))
    val_losses[val_set][val_set + '_{}'.format(metric_name)] = \
        sisdr_lib.StabilizedPermInvSISDRMetric(
            zero_mean=True,
            single_source=False,
            n_estimated_sources=n_estimated_sources,
            n_actual_sources=n_actual_sources,
            backward_loss=False,
            improvement=False,
            return_individual_results=True)
all_losses.append(back_loss_tr_loss_name)

if hparams['model_type'] == 'relu':
    model = improved_sudormrf.SuDORMRF(out_channels=hparams['out_channels'],
                                       in_channels=hparams['in_channels'],
                                       num_blocks=hparams['num_blocks'],
                                       upsampling_depth=hparams['upsampling_depth'],
                                       enc_kernel_size=hparams['enc_kernel_size'],
                                       enc_num_basis=hparams['enc_num_basis'],
                                       num_sources=hparams['max_num_sources'])
elif hparams['model_type'] == 'causal':
    model = causal_improved_sudormrf.CausalSuDORMRF(
        in_audio_channels=1,
        out_channels=hparams['out_channels'],
        in_channels=hparams['in_channels'],
        num_blocks=hparams['num_blocks'],
        upsampling_depth=hparams['upsampling_depth'],
        enc_kernel_size=hparams['enc_kernel_size'],
        enc_num_basis=hparams['enc_num_basis'],
        num_sources=hparams['max_num_sources'])
elif hparams['model_type'] == 'softmax':
    model = initial_sudormrf.SuDORMRF(out_channels=hparams['out_channels'],
                                      in_channels=hparams['in_channels'],
                                      num_blocks=hparams['num_blocks'],
                                      upsampling_depth=hparams['upsampling_depth'],
                                      enc_kernel_size=hparams['enc_kernel_size'],
                                      enc_num_basis=hparams['enc_num_basis'],
                                      num_sources=hparams['max_num_sources'])
elif hparams['model_type'] == 'groupcomm_v2':
    model = sudormrf_gc_v2.GroupCommSudoRmRf(
        in_audio_channels=hparams['n_channels'],
        out_channels=hparams['out_channels'],
        in_channels=hparams['in_channels'],
        num_blocks=hparams['num_blocks'],
        upsampling_depth=hparams['upsampling_depth'],
        enc_kernel_size=hparams['enc_kernel_size'],
        enc_num_basis=hparams['enc_num_basis'],
        num_sources=hparams['max_num_sources'],
        group_size=hparams['group_size'])
else:
    raise ValueError('Invalid model: {}.'.format(hparams['model_type']))

numparams = 0
for f in model.parameters():
    if f.requires_grad:
        numparams += f.numel()
experiment.log_parameter('Parameters', numparams)
print('Trainable Parameters: {}'.format(numparams))

model = torch.nn.DataParallel(model).cuda()
opt = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])

model_shaping_hparams = ['model_type', 'out_channels', 'in_channels',
                         'num_blocks', 'upsampling_depth', 'enc_kernel_size',
                         'enc_num_basis', 'max_num_sources']

start_epoch = 0
if hparams['resume_from_checkpoint'] is not None:
    resume_path = hparams['resume_from_checkpoint']
    if resume_path == 'latest':
        if hparams['checkpoints_path'] is None:
            raise ValueError("'latest' resume requires --checkpoints_path "
                             "to be set.")
        resume_path = os.path.join(hparams['checkpoints_path'],
                                   'latest_checkpoint.pt')
    if not os.path.lexists(resume_path):
        raise ValueError('Requested resume checkpoint: {} but it was '
                         'not found.'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cuda')
    for hp_name in model_shaping_hparams:
        if hparams[hp_name] != checkpoint['hparams'][hp_name]:
            raise ValueError('Resumed checkpoint was created with different '
                             'hparams. For {} the requested value was {} '
                             'while the stored one was {}.'.format(
                                 hp_name, hparams[hp_name],
                                 checkpoint['hparams'][hp_name]))
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    tr_step = checkpoint['tr_step']
    val_step = checkpoint['val_step']
    start_epoch = checkpoint['epoch'] + 1
    print('Resumed training from checkpoint: {} for epoch {} '
          'onwards.'.format(resume_path, start_epoch))


def scale_normalize(tensor_wav, eps=1e-8, std=None):
    """Normalize by std only (optionally with an externally computed
    std), preserving relative levels across channels."""
    if std is None:
        std = tensor_wav.std(-1, keepdim=True)
    return tensor_wav / (std + eps)


if hparams['resume_from_checkpoint'] is None:
    tr_step = 0
    val_step = 0
prev_epoch_val_loss = 0.
for i in range(start_epoch, hparams['n_epochs']):
    res_dic = {}
    for loss_name in all_losses:
        res_dic[loss_name] = {'mean': 0., 'std': 0., 'median': 0., 'acc': []}
    print("FARSI_WHAM SuDo-RM-RF: {} - {} || Epoch: {}/{}".format(
        experiment.get_key(), experiment.get_tags(), i+1, hparams['n_epochs']))
    model.train()

    sum_loss = 0.
    train_tqdm_gen = tqdm(generators['train'], desc='Training')
    # Re-seed augmentation for each epoch
    underlying_dataset = train_tqdm_gen.dataset
    if hasattr(underlying_dataset, 'set_epoch'):
        underlying_dataset.set_epoch(i)
    for cnt, data in enumerate(train_tqdm_gen):
        opt.zero_grad()

        input_mixture = data['mixture'].cuda()
        clean_wavs = data['targets'].cuda()
        n_speakers = data['n_speakers']

        # joint normalization of input and references
        input_mix_std = input_mixture.std(-1, keepdim=True)
        input_mixture = scale_normalize(input_mixture, std=input_mix_std)
        clean_wavs = scale_normalize(clean_wavs, std=input_mix_std)

        rec_sources_wavs = model(input_mixture)

        # NO mixture consistency: the model is not supposed to
        # reconstruct the WHAM noise.

        l = back_loss_tr_loss(rec_sources_wavs,
                              clean_wavs,
                              input_mixture)
        l.backward()

        if hparams['clip_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           hparams['clip_grad_norm'])

        opt.step()
        sum_loss += l.detach().item()
        train_tqdm_gen.set_description(
            "Training, Running Avg Loss: {}".format(sum_loss / (cnt + 1)))

    if hparams['patience'] > 0:
        if tr_step % hparams['patience'] == 0:
            new_lr = (hparams['learning_rate']
                      / (hparams['divide_lr_by'] ** (tr_step // hparams['patience'])))
            print('Reducing Learning rate to: {}'.format(new_lr))
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr
    tr_step += 1

    for val_set in [x for x in generators if not x == 'train']:
        n_actual_sources = int(val_set.split('_')[1])
        model.eval()
        with torch.no_grad():
            for data in tqdm(generators[val_set],
                             desc='Validation on {}'.format(val_set)):
                input_mixture = data['mixture'].cuda()
                # only the actually active targets should be evaluated
                clean_wavs = data['targets'][:, :n_actual_sources].cuda()
                # joint normalization of input and references
                input_mix_std = input_mixture.std(-1, keepdim=True)
                input_mixture = scale_normalize(input_mixture,
                                                std=input_mix_std)
                clean_wavs = scale_normalize(clean_wavs,
                                             std=input_mix_std)
                rec_sources_wavs = model(input_mixture)

                for loss_name, loss_func in val_losses[val_set].items():
                    l, best_perm = loss_func(
                        rec_sources_wavs,
                        clean_wavs,
                        return_best_permutation=True)
                    res_dic[loss_name]['acc'] += l.tolist()

        audio_loggers[n_actual_sources].log_batch(
            rec_sources_wavs[:, best_perm.long().cuda()][0, 0].unsqueeze(0),
            clean_wavs[0].unsqueeze(0),
            input_mixture[0].unsqueeze(0),
            experiment, step=val_step, tag=val_set)

    val_step += 1

    res_dic = cometml_report.report_losses_mean_and_std(res_dic,
                                                        experiment,
                                                        tr_step,
                                                        val_step)

    for loss_name in res_dic:
        res_dic[loss_name]['acc'] = []
    pprint(res_dic)

    if hparams["checkpoints_path"] is not None:
        checkpoint = {
            'epoch': i,
            'tr_step': tr_step,
            'val_step': val_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'hparams': {name: hparams[name]
                        for name in model_shaping_hparams},
        }
        latest_path = os.path.join(hparams['checkpoints_path'],
                                   'latest_checkpoint.pt')
        tmp_path = latest_path + '.tmp'
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, latest_path)
        if tr_step % hparams["save_checkpoint_every"] == 0:
            torch.save(
                checkpoint,
                os.path.join(
                    hparams["checkpoints_path"],
                    "farsi_wham_sudo_epoch_{}".format(tr_step)),
            )
        print('Saved checkpoint at epoch: {} || tr_step: {}'.format(
            i, tr_step))
