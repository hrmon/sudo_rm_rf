With **GroupComm v2**, I would keep the model architecture almost untouched and modify the **dataset, training loop, loss, and validation**. The key design is:

$$
x = s_1+s_2+s_3+n
$$

while the model predicts only:

$$
[\hat{s}_1,\hat{s}_2,\hat{s}_3]
$$

with zero targets for missing speakers. WHAM noise is interference, not a fourth output.

The repo already lets `GroupCommSudoRmRf` take `num_sources`, and `run_fuss_separation.py` passes `max_num_sources` directly to it. So `--max_num_sources 3` is enough for the architecture itself. 

### Changes I would make

1. **Create a new dataset loader**, e.g. `sudo_rm_rf/dnn/dataset_loader/farsi_wham.py`. Have each sample return the noisy mixture and three speech targets directly:

```python
mixture.shape  # [1, T]
speech_targets.shape  # [3, T]

# 1 speaker
targets = [s1, zero, zero]

# 2 speakers
targets = [s1, s2, zero]

# 3 speakers
targets = [s1, s2, s3]
```

For training, the loader should randomly choose 1–3 **different speaker IDs** from YodaLingua, randomly position/crop them inside the segment, apply relative gains, and optionally add a WHAM noise segment at a random SNR. Validation/test mixtures should instead be fixed/reproducible. The existing FUSS loader already illustrates the zero-padding pattern: it allocates `max_num_sources` outputs and fills unused ones with zeros. 

I would have the loader return:

```python
return {
    "mixture": mixture,  # [1, T]
    "targets": targets,  # [3, T]
    "n_speakers": n_speakers,
}
```

rather than letting `run_fuss_separation.py` construct the mixture.

2. **Register the dataset** in `dataset_setup.py` and the CLI parser. The current setup only knows `WHAM`, `WHAMR`, `FUSS`, `LIBRI2MIX`, and `MUSDB`. 

For example:

```python
# dataset_setup.py
import sudo_rm_rf.dnn.dataset_loader.farsi_wham as farsi_wham_loader

...

elif dataset_name == "FARSI_WHAM":
    loader = farsi_wham_loader
    root_path = FARSI_WHAM_ROOT_PATH
    translated_split = data_split
```

and add:

```python
"FARSI_WHAM"
```

to the parser choices.

Also add something like:

```python
FARSI_WHAM_ROOT_PATH = "/data/farsi_wham"
```

to `__config__.py`.

3. **Remove `online_augment()` from the FUSS training loop.** Right now `run_fuss_separation.py` receives source tensors and randomly reshuffles them across the batch before summing them. 

You don't want that because your custom loader already needs to control:

* speaker identity
* number of speakers
* temporal overlap
* WHAM SNR
* source levels

So replace:

```python
clean_wavs = online_augment(data)
clean_wavs = clean_wavs.cuda()

input_mixture = torch.sum(clean_wavs, -2, keepdim=True)
```

with approximately:

```python
input_mixture = data["mixture"].cuda()
clean_wavs = data["targets"].cuda()
```

This is probably the single biggest structural change.

4. **Do not use the current mixture-consistency step.** The FUSS script currently does:

```python
rec_sources_wavs = mixture_consistency.apply(rec_sources_wavs, input_mixture)
```

which forces:

$$
\sum_i \hat{s}_i \approx x.
$$



But in your task:

$$
x = s_1+s_2+s_3+n
$$

and you're intentionally **not predicting \(n\)**.

Forcing:

$$
\hat{s}_1+\hat{s}_2+\hat{s}_3=x
$$

would force WHAM noise back into the speech estimates.

So remove this in both training and validation:

```python
# DON'T do this
rec_sources_wavs = mixture_consistency.apply(...)
```

GroupComm itself does not require mixture consistency to execute; this is an extra projection added by the recipe.

5. **Modify the variable-source loss. This is important.** The existing FUSS recipe uses:

```python
PermInvariantSNRwithZeroRefs(n_sources=max_num_sources, inactivity_threshold=-40.0)
```



You can retain its PIT handling for active speech sources, but once you've removed mixture consistency, I would add an explicit loss on inactive outputs.

The official FUSS baseline also handles active and inactive references differently: active sources get an SNR objective, while inactive outputs are penalized according to their energy relative to the mixture. ([GitHub][1])

Conceptually:

```python
loss = active_pit_loss + lambda_inactive * inactive_loss
```

After finding the best PIT permutation:

```python
# example with only 1 active speaker

targets =
[speech, zero, zero]

matched_outputs =
[speech_est, unused_est_1, unused_est_2]
```

penalize:

```python
inactive_power = mean(unused_est_1**2 + unused_est_2**2)

mixture_power = mean(input_mixture**2)

inactive_loss = inactive_power / (mixture_power + eps)
```

A more FUSS-like thresholded version would stop caring once inactive output power is about 20 dB below mixture power. The official FUSS baseline uses a 20 dB-below-mixture threshold for inactive references. ([GitHub][1])

This prevents:

```text
Input: one speaker
Output 1: good speaker
Output 2: random residual/noise
Output 3: random residual/noise
```

which is otherwise possible.

6. **Normalize mixture and targets together.** Don't independently normalize every target, because that destroys relative levels.

For each generated example, calculate one scale:

```python
scale = mixture.std() + 1e-8

mixture = mixture / scale
targets = targets / scale
```

Then:

```python
mixture = s1 + s2 + ... + noise
```

remains physically meaningful up to the excluded noise target.

This is particularly important because the existing training loss is ordinary SNR rather than SI-SDR and therefore cares about absolute relative scale. 

7. **Fix validation for one-speaker examples.** There is a subtle issue in the current FUSS script. For one actual source it does:

```python
if n_actual_sources == 1:
    n_estimated_sources = 1
```

even though the model may produce four outputs. 

For your PIT model with three outputs, the active speaker can appear in **any of the three channels**.

Use:

```python
n_estimated_sources = hparams["max_num_sources"]
```

for every case, including one speaker:

```text
actual = 1 → estimated = 3
actual = 2 → estimated = 3
actual = 3 → estimated = 3
```

The repo's stabilized SI-SDR metric already supports having more estimated sources than actual sources by considering permutations of the estimated channels. 

8. **Don't use the current SI-SDRi calculation unchanged for noisy data.** The repo's metric calculates its SI-SDRi baseline by reconstructing the "initial mixture" as:

```python
initial_mixture = torch.sum(t_batch, -2, keepdim=True)
```



For you that becomes:

$$
s_1+s_2+s_3
$$

instead of the actual model input:

$$
s_1+s_2+s_3+n.
$$

So its reported SI-SDRi would be wrong for noisy examples.

Initially I would simply report:

```text
absolute SI-SDR
```

with:

```python
improvement = False
```

and later add a proper SI-SDRi metric that takes the actual noisy `input_mixture` as its baseline.

---

### What the modified training loop should look like

The core becomes very simple:

```python
for data in generators["train"]:
    opt.zero_grad()

    mixture = data["mixture"].cuda()  # [B, 1, T]
    targets = data["targets"].cuda()  # [B, 3, T]

    # Same normalization for input and references
    scale = mixture.std(-1, keepdim=True) + 1e-8
    mixture = mixture / scale
    targets = targets / scale

    estimates = model(mixture)  # [B, 3, T]

    # NO mixture consistency

    loss = variable_speaker_loss(estimates, targets, mixture)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

    opt.step()
```

That's the architecture I would aim for.

### Data generation inside `farsi_wham.py`

For a training example:

```text
choose N ∈ {1,2,3}

choose N distinct YodaLingua speakers

for every speaker:
    choose utterance
    crop/pad
    choose temporal offset
    choose level

speech_mix = Σ speech_targets

with probability P_noise:
    choose WHAM noise
    crop noise
    choose SNR
    noise *= appropriate_gain

mixture = speech_mix + noise

targets =
[s1, s2, s3]

unused targets = zeros
```

I'd start with something like:

```text
speaker count:
    1: 20%
    2: 45%
    3: 35%

noise:
    present: 75%
    absent:  25%

WHAM SNR:
    -5 to +20 dB

sample rate:
    16 kHz
```

For overlap, don't force each speaker to occupy the entire segment. Random placement within the window will give you partial overlap, which is much more useful for your expected task.

### Compute-conscious GroupComm configuration

I also would **not start with the full 10-second / 16-block FUSS configuration**. The repo's published FUSS command uses 16 blocks, 512 bases, and 10-second 16 kHz clips. ([GitHub][2])

For your first training run, I'd use:

```bash
python run_farsi_wham_separation.py \
  --train FARSI_WHAM \
  --val FARSI_WHAM \
  --test FARSI_WHAM \
  --min_num_sources 1 \
  --max_num_sources 3 \
  --n_channels 1 \
  --model_type groupcomm_v2 \
  --enc_kernel_size 41 \
  --enc_num_basis 512 \
  --out_channels 256 \
  --in_channels 512 \
  --num_blocks 8 \
  --group_size 16 \
  --upsampling_depth 5 \
  --audio_timelength 4.0 \
  -fs 16000 \
  -bs 4 \
  -lr 0.001 \
  --clip_grad_norm 5.0
```

I would start with **8 blocks / 4-second segments**. If the model clearly underfits or plateaus too low, the first change I'd try is:

```text
8 blocks → 16 blocks
```

rather than increasing encoder bases. GroupComm is specifically attractive because of its much lower parameter and memory footprint; the repo reports roughly 0.5M parameters for one of its 16-block/512-basis GroupComm configurations. ([GitHub][3])

One small repo bug/design oddity is also worth fixing: although the parser exposes `--group_size`, `run_fuss_separation.py` currently hardcodes:

```python
group_size = 16
```

instead of using the CLI value. 

Change it to:

```python
group_size = hparams["group_size"]
```

even if you keep it at 16 for now.

So overall, **the model file does not need modification**. The work is concentrated in:

```text
__config__.py
    → dataset path

improved_cmd_args_parser_v2.py
    → add FARSI_WHAM

dataset_setup.py
    → register FARSI_WHAM

dataset_loader/farsi_wham.py
    → new mixture generator

run_fuss_separation.py
    → unpack mixture/targets
    → remove online_augment
    → remove mixture consistency
    → max outputs = 3
    → fix validation

losses/
    → PIT active-source loss
    → explicit inactive-output penalty
```

That is the version I would implement rather than trying to force YodaLingua + WHAM into the existing FUSS folder layout.

[1]: https://github.com/google-research/sound-separation/blob/master/models/dcase2020_fuss_baseline/README.md?utm_source=chatgpt.com "sound-separation/models/dcase2020_fuss_baseline/README.md at master · google-research/sound-separation · GitHub"
[2]: https://github.com/etzinis/sudo_rm_rf "GitHub - etzinis/sudo_rm_rf: Code for SuDoRm-Rf networks for efficient audio source separation. SuDoRm-Rf stands for SUccessive DOwnsampling and Resampling of Multi-Resolution Features which enables a more efficient way of separating sources from mixtures. · GitHub"
[3]: https://github.com/etzinis/sudo_rm_rf?utm_source=chatgpt.com "GitHub - etzinis/sudo_rm_rf: Code for SuDoRm-Rf networks for efficient audio source separation. SuDoRm-Rf stands for SUccessive DOwnsampling and Resampling of Multi-Resolution Features which enables a more efficient way of separating sources from mixtures. · GitHub"
