# Training recipe for israwave

Dataset https://openslr.elda.org/134/

Training recipe https://github.com/thewh1teagle/optispeech/tree/he-v1

Model based on vitshttps://github.com/jaywalnut310/vits

10% kept for validation.

_configs/data/feature_extractor/44.10khz.yaml_

```yaml
defaults:
  - default
  - _self_

sample_rate: 44100
n_feats: 80
n_fft: 2048
hop_length: 512
win_length: 2048
f_min: 20
f_max: 11025
```

_configs/data/saspeech-he.yaml_

```yaml
defaults:
  - _self_
  - text_processor: he
  - feature_extractor: 44.10khz

_target_: optispeech.dataset.TextWavDataModule
name: saspeech-he
num_speakers: 1
train_filelist_path: data/saspeech/train.txt
valid_filelist_path: data/saspeech/val.txt
batch_size: 16
num_workers: 8
pin_memory: True
data_statistics:  # Computed for saspeech dataset
  pitch_min: 55.238178
  pitch_max: 550.746582
  pitch_mean: 143.805801
  pitch_std: 36.134865
  energy_min: 0.001879
  energy_max: 578.68219
  energy_mean: 94.595634
  energy_std: 79.05619
  mel_mean: -4.198716
  mel_std: 2.186619
seed: ${seed}
```

_configs/text_processor/he.yaml_

```yaml
_target_: optispeech.text.TextProcessor
tokenizer: ipa
add_blank: false
add_bos_eos: false
normalize_text: false
languages:
  - he
```

_configs/experiment/saspeech-he.yaml_

```yaml
# @package _global_

# to execute this experiment run:
# python train.py experiment=multispeaker

defaults:
  - override /data: saspeech-he.yaml
  - override /model: convnext_tts.yaml

# all parameters below will be merged with parameters from default configurations set above
# this allows you to overwrite only specified parameters

tags: ["saspeech-he"]

run_name: saspeech_he


trainer:
  max_steps: 2000000
  check_val_every_n_epoch: 1000
```