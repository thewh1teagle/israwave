# Training recipe for israwave

Dataset https://openslr.elda.org/134/

Training recipe https://github.com/thewh1teagle/optispeech/tree/he-v1

Model based on vitshttps://github.com/jaywalnut310/vits

10% kept for validation.


## Summary of Steps Taken to Make This Work in Hebrew

1. Download [saspeech dataset](https://openslr.elda.org/134/) (4 hours, diactirized audio with 2-15s audio files)
2. Improve espeak-ng phonemes in [this PR](https://github.com/espeak-ng/espeak-ng/pull/1983)
3. Remove invalid / too short file from dataset `sed -i '/^gold_000_line_104/d' saspeech_gold_standard/metadata.csv`
4. Split dataset into two folders (train/val) and keep 10% for validation
5. [Patch](https://github.com/mush42/optispeech/pull/4) training code to accept custom espeak-ng phonemes that I added in Hebrew
6. Prepare feature extractor configurations for dataset (44.10khz)
7. Prepare main train config for Hebrew with correct langauge code and feature extractor (44.10khz)
8. Prepare `TextProcessor` config and disable normalization (important!!) since it remove / change diacritics. kept ipa tokenizer.
9. Execute preprocess to create folders for train/validation and `npz` files and `json` files. `npz` contains sort of waveform and `json` contains phonemes IDS (converted from IPA)
10. Prepare `data_statistics` from dataset and set in train config.
11. Start the train. it should store by default 10 last checkpoints including metadata / how many steps (and overwrite in cycle)

Notes:

Since Hebrew is complex language I used diacritics in this model. I created [nakdimon-ort](https://github.com/thewh1teagle/nakdimon-ort) for fast diacritics inferencing based on nakdimon model.

It took me 3-4 days to reach 1M steps on rtx3090 batch size of 4/8/16 depending on VRAM. Better to use 8-16 batch size - less noisy.

You can stop the training and convert to onnx and inference with this repository to check that it's correct.

Use [tensorboard](https://www.tensorflow.org/tensorboard) to see information while training running.

Notes for improving:

Better diacritics model is needed


## Config

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