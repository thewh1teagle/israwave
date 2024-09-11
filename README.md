# israwave

Mission to create a Hebrew TTS model as powerful and user-friendly as WaveNet

## Features

- Generate sentence in less than 1ms on CPU
- Powerful text processor by espeak-ng
- Support for SSML (soon)

## Samples

See example audio [here](https://github.com/thewh1teagle/israwave/releases/download/v0.1.0/israwave.wav)

## Usage

```console
wget https://github.com/thewh1teagle/optispeech/releases/download/v0.1.0/epoch-192-step-128696.onnx
wget https://github.com/thewh1teagle/optispeech/releases/download/v0.1.0/espeak-ng-data.7z
wget https://github.com/thewh1teagle/nakdimon-ort/releases/download/v0.1.0/nakdimon.onnx

7z x espeak-ng-data.7z
export ESPEAK_DATA_PATH=$(pwd)/espeak-ng-data
rye sync
python3 -m src.israwave epoch-192-step-128696.onnx "מָה קוֹרֶה אָחִי? אֵיךְ הוֹלֵךְ?" output --no-split
```
