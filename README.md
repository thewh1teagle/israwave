# israwave

Mission to create a Hebrew TTS model as powerful and user-friendly as WaveNet

## Features

- Generate sentence in less than 1ms on CPU
- Powerful text processor by espeak-ng
- Support for SSML (soon)

## Play with it!

You can play with it on [HuggingFace Space](https://huggingface.co/spaces/thewh1teagle/tts-with-israwave)

## Samples

<video src="https://github.com/user-attachments/assets/919cb5ed-ba2c-453b-8241-47d75fe3bd08" width="100" height="100"></video>

## Setup

```console
pip install -U israwave
```

You also need [`israwave.onnx`](https://github.com/thewh1teagle/israwave/releases/download/v0.1.0/israwave.onnx), [`espeak-ng-data`](https://github.com/thewh1teagle/israwave/releases/download/v0.1.0/espeak-ng-data.tar.gz), and [`nakdimon.onnx`](https://github.com/thewh1teagle/israwave/releases/download/v0.1.0/nakdimon.onnx). Please see examples.

## Examples

See [examples](examples)
