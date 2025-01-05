# israwave

Mission to create a Hebrew TTS model as powerful and user-friendly as WaveNet

## Features

- Generate sentence in less than 1ms on CPU
- Powerful text processor by espeak-ng
- Support for SSML (soon)

## Play with it!

You can play with it on [HuggingFace Space](https://huggingface.co/spaces/thewh1teagle/tts-with-israwave)

## Samples

https://github.com/user-attachments/assets/3212a800-406f-4d79-8aa1-d814eed815d6

## Setup

```console
pip install -U israwave
```

Some dependencies are only available in python version 3.12. We recommend use [uv](https://docs.astral.sh/uv/getting-started/installation).

- You also need to place the files [`israwave.onnx`](https://github.com/thewh1teagle/israwave/releases/download/v0.1.0/israwave.onnx), [`nakdimon.onnx`](https://github.com/thewh1teagle/israwave/releases/download/v0.1.0/nakdimon.onnx), and extracted [`espeak-ng-data`](https://github.com/thewh1teagle/israwave/releases/download/v0.1.0/espeak-ng-data.tar.gz) in the project folder. 
- We recommend use [uv](https://docs.astral.sh/uv/getting-started/installation) for Python.
- Python 3.12 is recommended. (`uv init -p 3.12`)
- Please see examples.

## Examples

See [examples](examples)

## Dataset

The model trained on [saspeech gold standard](https://openslr.org/134/).

## Thanks

Thanks to [Kan11](https://www.kan.org.il/) and [Shaul](https://www.kan.org.il/authors/%D7%A9%D7%90%D7%95%D7%9C-%D7%90%D7%9E%D7%A1%D7%98%D7%A8%D7%93%D7%9E%D7%A1%D7%A7%D7%99/) for providing the dataset.

Thanks to [elazarg](https://github.com/elazarg) for sharing the [Nakdimon](https://github.com/elazarg/nakdimon) diacritics model, which was instrumental in our project.

For [mush42](https://github.com/mush42) for their excellent TTS training recipe.
