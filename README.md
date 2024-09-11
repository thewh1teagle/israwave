# israwave

Mission to create a Hebrew TTS model as powerful and user-friendly as WaveNet

## Usage

```console
wget https://github.com/thewh1teagle/optispeech/releases/download/v0.1.0/epoch-192-step-128696.onnx
rye sync
python3 -m src.israwave epoch-192-step-128696.onnx "מָה קוֹרֶה אָחִי? אֵיךְ הוֹלֵךְ?" output --no-split
```