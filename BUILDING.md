
```console
python -m venv venv
venv\scripts\activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
ffmpeg -i dataset\LJSpeech-1.1\wavs\0001.wav -acodec pcm_s16le -ac 1 -ar 22050 dataset\LJSpeech-1.1\wavs\0001_corrected.wav
```