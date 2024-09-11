"""
wget https://github.com/thewh1teagle/optispeech/releases/download/v0.1.0/epoch-192-step-128696.onnx
wget https://github.com/thewh1teagle/optispeech/releases/download/v0.1.0/espeak-ng-data.7z
wget https://github.com/thewh1teagle/nakdimon-ort/releases/download/v0.1.0/nakdimon.onnx

python3 usage.py israwave.onnx espeak-ng-data nakdimon.onnx "שלום! מה קורה?" output.wav
"""

from israwave import IsraWave
from nakdimon_ort import Nakdimon
import sys

if __name__ == '__main__':
    speech_model = IsraWave(sys.argv[1], sys.argv[2])
    niqqud_model = Nakdimon(sys.argv[3])
    text = sys.argv[4]
    out_path = sys.argv[5]
    
    text = niqqud_model.compute(text)
    waveform = speech_model.create(text)
    waveform.save(out_path)
    