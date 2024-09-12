"""
pip install -U israwave
wget https://github.com/thewh1teagle/optispeech/releases/download/v0.1.0/epoch-192-step-128696.onnx
wget https://github.com/thewh1teagle/optispeech/releases/download/v0.1.0/espeak-ng-data.7z
wget https://github.com/thewh1teagle/nakdimon-ort/releases/download/v0.1.0/nakdimon.onnx


python3 play.py israwave.onnx espeak-ng-data nakdimon.onnx 'תגידו, גנבו לכם פעם את האוטו ופשוט ידעתם שאין טעם להגיש תלונה במשטרה?'
"""

from israwave import IsraWave
from nakdimon_ort import Nakdimon
import sys

if __name__ == '__main__':
    speech_model_path, espeak_data_path = sys.argv[1], sys.argv[2]
    niqqud_model_path, text = sys.argv[3], sys.argv[4]
    
    speech_model = IsraWave(speech_model_path, espeak_data_path)
    niqqud_model = Nakdimon(niqqud_model_path)
    text = niqqud_model.compute(text)
    waveform = speech_model.create(text)
    waveform.play()
