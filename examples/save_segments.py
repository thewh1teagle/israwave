"""
pip install -U israwave
wget https://github.com/thewh1teagle/israwave/releases/download/v0.1.0/israwave.onnx
wget https://github.com/thewh1teagle/optispeech/releases/download/v0.1.0/espeak-ng-data.7z
wget https://github.com/thewh1teagle/nakdimon-ort/releases/download/v0.1.0/nakdimon.onnx

python save_segments.py israwave.onnx espeak-ng-data nakdimon.onnx input.txt output.wav
"""

import sys
import numpy as np
import soundfile as sf
from israwave import IsraWave
from israwave.segment import SegmentExtractor
from nakdimon_ort import Nakdimon
from pathlib import Path

if __name__ == '__main__':
    speech_model_path, espeak_data_path = sys.argv[1], sys.argv[2]
    niqqud_model_path, text, out_path = sys.argv[3], sys.argv[4], sys.argv[5]
    
    if Path(text).exists():
        text = open(text, encoding='utf-8').read()
    
    segment_extractor = SegmentExtractor()
    speech_model = IsraWave(speech_model_path, espeak_data_path)
    niqqud_model = Nakdimon(niqqud_model_path)
    
    text = niqqud_model.compute(text)
    
    waveforms = []
    sample_rate = None
    
    for i, segment in enumerate(segment_extractor.extract_segments(text)):
        waveform = speech_model.create(segment.text)
        if sample_rate is None:
            sample_rate = waveform.sample_rate
        
        waveforms.append(waveform.samples)
        silence = np.zeros(int(segment.next_pause * sample_rate))
        waveforms.append(silence)

    # Join segments into a single waveform
    final_waveform = np.concatenate(waveforms)
    sf.write(out_path, final_waveform, sample_rate)
