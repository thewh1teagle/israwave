"""
pip install -U israwave
wget https://github.com/thewh1teagle/israwave/releases/download/v0.1.0/israwave.onnx
wget https://github.com/thewh1teagle/israwave/releases/download/v0.1.0/nakdimon.onnx
wget https://github.com/thewh1teagle/israwave/releases/download/v0.1.0/espeak-ng-data.tar.gz
tar xf espeak-ng-data.tar.gz

python examples/play.py israwave.onnx espeak-ng-data nakdimon.onnx 'הי, אתם על חיות כיס, אני שאול אמסטרדמסקי. אתם האזנתם לחיות כיס, הפודקאסט הכלכלי של כאן. עורך חיות כיס הוא תומר מיכלזון. במערכת חיות כיס תמצאו גם את צליל אברהם.'
"""

import sys
import numpy as np
import sounddevice as sd
from israwave import IsraWave
from israwave.segment import SegmentExtractor
from nakdimon_ort import Nakdimon
from pathlib import Path

if __name__ == '__main__':
    speech_model_path, espeak_data_path = sys.argv[1], sys.argv[2]
    niqqud_model_path, text = sys.argv[3], sys.argv[4]
    
    if Path(text).exists():
        text = open(text, encoding='utf-8').read()
    
    segment_extractor = SegmentExtractor()
    speech_model = IsraWave(speech_model_path, espeak_data_path)
    niqqud_model = Nakdimon(niqqud_model_path)
    
    text = niqqud_model.compute(text)
    
    waveforms = []
    
    for segment in segment_extractor.extract_segments(text):
        waveform = speech_model.create(segment.text)
        waveforms.append(waveform.samples)
        silence = segment.create_pause(waveform.sample_rate)
        waveforms.append(silence)

    # Join segments into a single waveform
    final_waveform = np.concatenate(waveforms)
    sd.play(final_waveform, speech_model.sample_rate)
    sd.wait()
