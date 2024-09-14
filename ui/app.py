"""
pip install -r requirements.txt
wget https://github.com/thewh1teagle/israwave/releases/download/v0.1.0/israwave.onnx
wget https://github.com/thewh1teagle/israwave/releases/download/v0.1.0/nakdimon.onnx
wget https://github.com/thewh1teagle/israwave/releases/download/v0.1.0/espeak-ng-data.tar.gz
tar xf espeak-ng-data.tar.gz

python3 app.py
"""

import gradio as gr
from israwave import IsraWave
from nakdimon_ort import Nakdimon
from israwave.segment import SegmentExtractor
import numpy as np

segment_extractor = SegmentExtractor()
speech_model = IsraWave('israwave.onnx', 'espeak-ng-data')
niqqud_model = Nakdimon('nakdimon.onnx')

def create_audio(text: str, rate, pitch, energy):
    text = niqqud_model.compute(text)
    waveforms = []
    for segment in segment_extractor.extract_segments(text):
        waveform = speech_model.create(segment.text, rate=rate, pitch=pitch, energy=energy)
        waveforms.append(waveform.samples)
        silence = segment.create_pause(waveform.sample_rate)
        waveforms.append(silence)
    return np.concatenate(waveforms), speech_model.sample_rate

def create(text, rate, pitch, energy):
    samples, sample_rate = create_audio(text, rate, pitch, energy)
    return (sample_rate, samples)


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Centered title
    gr.Markdown("<h1 style='text-align: center;'>IsraWave</h1>")
    
    # Use Textarea with RTL direction
    text = gr.TextArea(label="text", lines=4, elem_id="rtl_textarea", value='זה כיף להזמין דברים באינטרנט, אבל הרבה פחות כיף לחכות ולחכות עד שהם יגיעו אלינו. אז מה בעצם עובר על החבילות בדרך הארוכה עד לבית שלנו? והאם אפשר לגרום לכך שהן יגיעו מהר יותר? ')
    rate = gr.Slider(0.1, 10, label="rate", value=1.0)
    pitch = gr.Slider(0.1, 10, label="pitch", value=1.0)
    energy = gr.Slider(0.1, 10, label="energy", value=1.0)

    button = gr.Button("Create", elem_id="create_button")
    output = gr.Audio()
    
    button.click(fn=create, inputs=[text, rate, pitch, energy], outputs=output)

    # Custom CSS for RTL direction
    demo.css = """
    #rtl_textarea textarea {
        direction: rtl;
        font-size: 20px;
    }
    """

demo.launch()