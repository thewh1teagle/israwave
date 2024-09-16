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
from israwave.helpers import text_has_niqqud
from nakdimon_ort import Nakdimon
from israwave.segment import SegmentExtractor
import numpy as np
from pydub import AudioSegment
import io


def numpy_to_mp3(audio_array, sampling_rate):
    # Normalize audio_array if it's floating-point
    if np.issubdtype(audio_array.dtype, np.floating):
        max_val = np.max(np.abs(audio_array))
        audio_array = (audio_array / max_val) * 32767 # Normalize to 16-bit range
        audio_array = audio_array.astype(np.int16)

    # Create an audio segment from the numpy array
    audio_segment = AudioSegment(
        audio_array.tobytes(),
        frame_rate=sampling_rate,
        sample_width=audio_array.dtype.itemsize,
        channels=1
    )

    # Export the audio segment to MP3 bytes - use a high bitrate to maximise quality
    mp3_io = io.BytesIO()
    audio_segment.export(mp3_io, format="mp3", bitrate="320k")

    # Get the MP3 bytes
    mp3_bytes = mp3_io.getvalue()
    mp3_io.close()

    return mp3_bytes

segment_extractor = SegmentExtractor()
speech_model = IsraWave('israwave.onnx', 'espeak-ng-data')
niqqud_model = Nakdimon('nakdimon.onnx')

def create_audio(text: str, rate, pitch, energy):
    if not text_has_niqqud(text):
        text = niqqud_model.compute(text)
    for segment in segment_extractor.extract_segments(text):
        waveform = speech_model.create(segment.text, rate=rate, pitch=pitch, energy=energy)
        silence = segment.create_pause(waveform.sample_rate)
        audio = np.concatenate([waveform.samples, silence])
        waveform_bytes = numpy_to_mp3(audio, waveform.sample_rate)
        yield waveform_bytes

def create(text, rate, pitch, energy):
    yield from create_audio(text, rate, pitch, energy)


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Centered title
    gr.Markdown("""
    <h1 style='text-align: center;'>IsraWave</h1>
    <p style='text-align: center;'>Text-to-Speech model for Hebrew</p>
    """)
    
    # Use Textarea with RTL direction
    text = gr.TextArea(label="text", lines=4, elem_id="rtl_textarea", value='זה כיף להזמין דברים באינטרנט, אבל הרבה פחות כיף לחכות ולחכות עד שהם יגיעו אלינו. אז מה בעצם עובר על החבילות בדרך הארוכה עד לבית שלנו? והאם אפשר לגרום לכך שהן יגיעו מהר יותר? ')
    rate = gr.Slider(0.1, 10, label="rate", value=1.0)
    pitch = gr.Slider(0.1, 10, label="pitch", value=1.0)
    energy = gr.Slider(0.1, 10, label="energy", value=1.0)

    button = gr.Button("Create", elem_id="create_button")
    output = gr.Audio(streaming=True, autoplay=True)
    
    button.click(fn=create, inputs=[text, rate, pitch, energy], outputs=output)

    # Custom CSS for RTL direction
    demo.css = """
    #rtl_textarea textarea {
        direction: rtl;
        font-size: 20px;
    }
    """
    
    gr.Markdown("""
    <p style='text-align: center;'><a href='https://github.com/thewh1teagle/israwave' target='_blank'>Israwave on Github</a></p>
    """)

demo.launch()