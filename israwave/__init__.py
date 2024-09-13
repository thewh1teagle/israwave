from israwave.tensors import FloatArray
from .model import Model
import soundfile as sf
import sounddevice as sd
import os
from israwave.logging import init_logging

init_logging()

class WaveForm:
    def __init__(self, samples: FloatArray, sample_rate: int) -> None:
        self.samples =  samples
        self.sample_rate = sample_rate
    
    def save(self, path: str):
        if os.path.exists(path):
            os.remove(path)
        sf.write(path, self.samples, self.sample_rate)
        
    def play(self):
        sd.play(self.samples, self.sample_rate)
        sd.wait()

class IsraWave:
    def __init__(self, model_path: str, espeak_data_path: str) -> None:
        self.speech_model = Model(model_path, espeak_data_path)
    
    def create(self, text, d_factor = 1.0, p_factor = 1.0, e_factor = 1.0):
        """create speech waveform

        Args:
            text str: _description_
            d_factor (float, optional): Control rate. Defaults to 1.0.
            p_factor (float, optional): Control pitch. Defaults to 1.0.
            e_factor (float, optional): Control energy. Defaults to 1.0.

        Returns:
            _type_: _description_
        """
        inputs = self.speech_model.prepare_input(
            text,
            d_factor=d_factor,
            p_factor=p_factor,
            e_factor=e_factor,
            lang='he'
        )
        outputs = self.speech_model.synthesise(inputs)
        wav = outputs.unbatched_wavs()[0]
        waveform = WaveForm(wav, self.speech_model.sample_rate)
        return waveform
        
        