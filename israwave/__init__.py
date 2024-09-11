from israwave.values import FloatArray
from .model import Model
from nakdimon_ort import Nakdimon
import soundfile as sf

class WaveForm:
    def __init__(self, waveform: FloatArray, sample_rate: int) -> None:
        self.waveform =  waveform
        self.sample_rate = sample_rate
    
    def save(self, path: str):
        sf.write(path, self.waveform, self.sample_rate)

class IsraWave:
    def __init__(self, speech_model_path: str, espeak_data_path: str) -> None:
        self.speech_model = Model(speech_model_path, espeak_data_path)
        
        # Scale to control speech rate.
        self.d_factor = 1.0
        # Scale to control pitch.
        self.p_factor = 1.0
        # Scale to control energy.
        self.e_factor = 1.0
    
    def create(self, text):
        inputs = self.speech_model.prepare_input(
            text,
            d_factor=self.d_factor,
            p_factor=self.p_factor,
            e_factor=self.e_factor,
            lang='he'
        )
        outputs = self.speech_model.synthesise(inputs)
        wav = outputs.unbatched_wavs()[0]
        waveform = WaveForm(wav, self.speech_model.sample_rate)
        return waveform
        
        