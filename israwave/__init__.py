from israwave.tensors import FloatArray
from .model import Model
import soundfile as sf
import sounddevice as sd
import os
from israwave.logging import setup_logging
from pathlib import Path

setup_logging()

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
        # Check if the speech model exists
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Speech model not found at {model_path}\n"
                "Please download and prepare the model using the following commands:\n"
                "wget https://github.com/thewh1teagle/israwave/releases/download/v0.1.0/israwave.onnx"
            )

        # Check if the espeak data folder exists
        if not Path(espeak_data_path).exists():
            raise FileNotFoundError(
                f"Espeak data folder not found at {espeak_data_path}\n"
                "Please download and extract the espeak data using the following commands:\n"
                "wget https://github.com/thewh1teagle/israwave/releases/download/v0.1.0/espeak-ng-data.tar.gz\n"
                "tar xf espeak-ng-data.tar.gz"
            )

        self.model = Model(model_path, espeak_data_path)
    
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
        inputs = self.model.prepare_input(
            text,
            d_factor=d_factor,
            p_factor=p_factor,
            e_factor=e_factor,
            lang='he'
        )
        outputs = self.model.synthesise(inputs)
        wav = outputs.unbatched_wavs()[0]
        waveform = WaveForm(wav, self.model.sample_rate)
        return waveform
        
        