import numpy as np
import soundfile as sf
import io

def text_has_niqqud(text: str) -> bool:
    return any("\u0591" <= char <= "\u05C7" for char in text)

def text_has_ipa(text: str) -> bool:
    return any("\u0250" <= char <= "\u02AF" for char in text)

def float_to_int16(samples: np.floating) -> np.int16:
    """
    Normalize audio_array if it's floating-point
    """
    if np.issubdtype(samples.dtype, np.floating):
        max_val = np.max(np.abs(samples))
        samples = (samples / max_val) * 32767 # Normalize to 16-bit range
        samples = samples.astype(np.int16)
    return samples

def to_ogg(samples: np.array, sample_rate: int):
    # Normalize audio_array if it's floating-point
    samples = float_to_int16(samples)
    # Create in memory buffer of ogg
    buf = io.BytesIO()
    buf.name = 'audio.ogg'
    sf.write(buf, samples, sample_rate, format="ogg")
    buf.seek(0)
    return buf
