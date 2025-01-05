import re
from .symbols import phonemes_to_ids
from israwave.logging import log
from phonemizer.backend.espeak.wrapper import EspeakWrapper
import phonemizer
import espeakng_loader

WHITESPACE_RE = re.compile(r"\s+")

class IPATokenizer:
    def __init__(self, espeak_data_path = None) -> None:
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeak_data_path or espeakng_loader.get_data_path())
    
    def preprocess_text(self, text, _language):
        return self.collapse_whitespace(text)
    
    def collapse_whitespace(self, text):
        text = re.sub(WHITESPACE_RE, " ", text)
        return text
    
    def phonemize_text(self, text: str, language: str) -> str:
        # Preprocess
        text = self.preprocess_text(text, language)
        # Phonemize
        phonemes = phonemizer.phonemize(text, language, preserve_punctuation=True, with_stress=True)        
        return phonemes, text
    
    def tokenize(self, text, language):
        try:
            # Accept phonemes directly
            phoneme_ids, normalized_text = phonemes_to_ids(text), self.preprocess_text(text, 'he')
        except:
            # Create phoenems
            phonemes, normalized_text = self.phonemize_text(text, language)
            phonemes = [phoneme for sentence_phonemes in phonemes for phoneme in sentence_phonemes]
            phonemes = list(self.collapse_whitespace("".join(phonemes)))
            phoneme_ids = phonemes_to_ids(phonemes)
            log.debug(f"phonemes: {''.join(phonemes)} text: {text}")
        return phoneme_ids, normalized_text