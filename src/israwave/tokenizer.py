from os import getenv
import re
from .symbols import phonemes_to_ids

WHITESPACE_RE = re.compile(r"\s+")

class IPATokenizer:
    def __init__(self) -> None:
        pass
    
    def preprocess_text(self, text, _language):
        text = re.sub(WHITESPACE_RE, " ", text)
        return text
    
    def collapse_whitespace(self, text):
        text = re.sub(WHITESPACE_RE, " ", text)
        return text
    
    def phonemize_text(self, text: str, language: str) -> str:
        try:
            from piper_phonemize import phonemize_espeak
        except ImportError:
            raise ImportError(
                "piper-phonemize package is needed for the IPA tokenizer.\n"
                "pip install piper-phonemize\n"
                "or build it yourself from the following repository:\n"
                "https://github.com/rhasspy/piper-phonemize"
            )

        # Preprocess
        text = self.preprocess_text(text, language)
        # Phonemize        
        espeak_data_path = getenv('ESPEAK_DATA_PATH')
        phonemes = phonemize_espeak(text, language, data_path=espeak_data_path)
        return phonemes, text
    
    def tokenize(self, text, language):
        phonemes, normalized_text = self.phonemize_text(text, language)
        phonemes = [phoneme for sentence_phonemes in phonemes for phoneme in sentence_phonemes]
        phonemes = list(self.collapse_whitespace("".join(phonemes)))
        phoneme_ids = phonemes_to_ids(phonemes)
        return phoneme_ids, normalized_text