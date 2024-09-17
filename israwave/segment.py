import re
from dataclasses import dataclass
import numpy as np
import logging

@dataclass
class Segment:
    text: str
    next_pause: float
    
    def create_pause(self, sample_rate: int):
        """
        Return array of zeros (silence) with size of the next_pause
        """
        num_samples = int(self.next_pause * sample_rate)
        return np.zeros(num_samples, dtype=np.float32)

class SegmentExtractor:
    def __init__(self, default_pause: float = 0.05, question_pause: float = 0.05, dot_pause: float = 0.05, new_line_pause = 0.05, hyphen_pause = 0.01):
        self.default_pause = default_pause
        self.question_pause = question_pause
        self.period_pause = dot_pause
        self.new_line_pause = new_line_pause
        self.hyphen_pause = hyphen_pause
    
    def extract_segments(self, text: str):
        """
        Break the text into segment items. eg. on '?' or '.' or '!' or '\n'
        Used later to add pauses when speaking
        """
        logging.debug(f'extract segments from {text}')
        sentences = re.split(r'([.?!:–\n\-])', text)
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            punctuation = sentences[i + 1]
            if sentence:  # Ensure the sentence is not empty
                if punctuation == '.':
                    yield Segment(text=f"{sentence}{punctuation}", next_pause=self.period_pause)
                elif punctuation == '?':
                    yield Segment(text=f"{sentence}{punctuation}", next_pause=self.question_pause)
                elif punctuation == '\n':
                    yield Segment(text=f"{sentence}{punctuation}.", next_pause=self.new_line_pause)
                elif punctuation == '-':
                    yield Segment(text=f"{sentence}{punctuation}", next_pause=self.hyphen_pause)
                else:
                    yield Segment(text=f"{sentence}{punctuation}", next_pause=self.default_pause)
        last_sentence = sentences[-1].strip()
        if last_sentence:
            yield Segment(text=f"{last_sentence}.", next_pause=self.default_pause)