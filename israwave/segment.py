import re
from dataclasses import dataclass

@dataclass
class Segment:
    text: str
    next_pause: float

class SegmentExtractor:
    def __init__(self, default_pause: float = 0.02, question_pause: float = 0.05, period_pause: float = 0.05, new_line_pause = 0.3):
        self.default_pause = default_pause
        self.question_pause = question_pause
        self.period_pause = period_pause
        self.new_line_pause = new_line_pause
    
    def extract_segments(self, text: str):
        segments: list[Segment] = []
        sentences = re.split(r'([.?!:\n])', text)
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            punctuation = sentences[i + 1]
            if sentence:  # Ensure the sentence is not empty
                if punctuation == '.':
                    segments.append(Segment(text=f"{sentence}{punctuation}", next_pause=self.period_pause))
                elif punctuation == '?':
                    segments.append(Segment(text=f"{sentence}{punctuation}", next_pause=self.question_pause))
                elif punctuation == '\n':
                    segments.append(Segment(text=f"{sentence}{punctuation}.", next_pause=self.new_line_pause))
                else:
                    segments.append(Segment(text=f"{sentence}{punctuation}", next_pause=self.default_pause))
        last_sentence = sentences[-1].strip()
        if last_sentence:
            segments.append(Segment(text=f"{last_sentence}.", next_pause=self.default_pause))
        return segments