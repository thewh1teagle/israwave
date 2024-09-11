from pathlib import Path
import json

config_path = Path(__file__).parent / 'config.json'
with open(config_path) as f:
    config = json.load(f)
    SYMBOLS = config['symbols']
   
# Special symbols
PAD = "_"
BOS = "^"
EOS = "$"

# Special symbol ids
PAD_ID = SYMBOLS.index(PAD)
BOS_ID = SYMBOLS.index(BOS)
EOS_ID = SYMBOLS.index(EOS)
SPACE_ID = SYMBOLS.index(" ")

# Mappings from symbol to numeric ID and vice versa:
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
ID_TO_SYMBOL = {i: s for i, s in enumerate(SYMBOLS)}  # pylint: disable=unnecessary-comprehension


def phonemes_to_ids(text):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []
    for symbol in text:
        symbol_id = SYMBOL_TO_ID[symbol]
        sequence.append(symbol_id)
    return sequence


def ids_to_phonemes(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        s = ID_TO_SYMBOL[symbol_id]
        result += s
    return result
