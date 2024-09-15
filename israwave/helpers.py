def text_has_niqqud(text: str) -> bool:
    return any("\u0591" <= char <= "\u05C7" for char in text)
