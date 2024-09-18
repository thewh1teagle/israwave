def text_has_niqqud(text: str) -> bool:
    return any("\u0591" <= char <= "\u05C7" for char in text)

def text_has_ipa(text: str) -> bool:
    return any("\u0250" <= char <= "\u02AF" for char in text)