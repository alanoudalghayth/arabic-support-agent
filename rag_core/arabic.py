
"""Arabic text handling for retrieval.

The goal is to normalize text for KEYWORD search (BM25), but keep the raw text for

meaning-based search (embeddings) and for display. Arabic writes the same letter

several ways -- أ إ آ ا are one letter to a reader but four different characters

to a computer -- so without normalization, keyword search silently misses matches.

"""

import re

DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")

TATWEEL = "\u0640"

ARABIC_RANGE = re.compile(r"[\u0600-\u06FF]")

LATIN_RANGE = re.compile(r"[A-Za-z]")

def normalize_ar(text: str) -> str:

    """Normalize Arabic for keyword matching. Not for display."""

    text = DIACRITICS.sub("", text)

    text = text.replace(TATWEEL, "")

    text = re.sub(r"[إأآٱا]", "ا", text)

    text = re.sub(r"ى", "ي", text)

    text = re.sub(r"ؤ", "و", text)

    text = re.sub(r"ئ", "ي", text)

    text = re.sub(r"ة", "ه", text)

    text = re.sub(r"[\u0660-\u0669]", lambda m: str(ord(m.group()) - 0x0660), text)

    text = re.sub(r"[؟،؛٪؉۔]", " ", text)
   
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)

    return re.sub(r"\s+", " ", text).strip()

def detect_lang(text: str) -> str:

    """Cheap check: is this text mostly Arabic or mostly English?"""

    ar = len(ARABIC_RANGE.findall(text))

    la = len(LATIN_RANGE.findall(text))

    if ar == 0 and la == 0:

        return "unknown"

    return "ar" if ar >= la else "en"

AR_PREFIXES = ("ال", "وال", "بال", "كال", "فال", "لل", "و", "ف", "ب", "ك", "ل")
AR_SUFFIXES = ("ها", "ان", "ات", "ون", "ين", "ية", "ه", "ة", "ي", "ا")

def light_stem(token: str) -> str:
    """Strip common Arabic prefixes/suffixes so related word-forms match.
    Crude but effective for keyword search: الارجاع, ارجاع, ارجع move closer."""
    for p in AR_PREFIXES:
        if token.startswith(p) and len(token) - len(p) >= 3:
            token = token[len(p):]
            break
    for s in AR_SUFFIXES:
        if token.endswith(s) and len(token) - len(s) >= 3:
            token = token[:-len(s)]
            break
    return token

def stem_tokens(text: str) -> list[str]:
    """Normalize then light-stem each token, for BM25."""
    return [light_stem(t) for t in normalize_ar(text).split()]

if __name__ == "__main__":

    s = "الْمُسْتَخْدِمُونَ الأعزّاء ــــ الطلب رقم ١٢٣"

    print("raw :", s)

    print("norm:", normalize_ar(s))

    print("lang:", detect_lang(s))
