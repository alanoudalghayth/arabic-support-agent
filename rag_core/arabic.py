
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

    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)

    return re.sub(r"\s+", " ", text).strip()

def detect_lang(text: str) -> str:

    """Cheap check: is this text mostly Arabic or mostly English?"""

    ar = len(ARABIC_RANGE.findall(text))

    la = len(LATIN_RANGE.findall(text))

    if ar == 0 and la == 0:

        return "unknown"

    return "ar" if ar >= la else "en"

if __name__ == "__main__":

    s = "الْمُسْتَخْدِمُونَ الأعزّاء ــــ الطلب رقم ١٢٣"

    print("raw :", s)

    print("norm:", normalize_ar(s))

    print("lang:", detect_lang(s))
