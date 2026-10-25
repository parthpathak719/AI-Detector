"""Label-independent inference equivalent of the project's strict normalization.

The training-only, label-dependent sentence trimming is deliberately excluded.
"""
import re
import unicodedata

LANGUAGES = {
    "en": "English", "hi": "Hindi", "ta": "Tamil", "kn": "Kannada",
    "hi_en": "Hinglish", "ta_en": "Tanglish", "kn_en": "Kanglish",
}
MAX_CHARS = 500000
MAX_WORDS = 50000


def clean_text(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    t = re.sub(r"^\s*RT\s*:?\s+", "", t, flags=re.I)
    t = re.sub(r"https?://\S+|www\.\S+", " ", t, flags=re.I)
    t = re.sub(r"@\w+", " ", t).replace("#", "")
    t = re.sub("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]", " ", t)
    replacements = {
        "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2018": "'", "\u2019": "'",
        "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2026": "...",
        "\u00a0": " ", "\u200b": "", "\ufeff": "", "\u0964": ".", "\u0965": ".",
    }
    for a, b in replacements.items():
        t = t.replace(a, b)
    t = t.replace("**", "").replace("__", "").replace("`", "")
    t = re.sub(r"\d", "", t.lower()).replace("-", " ")
    t = "".join(c for c in t if unicodedata.category(c)[0] not in ("P", "S"))
    return " ".join(t.split())


def validate_text(text: str, language: str) -> str:
    if language not in LANGUAGES:
        raise ValueError("Choose one of the seven supported language settings.")
    if len(text) > MAX_CHARS:
        raise ValueError(f"Please use a passage under {MAX_CHARS:,} characters.")
    cleaned = clean_text(text)
    count = len(cleaned.split())
    minimum = 4 if language.endswith("_en") else 8
    if count < minimum or not any(c.isalpha() for c in cleaned):
        raise ValueError(f"Please enter at least {minimum} words of text after cleaning.")
    if count > MAX_WORDS:
        raise ValueError(f"Please use a document with up to {MAX_WORDS:,} words.")
    return cleaned


def source_excerpts(text, cleaned, chunks):
    """Map unchanged normalized model windows back to readable source excerpts.

    A compound source token can yield multiple cleaned words. If a window splits
    that token, adjacent excerpts share it rather than displaying a broken word.
    No source-to-model alignment is guessed when normalization differs.
    """
    prefix = re.match(r'^\s*RT\s*:?\s+', text, flags=re.I)
    offset = prefix.end() if prefix else 0
    words, spans = [], []
    for match in re.finditer(r'\S+', text[offset:]):
        normalized = clean_text(match.group()).split()
        words.extend(normalized)
        spans.extend([(offset + match.start(), offset + match.end())] * len(normalized))
    if ' '.join(words) != cleaned or ' '.join(chunks) != cleaned:
        raise ValueError('Unable to align source text with model segments. Try plain text without unusual control characters.')
    excerpts, index = [], 0
    for number, chunk in enumerate(chunks):
        size = len(chunk.split())
        start = 0 if number == 0 else spans[index][0]
        index += size
        end = len(text) if index == len(words) else max(spans[index - 1][1], spans[index][0])
        excerpts.append({'text': text[start:end], 'source_start': start, 'source_end': end})
    return excerpts
