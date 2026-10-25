"""Conservative local language hints, not a trained language detector."""
import re
import unicodedata
from collections import Counter
from detector.text import LANGUAGES

ROMAN = {
    'hi_en': {'hai', 'hain', 'nahi', 'nahin', 'mera', 'meri', 'mujhe', 'tum', 'aap', 'kya', 'kaise', 'bahut', 'lekin', 'kyunki'},
    'ta_en': {'enna', 'enakku', 'ungal', 'unga', 'irukku', 'illa', 'romba', 'nalla', 'pannunga', 'venum', 'naan', 'namma'},
    'kn_en': {'nanu', 'naanu', 'nanage', 'nimma', 'ninna', 'yenu', 'hege', 'tumba', 'beku', 'illa', 'ide', 'madbeku'},
}


def suggest_language(text):
    counts = Counter()
    for c in text:
        if not c.isalpha() and not unicodedata.category(c).startswith("M"):
            continue
        value = ord(c)
        code = ('hi' if 0x0900 <= value <= 0x097f else
                'ta' if 0x0b80 <= value <= 0x0bff else
                'kn' if 0x0c80 <= value <= 0x0cff else
                'latin' if 'a' <= c.lower() <= 'z' else 'other')
        counts[code] += 1
    native = [(counts[code], code) for code in ('hi', 'ta', 'kn')]
    amount, code = max(native)
    if amount >= 8 and amount / max(sum(counts.values()), 1) >= .15:
        mixed = counts['latin'] >= 10 and counts['latin'] / sum(counts.values()) >= .2
        selected = code + '_en' if mixed else code
        reason = 'Script-based suggestion; check it if the document contains several languages.'
    else:
        words = set(re.findall(r'[a-z]+', text.lower()))
        ranked = sorted(((len(words & vocabulary), lang) for lang, vocabulary in ROMAN.items()), reverse=True)
        if ranked[0][0] >= 3 and ranked[0][0] > ranked[1][0]:
            selected = ranked[0][1]
            reason = 'Possible Romanized language based on word clues. Please confirm.'
        else:
            selected = 'en'
            reason = ('Using English as a fallback. Romanized or unsupported languages may need manual selection.'
                      if counts else 'Add text to get a suggestion.')
    return {'language': selected, 'name': LANGUAGES[selected], 'reason': reason}
