import numpy as np
import pytest
from detector.service import Detector
from detector.text import clean_text, validate_text


def test_cleaning_preserves_indic_marks_and_joiners():
    assert clean_text('RT @person: **Hello**—WORLD! 123 https://example.com 😀') == 'hello world'
    text = 'क्\u200dष ಕನ್ನಡ தமிழ்'
    assert clean_text(text) == text
    assert clean_text(clean_text('A—B, C!')) == clean_text('A—B, C!')


class Encoder:
    def token_count(self, text):
        return len(text.split()) + 2
    def encode(self, texts):
        return np.zeros((len(texts), 768))


class Classifier:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.calls = 0
    def predict_proba(self, texts, embeddings):
        self.calls += 1
        return np.array([self.probabilities])


def detector(binary):
    service = Detector.__new__(Detector)
    service.encoder = Encoder()
    service.artifacts = {
        'task1_binary': {'model': Classifier(binary), 'metadata': {'classes': ['human', 'ai']}},
        'task3_codemix': {'model': Classifier(binary), 'metadata': {'classes': ['human', 'ai']}},
        'task2_attribution': {'model': Classifier([.2, .5, .3]), 'metadata': {'classes': ['gemini', 'muse', 'sarvam']}},
    }
    return service


def test_attribution_is_conditional_and_uncertainty_is_explicit():
    text = 'this is a passage with enough words for family attribution'
    human = detector([.8, .2])
    assert human.predict(text, 'en')['family'] is None
    assert human.artifacts['task2_attribution']['model'].calls == 0
    ai = detector([.45, .55])
    result = ai.predict(text, 'hi_en')
    assert result['uncertain'] and result['family']['uncertain']
    assert result['family']['label'] == 'muse'
    assert ai.artifacts['task1_binary']['model'].calls == 0
    assert ai.artifacts['task3_codemix']['model'].calls == 1


def test_short_code_mixed_text_has_no_family_and_long_tokens_are_rejected():
    service = detector([.1, .9])
    assert service.predict('kal main market gaya', 'hi_en')['family'] is None
    service.encoder.token_count = lambda _: 129
    with pytest.raises(ValueError, match='128-token'):
        service.predict('there are enough words here to submit a passage', 'en')
