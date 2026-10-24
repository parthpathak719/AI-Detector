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
    def chunks(self, text):
        words = text.split()
        return [' '.join(words[i:i + 126]) for i in range(0, len(words), 126)]
    def encode(self, texts, task=None):
        return np.zeros((len(texts), 768))


class Classifier:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.calls = 0
    def predict_proba(self, texts, embeddings):
        self.calls += 1
        return np.tile(self.probabilities, (len(texts), 1))


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


def test_short_code_mixed_text_has_no_family_and_long_text_is_segmented():
    service = detector([.1, .9])
    assert service.predict('kal main market gaya', 'hi_en')['family'] is None
    result = service.predict('word ' * 300, 'en')
    assert result['chunk_count'] == 3
    assert result['word_count'] == 300
    assert result['family']['analyzed_segments'] == 3


def test_document_aggregation_weights_words_and_filters_human_segments():
    service = detector([.1, .9])
    class VariableClassifier:
        def predict_proba(self, texts, embeddings):
            return np.array([[.1, .9], [.8, .2]])
    service.artifacts['task1_binary']['model'] = VariableClassifier()
    result = service.predict('word ' * 150, 'en')
    assert result['scores']['ai'] == pytest.approx((126 * .9 + 24 * .2) / 150)
    assert result['ai_segments'] == 1
    assert result['family']['analyzed_segments'] == 1


def test_task_specific_encoder_is_used_for_attribution():
    service = detector([.1, .9])
    service.artifacts['task2_attribution']['metadata']['encoder'] = {'fine_tuned_last_layer': 'attribution.safetensors'}
    calls = []
    def encode(texts, task=None):
        calls.append(task)
        return np.zeros((len(texts), 768))
    service.encoder.encode = encode
    service.predict('this passage has enough words for a family prediction', 'en')
    assert calls == ['task1_binary', 'task2_attribution']


def test_segments_show_source_without_changing_normalized_model_input():
    text = 'RT @author: Algorithm Time Space\nGBFS O(b^m) O(b^m)\nA* O(b^d) (worst case).\nThe search uses 4 cells and costs 12 units.'
    service = detector([.993, .007])
    result = service.predict(text, 'en')
    assert result['segments'][0]['text'] == text
    assert result['segments'][0]['model_text'] == clean_text(text)
    assert 'A*' in result['segments'][0]['text']
    assert '4 cells' in result['segments'][0]['text']
    assert result['source_text'] == text
    assert result['scores']['human'] == pytest.approx(.993)


def test_high_model_score_does_not_claim_document_certainty():
    result = detector([.993, .007]).predict('A technical assignment sentence with formulas A* O(b^m).\n' * 40, 'en')
    assert result['assessment'] == 'document_model_signal'
    assert result['uncertain'] and not result['model_uncertain']
    assert result['label'] == 'human'  # No hard-coded AI override or score cap.
    assert result['score'] == pytest.approx(.993)
    assert any('not the probability' in w for w in result['warnings'])
    assert ' '.join(s['model_text'] for s in result['segments']) == clean_text(result['source_text'])
    for segment in result['segments']:
        assert segment['text'] == result['source_text'][segment['source_start']:segment['source_end']]


def test_source_alignment_keeps_compounds_and_multilingual_marks():
    from detector.text import source_excerpts
    text = 'RT @author: आज क्\u200dष ಕನ್ನಡ தமிழ் state-space **Hello**—WORLD! 123 https://example.com 😀'
    cleaned = clean_text(text)
    chunks = cleaned.split()  # Exercise a split inside a normalized compound.
    excerpts = source_excerpts(text, cleaned, chunks)
    assert any(e['text'] == 'state-space ' or 'state-space' in e['text'] for e in excerpts)
    assert all(e['text'] == text[e['source_start']:e['source_end']] for e in excerpts)
    assert excerpts[0]['source_start'] == 0
    assert excerpts[-1]['source_end'] == len(text)
