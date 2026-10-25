"""HTTP integration tests use the actual trained models and local MuRIL encoder."""
import pytest
from fastapi.testclient import TestClient
from detector.app import app

PASSAGE = 'I went back to the little bookshop near our old house yesterday. The owner still remembered the mystery novels I used to borrow after school.'


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        assert c.get('/api/health').status_code == 200, 'Install encoder and train artifacts before integration tests'
        yield c


def test_home_and_real_prediction(client):
    home = client.get('/')
    assert home.status_code == 200 and 'Lexora' in home.text
    assert "frame-ancestors 'none'" in home.headers['content-security-policy']
    response = client.post('/api/analyze', json={'text': PASSAGE, 'language': 'en'})
    assert response.status_code == 200
    data = response.json()
    assert data['label'] in ('human', 'ai')
    assert sum(data['scores'].values()) == pytest.approx(1)
    assert data['task'] == 'task1_binary'
    assert 0 <= data['score'] <= 1 and data['token_count'] <= 128
    assert data['warnings']
    if data['label'] == 'human':
        assert data['family'] is None
    else:
        assert set(data['family']['scores']) == {'gemini', 'muse', 'sarvam'}


def test_upload_matches_paste(client):
    pasted = client.post('/api/analyze', json={'text': PASSAGE, 'language': 'en'}).json()
    uploaded = client.post('/api/analyze-file', data={'language': 'en'}, files={'file': ('passage.txt', PASSAGE.encode(), 'text/plain')})
    assert uploaded.status_code == 200
    assert pasted == uploaded.json()


@pytest.mark.parametrize('language,text', [
    ('hi', 'आज मैं अपने दोस्तों के साथ बाजार गया और वहां हमने बहुत अच्छा खाना खाया'),
    ('ta', 'நான் இன்று என் நண்பர்களுடன் கடைக்கு சென்று ஒரு நல்ல புத்தகம் வாங்கினேன்'),
    ('kn', 'ನಾನು ಇಂದು ನನ್ನ ಸ್ನೇಹಿತರ ಜೊತೆ ಮನೆಗೆ ಹೋಗಿ ಒಂದು ಹೊಸ ಪುಸ್ತಕ ಓದಿದೆ'),
    ('hi_en', 'kal main apne friends ke saath market gaya tha aur wahan bahut maza aaya'),
    ('ta_en', 'innaiku naan friends kooda market ponen romba nalla irunthuchu'),
    ('kn_en', 'ivattu naanu friends jothe market ge hogidde tumba chennagittu'),
])
def test_supported_languages(client, language, text):
    response = client.post('/api/analyze', json={'text': text, 'language': language})
    assert response.status_code == 200, response.text
    assert response.json()['task'] == ('task3_codemix' if language.endswith('_en') else 'task1_binary')


@pytest.mark.parametrize('text,language', [('', 'en'), ('hi', 'en'), ('123 !!! 😀', 'en'), (PASSAGE, 'fr'), ('word ' * 50001, 'en'), ('a' * 500001, 'en')])
def test_invalid_text(client, text, language):
    assert client.post('/api/analyze', json={'text': text, 'language': language}).status_code == 422


def test_invalid_files_and_size(client):
    for name, content, expected in [('a.exe', b'MZ', 415), ('a.txt', b'\xff\xfe', 422), ('a.txt', b'\x00binary', 422), ('a.pdf', b'%PDF', 422), ('a.txt', b'a' * (10 * 1024 * 1024 + 1), 413), ('a.txt', b'a' * (11 * 1024 * 1024), 413)]:
        r = client.post('/api/analyze-file', data={'language': 'en'}, files={'file': (name, content)})
        assert r.status_code == expected, r.text


def test_unavailable_and_busy(client):
    saved = app.state.detector
    try:
        app.state.detector = None
        assert client.get('/api/health').status_code == 503
        assert client.post('/api/analyze', json={'text': PASSAGE, 'language': 'en'}).status_code == 503
    finally:
        app.state.detector = saved


@pytest.mark.parametrize('extension', ['pdf', 'docx', 'doc'])
def test_document_formats_match_pasted_text(client, extension):
    from tests import document_fixtures
    content = getattr(document_fixtures, extension)(PASSAGE)
    uploaded = client.post('/api/analyze-file', data={'language': 'en'}, files={'file': ('passage.' + extension, content)})
    assert uploaded.status_code == 200, uploaded.text
    pasted = client.post('/api/analyze', json={'text': PASSAGE, 'language': 'en'}).json()
    assert uploaded.json()['scores'] == pasted['scores']
    assert uploaded.json()['word_count'] == pasted['word_count']


def test_large_markdown_upload_analyzes_every_word(client):
    text = (PASSAGE + '\n\n') * 100
    response = client.post('/api/analyze-file', data={'language': 'en'}, files={'file': ('notes.md', text.encode())})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['chunk_count'] > 1
    assert data['word_count'] == len(text.split())
    assert sum(s['word_count'] for s in data['segments']) == data['word_count']
    assert all(s['token_count'] <= 128 for s in data['segments'])
    assert sum(data['scores'].values()) == pytest.approx(1)


def test_scanned_pdf_gets_actionable_error(client):
    from tests.document_fixtures import pdf
    r = client.post('/api/analyze-file', data={'language': 'en'}, files={'file': ('scan.pdf', pdf())})
    assert r.status_code == 422 and 'OCR' in r.json()['detail']


def test_real_tokenizer_chunking_has_complete_unicode_coverage(client):
    text = 'आज मैं अपने दोस्तों के साथ बाजार गया और वहां हमने बहुत अच्छा खाना खाया ' * 100
    cleaned = __import__('detector.text', fromlist=['clean_text']).clean_text(text)
    encoder = app.state.detector.encoder
    chunks = encoder.chunks(cleaned)
    assert len(chunks) > 1
    assert ' '.join(chunks) == cleaned
    assert all(encoder.token_count(c) <= 128 for c in chunks)


def test_encoder_variant_round_trip_matches_frozen_encoder(client, tmp_path):
    import numpy as np
    from safetensors.torch import save_file
    encoder = app.state.detector.encoder
    path = tmp_path / 'last_layer.safetensors'
    save_file({k: v.detach().cpu().contiguous() for k, v in encoder.model.encoder.layer[-1].state_dict().items()}, str(path))
    expected = encoder.encode([PASSAGE])
    try:
        encoder.configure_task('round_trip_test', path)
        actual = encoder.encode([PASSAGE], task='round_trip_test')
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
    finally:
        encoder.task_layers.pop('round_trip_test', None)


def test_jsonl_upload_matches_concatenated_text(client):
    import json
    text = PASSAGE + '\n\n' + PASSAGE
    raw = '\n'.join(json.dumps({'text': PASSAGE, 'label': label}) for label in ['ai', 'human'])
    uploaded = client.post('/api/analyze-file', data={'language': 'en'}, files={'file': ('data.jsonl', raw.encode())})
    assert uploaded.status_code == 200, uploaded.text
    result = uploaded.json()
    pasted = client.post('/api/analyze', json={'text': text, 'language': 'en'}).json()
    assert result['scores'] == pasted['scores']
    assert result['source_text'] == text
    assert any('2 JSONL record(s)' in note for note in result['warnings'])


def test_pdf_report_download_contains_results_and_source(client):
    from pypdf import PdfReader
    from io import BytesIO
    data = client.post('/api/analyze', json={'text': PASSAGE, 'language': 'en'}).json()
    response = client.post('/api/report.pdf', json={**data, 'source_name': 'example.txt'})
    assert response.status_code == 200, response.text[:200] if response.status_code != 200 else ''
    assert response.headers['content-type'] == 'application/pdf'
    assert 'lexora-analysis.pdf' in response.headers['content-disposition']
    assert response.headers['cache-control'] == 'no-store'
    assert response.content.startswith(b'%PDF-')
    text = '\n'.join(p.extract_text() for p in PdfReader(BytesIO(response.content)).pages)
    assert 'example.txt' in text and 'bookshop' in text and 'Segment 01' in text
    assert 'not calibrated authorship probabilities' in text


def test_pdf_export_rejects_incomplete_report_and_stale_assets_revalidate(client):
    assert client.post('/api/report.pdf', json={'label': 'human'}).status_code == 422
    assert client.get('/static/app.js?v=4').headers['cache-control'] == 'no-store'
    assert 'edit-source' not in client.get('/').text
