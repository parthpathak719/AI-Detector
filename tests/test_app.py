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
    assert home.status_code == 200 and 'Origin' in home.text
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


@pytest.mark.parametrize('text,language', [('', 'en'), ('hi', 'en'), ('123 !!! 😀', 'en'), (PASSAGE, 'fr'), ('word ' * 151, 'en'), ('a' * 12001, 'en')])
def test_invalid_text(client, text, language):
    assert client.post('/api/analyze', json={'text': text, 'language': language}).status_code == 422


def test_invalid_files_and_size(client):
    for name, content, expected in [('a.pdf', b'%PDF', 415), ('a.txt', b'\xff\xfe', 422), ('a.txt', b'\x00binary', 422), ('a.txt', b'a' * 48001, 413), ('a.txt', b'a' * 70000, 413)]:
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
