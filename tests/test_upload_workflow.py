import json
import pytest
from detector.language import suggest_language
from tests.test_app import client, PASSAGE


def test_independent_preview_preserves_record_boundaries_and_ids(client):
    data='\n'.join(json.dumps(row) for row in [dict(id='a',text=PASSAGE,label='ai'),dict(id='b',content='Short record',label='human')])
    response=client.post('/api/preview-file',data={'jsonl_mode':'independent'},files={'file':('records.jsonl',data)})
    assert response.status_code==200
    docs=response.json()['documents']
    assert len(docs)==2 and docs[0]['text']==PASSAGE and docs[1]['text']=='Short record'
    assert docs[0]['name']=='Line 1 · ID a'
    assert 'scores' not in docs[0]  # Preview does not classify or drop short records.
    combined=client.post('/api/preview-file',files={'file':('records.jsonl',data)}).json()['documents']
    assert len(combined)==1 and combined[0]['text']==PASSAGE+'\n\nShort record'


def test_preview_extracts_without_ready_model(client):
    from detector.app import app
    detector=app.state.detector
    try:
        app.state.detector=None
        response=client.post('/api/preview-file',files={'file':('example.txt',PASSAGE)})
        assert response.status_code==200
        assert response.json()['documents'][0]['text']==PASSAGE
    finally:
        app.state.detector=detector


def test_preview_limits_and_malformed_input(client):
    many='\n'.join(json.dumps({'text':'one record'}) for _ in range(101))
    response=client.post('/api/preview-file',data={'jsonl_mode':'independent'},files={'file':('many.jsonl',many)})
    assert response.status_code==422 and '100' in response.json()['detail']
    assert client.post('/api/preview-file',files={'file':('bad.jsonl','{broken')}).status_code==422
    assert client.post('/api/preview-file',files={'file':('bad.exe','hello')}).status_code==415
    assert client.post('/api/preview-file',data={'jsonl_mode':'invalid'},files={'file':('x.txt',PASSAGE)}).status_code==422


@pytest.mark.parametrize('text,expected',[
    ('यह एक हिंदी भाषा का उदाहरण है जिसमें बहुत सारे शब्द शामिल हैं।','hi'),
    ('தமிழ் மொழியில் எழுதப்பட்ட ஒரு எடுத்துக்காட்டு இது','ta'),
    ('ಇದು ಕನ್ನಡ ಭಾಷೆಯಲ್ಲಿ ಬರೆಯಲಾದ ಒಂದು ಉದಾಹರಣೆಯಾಗಿದೆ','kn'),
    ('This is an English paragraph about the weather today.','en'),
    ('mujhe nahi pata hai lekin this example has some mixed words','hi_en'),
    ('enakku romba nalla this example is useful','ta_en'),
    ('nanage tumba beku this example is useful','kn_en'),
    ('','en'),
])
def test_language_hints(text,expected):
    result=suggest_language(text)
    assert result['language']==expected and result['reason']


def test_latin_fallback_is_explicit_and_mixed_script_is_suggested():
    assert 'fallback' in suggest_language('bonjour les amis')['reason']
    assert suggest_language('यह हिंदी पाठ है and this is a long English explanation with several words')['language']=='hi_en'


@pytest.mark.parametrize('suffix', ['pdf','docx','doc'])
def test_document_preview_matches_extraction(client, suffix):
    from tests import document_fixtures
    source='This example keeps A* and O(b^m) notation in the preview.'
    response=client.post('/api/preview-file',files={'file':('notes.'+suffix,getattr(document_fixtures,suffix)(source))})
    assert response.status_code==200
    doc=response.json()['documents'][0]
    assert doc['text']==source and doc['notes']
