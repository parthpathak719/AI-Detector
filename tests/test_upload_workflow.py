import json
import pytest
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


@pytest.mark.parametrize('suffix', ['pdf','docx','doc'])
def test_document_preview_matches_extraction(client, suffix):
    from tests import document_fixtures
    source='This example keeps A* and O(b^m) notation in the preview.'
    response=client.post('/api/preview-file',files={'file':('notes.'+suffix,getattr(document_fixtures,suffix)(source))})
    assert response.status_code==200
    doc=response.json()['documents'][0]
    assert doc['text']==source and doc['notes']
