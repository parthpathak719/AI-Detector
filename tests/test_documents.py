from io import BytesIO
from zipfile import ZipFile
import pytest
from detector.documents import extract_document
from tests.document_fixtures import doc, docx, pdf


def test_word_formats_preserve_indic_text():
    text = 'आज मैं अपने दोस्तों के साथ बाजार गया ಕನ್ನಡ தமிழ்'
    for suffix, build in [('doc', doc), ('docx', docx)]:
        extracted, _ = extract_document('text.' + suffix, build(text))
        assert extracted == text


def test_invalid_and_unsafe_documents_fail_clearly():
    for name, content in [('a.doc', b'not a word file'), ('a.docx', b'not a zip'), ('a.pdf', b'broken'), ('a.exe', b'MZ')]:
        with pytest.raises(ValueError):
            extract_document(name, content)
    out = BytesIO()
    with ZipFile(out, 'w') as z:
        z.writestr('word/document.xml', '<!DOCTYPE body [<!ENTITY x "content">]><body>&x;</body>')
    with pytest.raises(ValueError, match='XML declarations'):
        extract_document('entities.docx', out.getvalue())


def test_encrypted_pdf_is_not_silently_analyzed():
    from pypdf import PdfReader, PdfWriter
    writer = PdfWriter()
    writer.append(PdfReader(BytesIO(pdf('this is some readable text in a PDF file'))))
    writer.encrypt('password')
    stream = BytesIO(); writer.write(stream)
    with pytest.raises(ValueError, match='Password-protected'):
        extract_document('locked.pdf', stream.getvalue())


def test_partly_scanned_pdf_reports_missing_pages():
    from pypdf import PdfReader, PdfWriter
    writer = PdfWriter()
    writer.append(PdfReader(BytesIO(pdf('This is a page that contains real readable text.'))))
    writer.add_blank_page(width=612, height=792)
    stream = BytesIO(); writer.write(stream)
    text, notes = extract_document('mixed.pdf', stream.getvalue())
    assert 'real readable text' in text
    assert notes and '1 PDF page(s)' in notes[0]


def test_pdf_preserves_notation_and_reports_layout_limitations():
    source = 'GBFS O(b^m) O(b^m). A* O(b^d). There are 4 cells and 12 actions.'
    text, notes = extract_document('table.pdf', pdf(source))
    assert 'A* O(b^d)' in text
    assert '4 cells' in text
    assert any('superscripts' in note for note in notes)


def test_jsonl_extracts_only_text_fields_in_order():
    import json
    records = [{'id': 1, 'label': 'human', 'text': 'पहला पाठ यहाँ है।'},
               {'content': 'Second passage.', 'metadata': 'DO NOT ANALYZE'}, 'Third passage.']
    raw = '\ufeff' + '\n\n'.join(json.dumps(r, ensure_ascii=False) for r in records)
    text, notes = extract_document('records.JSONL', raw.encode('utf-8'))
    assert text == 'पहला पाठ यहाँ है।\n\nSecond passage.\n\nThird passage.'
    assert 'DO NOT ANALYZE' not in text and 'human' not in text
    assert '3 JSONL record(s)' in notes[0]
    assert 'not separate predictions' in notes[0]


@pytest.mark.parametrize('content, message', [
    (b'{"text":"valid"}\nnot json', 'line 2'),
    (b'{"label":"ai"}', 'line 1'),
    (b'{"text":123}', 'line 1'),
    (b'[]', 'line 1'),
    (b'{"text":""}', 'line 1'),
    (b'\n\n', 'no text records'),
    (b'{"text":"bad\\u0000text"}', 'null characters'),
])
def test_invalid_jsonl_has_actionable_errors(content, message):
    with pytest.raises(ValueError, match=message):
        extract_document('data.jsonl', content)


def test_jsonl_extracted_text_limit():
    import json
    with pytest.raises(ValueError, match='exceeds'):
        extract_document('data.jsonl', json.dumps({'text':'a' * 100001}).encode())
