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
