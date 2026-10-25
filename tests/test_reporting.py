from io import BytesIO
from pypdf import PdfReader
import pytest
from pydantic import ValidationError
from detector.reporting import ReportRequest, render_report


def payload(text='English हिन्दी ಕನ್ನಡ தமிழ் A* O(b^m).', count=1):
    return dict(language='en', label='human', scores={'human':.99, 'ai':.01},
                word_count=20 * count, chunk_count=count, model_version='document-v3',
                encoder_variant='fully fine-tuned', warnings=['High scores can be wrong.'],
                segments=[dict(index=i+1,text=text,word_count=20,token_count=40,label='human',ai_score=.01)
                          for i in range(count)])


def test_multilingual_pdf_and_long_report_include_every_segment():
    pdf = render_report(ReportRequest.model_validate(payload(count=20)))
    reader = PdfReader(BytesIO(pdf))
    text = '\n'.join(page.extract_text() for page in reader.pages)
    assert len(reader.pages) > 2
    assert 'research' not in text.lower()
    assert 'Segment 01' not in reader.pages[0].extract_text()
    assert 'Segment-by-segment analysis' in reader.pages[1].extract_text()
    assert 'Segment 20' in text and 'Authorship inconclusive' in text
    # Shaped Indic clusters may add an unmapped auxiliary glyph in pypdf's
    # extraction. The rendered PDF was checked separately; remove reader C0 artifacts.
    text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\t')
    for word in ['हिन्दी', 'ಕನ್ನಡ', 'தமிழ்', 'O(b^m)']:
        assert word in text


def test_unsupported_glyphs_and_control_characters_do_not_break_export():
    pdf = render_report(ReportRequest.model_validate(payload('hello 😀\x00 <b>literal</b> ' + 'x'*2000)))
    text = '\n'.join(p.extract_text() for p in PdfReader(BytesIO(pdf)).pages)
    assert 'Font coverage' in text and '<b>literal</b>' in text


def test_report_limits_and_nonfinite_scores_are_rejected():
    data=payload();data['scores']['ai']=float('nan')
    with pytest.raises(ValidationError): ReportRequest.model_validate(data)
    data=payload();data['chunk_count']=2
    with pytest.raises(ValidationError): ReportRequest.model_validate(data)
    data=payload('x'*100000, count=11)
    with pytest.raises(ValidationError): ReportRequest.model_validate(data)


@pytest.mark.parametrize('reason,scores', [('low_score',{'gemini':.3,'muse':.5,'sarvam':.2}),('insufficient_text',{})])
def test_unknown_family_pdf(reason,scores):
    data=payload()
    data['label']='ai'
    data['family']={'label':'unknown','reason':reason,'scores':scores}
    pdf=render_report(ReportRequest.model_validate(data))
    text='\n'.join(page.extract_text() for page in PdfReader(BytesIO(pdf)).pages)
    assert 'Unknown / no reliable family match' in text
    assert 'AI-leaning model signal' in text
    assert 'Closest trained family:' not in text
