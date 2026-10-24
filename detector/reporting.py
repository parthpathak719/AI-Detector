"""Bounded, in-memory PDF exports with embedded multilingual fonts."""
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
import unicodedata

from fpdf import FPDF
from pydantic import BaseModel, Field, model_validator
from detector.text import LANGUAGES

Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
FONT_DIR = Path(__file__).parent / 'fonts'


class Scores(BaseModel):
    human: Score
    ai: Score


class Segment(BaseModel):
    index: int = Field(ge=1, le=1024)
    text: str = Field(max_length=100000)
    word_count: int = Field(ge=0, le=20000)
    token_count: int = Field(ge=0, le=128)
    label: Literal['human', 'ai']
    ai_score: Score


class Family(BaseModel):
    label: Literal['gemini', 'muse', 'sarvam']
    scores: dict[Literal['gemini', 'muse', 'sarvam'], Score]


class ReportRequest(BaseModel):
    source_name: str = Field(default='Pasted text', max_length=240)
    language: Literal['en', 'hi', 'ta', 'kn', 'hi_en', 'ta_en', 'kn_en']
    label: Literal['human', 'ai']
    scores: Scores
    word_count: int = Field(ge=0, le=20000)
    chunk_count: int = Field(ge=1, le=1024)
    model_version: str = Field(max_length=120)
    encoder_variant: str = Field(max_length=120)
    family: Family | None = None
    warnings: list[Annotated[str, Field(max_length=2000)]] = Field(default_factory=list, max_length=30)
    segments: list[Segment] = Field(min_length=1, max_length=1024)

    @model_validator(mode='after')
    def bound_report(self):
        if len(self.segments) != self.chunk_count:
            raise ValueError('Report segment count does not match its segments.')
        if sum(len(s.text) for s in self.segments) > 200000:
            raise ValueError('Report text is too long.')
        return self


class ReportPDF(FPDF):
    def footer(self):
        self.set_y(-14)
        self.set_font('NotoSans', size=9)
        self.set_text_color(100, 105, 120)
        self.cell(0, 5, f'Origin | Research report | Page {self.page_no()}', align='C')


def render_report(report: ReportRequest) -> bytes:
    pdf = ReportPDF()
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(True, margin=21)
    fonts = ['NotoSans', 'NotoSansDevanagari', 'NotoSansTamil', 'NotoSansKannada']
    for font in fonts:
        pdf.add_font(font, fname=str(FONT_DIR / (font + '.ttf')))
    pdf.set_font('NotoSans', size=11)
    pdf.set_fallback_fonts(fonts[1:])
    pdf.set_text_shaping(True)
    supported = set().union(*(font.cmap.keys() for font in pdf.fonts.values()))
    missing = False

    def safe(text):
        nonlocal missing
        result = []
        for char in text:
            if char in '\n\t':
                result.append('\n' if char == '\n' else '    ')
            elif unicodedata.category(char) in ('Cc', 'Cs'):
                continue
            elif ord(char) in supported or char in '\u200c\u200d':
                result.append(char)
            else:
                missing = True
                result.append('\ufffd')
        return ''.join(result)

    def paragraph(text, size=11, color=(35, 41, 55)):
        pdf.set_font('NotoSans', size=size)
        pdf.set_text_color(*color)
        pdf.multi_cell(0, 6, safe(text), new_x='LMARGIN', new_y='NEXT', wrapmode='WORD')

    def heading(text):
        if pdf.get_y() > pdf.h - 45:
            pdf.add_page()
        pdf.ln(5)
        paragraph(text, 14, (85, 67, 150))
        pdf.ln(2)

    pdf.set_title('Origin — Text analysis report')
    pdf.set_author('Origin research application')
    pdf.add_page()
    paragraph('ORIGIN / ANALYSIS REPORT', 18, (85, 67, 150))
    pdf.ln(4)
    paragraph('Generated: ' + datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'), 10)
    paragraph('Source: ' + report.source_name)
    paragraph(f'Language: {LANGUAGES[report.language]} | Normalized words: {report.word_count:,} | Segments: {report.chunk_count}')
    paragraph(f'Model: {report.model_version} | Encoder: {report.encoder_variant}', 10)
    heading('Summary')
    paragraph('Authorship inconclusive' if report.chunk_count > 1 else f'{report.label.capitalize()}-leaning model signal', 14)
    paragraph(f'Human model score: {report.scores.human:.1%}   |   AI model score: {report.scores.ai:.1%}')
    paragraph('Model scores are not authorship confidence, proof of origin, or percentages of AI-written text. This export is not independently authenticated.', 10)
    if report.family:
        heading('Closest trained family: ' + report.family.label.capitalize())
        paragraph(' | '.join(f'{name.capitalize()}: {value:.1%}' for name, value in report.family.scores.items()))
        paragraph('Comparison is limited to Gemini, Muse and Sarvam; unknown generators may resemble these families.', 10)
    if report.warnings:
        heading('Interpretation and extraction notes')
        for warning in dict.fromkeys(report.warnings):
            paragraph('- ' + warning, 10)
    heading('Segment analysis — extracted source text')
    paragraph('Source notation and PDF layout may differ from the original document. Scores use normalized model input.', 10)
    for segment in report.segments:
        heading(f'Segment {segment.index:02d} | {segment.label.capitalize()}-leaning | AI score {segment.ai_score:.1%}')
        paragraph(f'{segment.word_count} normalized words | {segment.token_count} model tokens', 9)
        paragraph(segment.text)
    if missing:
        heading('Font coverage note')
        paragraph('Some characters outside the bundled fonts were replaced with the replacement symbol. Refer to the original source for those characters.', 10)
    return bytes(pdf.output())
