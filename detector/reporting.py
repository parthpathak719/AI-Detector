"""Bounded, in-memory PDF exports with embedded multilingual fonts."""
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
import unicodedata

from fpdf import FPDF
from pydantic import BaseModel, Field, model_validator
from detector.text import LANGUAGES, MAX_CHARS, MAX_WORDS, MAX_SEGMENTS

Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
FONT_DIR = Path(__file__).parent / 'fonts'


class Scores(BaseModel):
    human: Score
    ai: Score


class Segment(BaseModel):
    index: int = Field(ge=1, le=MAX_SEGMENTS)
    text: str = Field(max_length=MAX_CHARS)
    word_count: int = Field(ge=0, le=MAX_WORDS)
    token_count: int = Field(ge=0, le=128)
    label: Literal['human', 'ai']
    ai_score: Score


class Family(BaseModel):
    reason: Literal['matched', 'low_score', 'insufficient_text'] | None = None
    label: Literal['gemini', 'muse', 'sarvam', 'unknown']
    scores: dict[Literal['gemini', 'muse', 'sarvam'], Score]


class ReportRequest(BaseModel):
    source_name: str = Field(default='Pasted text', max_length=240)
    language: Literal['en', 'hi', 'ta', 'kn', 'hi_en', 'ta_en', 'kn_en']
    label: Literal['human', 'ai']
    scores: Scores
    word_count: int = Field(ge=0, le=MAX_WORDS)
    chunk_count: int = Field(ge=1, le=MAX_SEGMENTS)
    model_version: str = Field(max_length=120)
    encoder_variant: str = Field(max_length=120)
    family: Family | None = None
    warnings: list[Annotated[str, Field(max_length=2000)]] = Field(default_factory=list, max_length=30)
    segments: list[Segment] = Field(min_length=1, max_length=MAX_SEGMENTS)

    @model_validator(mode='after')
    def bound_report(self):
        if len(self.segments) != self.chunk_count:
            raise ValueError('Report segment count does not match its segments.')
        if sum(len(s.text) for s in self.segments) > 2 * MAX_CHARS:
            raise ValueError('Report text is too long.')
        return self


class ReportPDF(FPDF):
    def header(self):
        self.set_xy(18, 18)
        self.set_font('NotoSans', size=10)
        self.set_text_color(85, 67, 150)
        self.cell(0, 6, 'LEXORA  /  TEXT ANALYSIS', new_x='LMARGIN', new_y='NEXT')
        self.set_draw_color(222, 225, 234)
        self.line(18, 27, self.w - 18, 27)
        self.set_y(34)

    def footer(self):
        self.set_y(-16)
        self.set_draw_color(222, 225, 234)
        self.line(18, self.get_y(), self.w - 18, self.get_y())
        self.set_font('NotoSans', size=9)
        self.set_text_color(100, 105, 120)
        self.cell(0, 9, f'Lexora  /  Analysis report                                      {self.page_no()}', align='R')


def render_report(report: ReportRequest) -> bytes:
    pdf = ReportPDF()
    pdf.set_margins(18, 34, 18)
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

    def paragraph(text, size=11, color=(35, 41, 55), line_height=6):
        pdf.set_font('NotoSans', size=size)
        pdf.set_text_color(*color)
        pdf.multi_cell(0, line_height, safe(text), new_x='LMARGIN', new_y='NEXT', wrapmode='WORD', align='L')

    def heading(text):
        if pdf.get_y() > pdf.h - 55:
            pdf.add_page()
        pdf.ln(7)
        paragraph(text, 15, (85, 67, 150), 8)
        pdf.ln(3)

    def score_cards():
        top = pdf.get_y()
        width = (pdf.epw - 8) / 2
        for i, (name, score) in enumerate([('Human', report.scores.human), ('AI', report.scores.ai)]):
            x = pdf.l_margin + i * (width + 8)
            pdf.set_fill_color(245, 245, 250)
            pdf.rect(x, top, width, 30, style='F')
            pdf.set_xy(x + 6, top + 4)
            pdf.set_font('NotoSans', size=10)
            pdf.set_text_color(100, 105, 120)
            pdf.cell(width - 12, 6, name + ' model score')
            pdf.set_xy(x + 6, top + 13)
            pdf.set_font('NotoSans', size=22)
            pdf.set_text_color(35, 41, 55)
            pdf.cell(width - 12, 12, f'{score:.1%}')
        pdf.set_xy(pdf.l_margin, top + 37)

    pdf.set_title('Lexora — Text analysis report')
    pdf.set_author('Lexora')
    pdf.add_page()
    paragraph('Text analysis report', 25, line_height=12)
    pdf.ln(5)
    paragraph('Source', 9, (100, 105, 120), 5)
    paragraph(report.source_name, 12)
    pdf.ln(4)
    paragraph(f'{LANGUAGES[report.language]}   /   {report.word_count:,} normalized words   /   {report.chunk_count} segments', 10)
    paragraph('Generated ' + datetime.now(timezone.utc).strftime('%d %b %Y, %H:%M UTC'), 9, (100, 105, 120))
    heading('Overall result')
    paragraph('Authorship inconclusive' if report.chunk_count > 1 else f"{'AI' if report.label == 'ai' else 'Human'}-leaning model signal", 18, line_height=9)
    pdf.ln(5)
    score_cards()
    paragraph('Scores describe model patterns. They are not proof of authorship or the percentage of text written by AI.', 10)
    if report.family:
        heading('Family attribution')
        if report.family.label == 'unknown':
            paragraph('Unknown / no reliable family match', 12)
            paragraph('Not enough eligible text for attribution.' if report.family.reason == 'insufficient_text'
                      else 'None of the trained families reached the attribution threshold. This does not prove an unseen generator was used.', 10)
            if report.family.scores:
                paragraph('Known-family comparison scores:', 10)
        else:
            paragraph('Closest trained family: ' + report.family.label.capitalize(), 12)
        pdf.ln(3)
        for name, value in report.family.scores.items():
            paragraph(f'{name.capitalize():<12}    {value:.1%}', 11)
        pdf.ln(3)
        paragraph('Comparison is limited to Gemini, Muse and Sarvam. Unknown generators can resemble these families.', 10)
    heading('Report contents')
    paragraph(f'The following pages contain all {report.chunk_count} segments, with their extracted text and individual model scores. Analysis and extraction notes appear at the end.', 10)

    pdf.add_page()
    paragraph('Segment-by-segment analysis', 21, line_height=10)
    paragraph('Original extracted text with an individual AI model score for each segment.', 10, (100, 105, 120))
    pdf.ln(8)
    for segment in report.segments:
        text = safe(segment.text)
        pdf.set_font('NotoSans', size=11)
        height = pdf.multi_cell(pdf.epw - 12, 6.5, text, dry_run=True, output='HEIGHT')
        # Keep ordinary segments together. Very long excerpts flow across pages.
        required = height + 32
        if pdf.get_y() + min(required, 85) > pdf.h - pdf.b_margin or (
            required <= pdf.h - pdf.t_margin - pdf.b_margin and
            pdf.get_y() + required > pdf.h - pdf.b_margin
        ):
            pdf.add_page()
        top = pdf.get_y()
        pdf.set_fill_color(239, 237, 247)
        pdf.rect(pdf.l_margin, top, pdf.epw, 12, style='F')
        pdf.set_xy(pdf.l_margin + 4, top + 2)
        pdf.set_font('NotoSans', size=12)
        pdf.set_text_color(85, 67, 150)
        pdf.cell(40, 8, f'Segment {segment.index:02d}')
        pdf.set_font('NotoSans', size=10)
        pdf.cell(pdf.epw - 48, 8, f"{'AI' if segment.label == 'ai' else 'Human'}-leaning  /  AI score {segment.ai_score:.1%}", align='R')
        pdf.set_xy(pdf.l_margin, top + 15)
        paragraph(f'{segment.word_count} normalized words  /  {segment.token_count} model tokens', 9, (100, 105, 120), 5)
        pdf.ln(3)
        pdf.set_x(pdf.l_margin + 6)
        pdf.set_font('NotoSans', size=11)
        pdf.set_text_color(35, 41, 55)
        pdf.multi_cell(pdf.epw - 12, 6.5, text, new_x='LMARGIN', new_y='NEXT', align='L')
        pdf.ln(9)

    pdf.add_page()
    paragraph('Analysis notes', 21, line_height=10)
    heading('Model details')
    paragraph('Model: ' + report.model_version, 10)
    paragraph('Encoder: ' + report.encoder_variant, 10)
    heading('How to read this report')
    paragraph('Model scores are not calibrated authorship probabilities. This exported report is not independently authenticated.', 10)
    paragraph('Scores use normalized model input. Extracted PDF text may flatten tables, diagrams or mathematical notation; consult the original document when needed.', 10)
    if report.warnings:
        heading('Extraction and interpretation')
        for number, warning in enumerate(dict.fromkeys(report.warnings), 1):
            paragraph(f'{number}. {warning}', 10)
            pdf.ln(3)
    if missing:
        heading('Font coverage')
        paragraph('Characters outside the bundled fonts are shown with a replacement symbol. Refer to the original source for those characters.', 10)
    return bytes(pdf.output())
