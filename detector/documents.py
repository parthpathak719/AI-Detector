"""Bounded plain-text extraction from byte buffers. Documents are never executed.

DOC support covers Word 97–2003 compound documents with a CLX piece table.
Encrypted files, image-only PDFs and older Word binary layouts are rejected.
"""
from io import BytesIO
from pathlib import Path
import re
import struct
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import olefile
import pdfplumber
from pypdf import PdfReader
from detector.text import MAX_CHARS

MAX_FILE_BYTES = 10 * 1024 * 1024
SUPPORTED = {".txt", ".md", ".pdf", ".docx", ".doc"}


def bounded(text):
    if len(text) > MAX_CHARS:
        raise ValueError(f"The extracted document exceeds {MAX_CHARS:,} characters. Please split it into smaller documents.")
    return text


def pdf_text(content):
    reader = PdfReader(BytesIO(content))
    if reader.is_encrypted:
        raise ValueError("Password-protected PDFs are not supported. Upload an unlocked copy.")
    if len(reader.pages) > 300:
        raise ValueError("Please upload a PDF with no more than 300 pages.")
    # Validate inflated content before asking the spacing-aware parser to work.
    for page in reader.pages:
        contents = page.get_contents()
        if contents is not None and len(contents.get_data()) > 8 * 1024 * 1024:
            raise ValueError("A PDF page is too complex to extract. Export it as a simpler PDF or text file.")
    pages, size, empty = [], 0, 0
    with pdfplumber.open(BytesIO(content)) as document:
        for page in document.pages:
            # Content-stream flow preserves the two-column reading order in
            # academic PDFs; geometric spacing separates adjacent font runs.
            text = page.extract_text(use_text_flow=True, x_tolerance=1, y_tolerance=3) or ""
            empty += not bool(text.strip())
            size += len(text) + 2
            if size > MAX_CHARS:
                raise ValueError(f"The PDF exceeds {MAX_CHARS:,} extracted characters. Please split it into smaller documents.")
            pages.append(text)
            page.close()
    if not any(p.strip() for p in pages):
        raise ValueError("This PDF has no extractable text. For a scanned PDF, run OCR first and upload the searchable PDF or text.")
    notes = [f"{empty} PDF page(s) contained no extractable text; images and scans were not analyzed."] if empty else []
    notes.append("PDF extraction can flatten tables, diagrams, superscripts and subscripts. Review the extracted text; use the original PDF to check mathematical notation.")
    return "\n\n".join(pages), notes


def docx_text(content):
    with ZipFile(BytesIO(content)) as archive:
        if sum(i.file_size for i in archive.infolist()) > 50 * 1024 * 1024:
            raise ValueError("The expanded Word document is too large.")
        info = archive.getinfo("word/document.xml")
        if info.file_size > 8 * 1024 * 1024:
            raise ValueError("The Word document body is too large.")
        xml = archive.read(info)
    if b"<!DOCTYPE" in xml.upper() or b"<!ENTITY" in xml.upper():
        raise ValueError("Unsupported XML declarations in Word document.")
    root = ET.fromstring(xml)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    body = root.find(ns + "body")
    if body is None:
        raise ValueError("The Word document has no document body.")
    parts = []
    # Paragraph order includes table cells; only main-body text is analyzed.
    for paragraph in body.iter(ns + "p"):
        parts.append("".join(node.text or "" if node.tag == ns + "t" else " "
                             for node in paragraph.iter() if node.tag in (ns + "t", ns + "tab", ns + "br")))
    return bounded("\n".join(parts)), ["Word extraction covers main-body text and tables; images, headers, footers and footnotes are excluded."]


def doc_text(content):
    with olefile.OleFileIO(BytesIO(content)) as ole:
        if not ole.exists("WordDocument"):
            raise ValueError("This is not a supported Word .doc file. Save it as .docx and try again.")
        word = ole.openstream("WordDocument").read()
        u16 = lambda offset: struct.unpack_from("<H", word, offset)[0]
        u32 = lambda offset: struct.unpack_from("<I", word, offset)[0]
        if u16(0) != 0xA5EC or u16(2) < 0x00C1:
            raise ValueError("This older .doc format is not supported. Save it as .docx first.")
        flags = u16(10)
        if flags & (0x0100 | 0x8000):
            raise ValueError("Password-protected Word documents are not supported. Upload an unlocked copy.")
        # FIB variable-length sections (MS-DOC FibRgLw97 / FibRgFcLcb97).
        lw_count_offset = 34 + 2 * u16(32)
        lw_offset = lw_count_offset + 2
        body_chars = u32(lw_offset + 12)
        fc_count_offset = lw_offset + 4 * u16(lw_count_offset)
        if u16(fc_count_offset) < 34:
            raise ValueError("The .doc file has no supported text table. Save it as .docx.")
        pair_offset = fc_count_offset + 2 + 33 * 8
        clx_offset, clx_size = u32(pair_offset), u32(pair_offset + 4)
        table = ole.openstream("1Table" if flags & 0x0200 else "0Table").read()
        if clx_offset + clx_size > len(table):
            raise ValueError("The Word document's text table is damaged.")
        clx = table[clx_offset:clx_offset + clx_size]
        pos = 0
        while pos < len(clx) and clx[pos] == 1:
            pos += 3 + struct.unpack_from("<H", clx, pos + 1)[0]
        if pos >= len(clx) or clx[pos] != 2:
            raise ValueError("The .doc file has no supported piece table. Save it as .docx.")
        size = struct.unpack_from("<I", clx, pos + 1)[0]
        pieces = clx[pos + 5:pos + 5 + size]
        if size < 4 or (size - 4) % 12 or len(pieces) != size:
            raise ValueError("The Word document's piece table is damaged.")
        count = (size - 4) // 12
        cp = struct.unpack_from(f"<{count + 1}I", pieces)
        if cp[0] != 0 or any(b < a for a, b in zip(cp, cp[1:])):
            raise ValueError("Invalid Word document text offsets.")
        parts = []
        for i in range(count):
            if cp[i] >= body_chars:
                break
            chars = min(cp[i + 1], body_chars) - cp[i]
            fc = struct.unpack_from("<I", pieces, 4 * (count + 1) + i * 8 + 2)[0]
            compressed = bool(fc & 0x40000000)
            offset = (fc & 0x3FFFFFFF) // (2 if compressed else 1)
            end = offset + chars * (1 if compressed else 2)
            if end > len(word):
                raise ValueError("The Word document's text data is incomplete.")
            parts.append(word[offset:end].decode("cp1252" if compressed else "utf-16-le", errors="replace"))
        text = "".join(parts)
        # Keep displayed field results, not hidden field instructions.
        text = re.sub(r"\x13[^\x14\x15]*\x14", "", text)
        text = re.sub(r"\x13[^\x15]*\x15", "", text)
        text = text.replace("\r", "\n").replace("\x07", "\t")
        text = "".join(c for c in text if ord(c) >= 32 or c in "\n\t")
        return bounded(text), ["Legacy Word extraction covers main-body text; embedded objects, headers and footnotes are excluded."]


def extract_document(filename, content):
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError("Choose a .txt, .md, .pdf, .docx or .doc file.")
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("Upload a file of up to 10 MB.")
    try:
        if suffix == ".pdf":
            text, notes = pdf_text(content)
        elif suffix == ".docx":
            text, notes = docx_text(content)
        elif suffix == ".doc":
            text, notes = doc_text(content)
        else:
            text, notes = content.decode("utf-8-sig"), []
            if "\x00" in text:
                raise ValueError("This file contains binary data. Upload a plain-text document.")
    except UnicodeDecodeError as e:
        raise ValueError("Save this text file with UTF-8 encoding and upload it again.") from e
    except ValueError:
        raise
    except Exception as e:
        raise ValueError("The document could not be read. It may be damaged or use an unsupported format. Try saving it as .docx or plain text.") from e
    if not text.strip():
        raise ValueError("No text was found in this document. Images and scanned pages require OCR first.")
    return bounded(text), notes
