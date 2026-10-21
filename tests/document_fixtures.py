"""Small valid documents constructed in memory for parser/HTTP tests."""
from io import BytesIO
import struct
from xml.sax.saxutils import escape
from zipfile import ZipFile
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject


def docx(text):
    stream = BytesIO()
    with ZipFile(stream, 'w') as z:
        z.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>' + escape(text) + '</w:t></w:r></w:p></w:body></w:document>')
    return stream.getvalue()


def pdf(text=None):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    if text:
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        content = DecodedStreamObject()
        escaped = text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
        content.set_data(f'BT /F1 11 Tf 20 740 Td ({escaped}) Tj ET'.encode('ascii'))
        page[NameObject('/Contents')] = writer._add_object(content)
    out = BytesIO(); writer.write(out)
    return out.getvalue()


def doc(text):
    """Word 97 Unicode text in an OLE compound file, with regular FAT streams."""
    free, end = 0xFFFFFFFF, 0xFFFFFFFE
    header = bytearray(512)
    header[:8] = bytes.fromhex('d0cf11e0a1b11ae1')
    struct.pack_into('<HHHHH', header, 24, 0x3E, 3, 0xFFFE, 9, 6)
    struct.pack_into('<IIIIIIIII', header, 40, 0, 1, 0, 0, 4096, end, 0, end, 0)
    struct.pack_into('<109I', header, 76, 17, *([free] * 108))
    directory = bytearray(512)
    def entry(index, name, kind, sibling, child, start, size):
        base = index * 128
        encoded = (name + '\0').encode('utf-16-le')
        directory[base:base + len(encoded)] = encoded
        struct.pack_into('<HBBIII', directory, base + 64, len(encoded), kind, 1, free, sibling, child)
        struct.pack_into('<IQ', directory, base + 116, start, size)
    entry(0, 'Root Entry', 5, free, 1, end, 0)
    entry(1, 'WordDocument', 2, 2, free, 1, 4096)
    entry(2, '0Table', 2, free, free, 9, 4096)
    word = bytearray(4096)
    raw = text.encode('utf-16-le'); count = len(raw) // 2
    struct.pack_into('<HH', word, 0, 0xA5EC, 0xC1)
    struct.pack_into('<H', word, 32, 14)
    struct.pack_into('<H', word, 62, 22)
    struct.pack_into('<I', word, 76, count)
    struct.pack_into('<H', word, 152, 93)
    struct.pack_into('<II', word, 418, 0, 21)
    word[1024:1024 + len(raw)] = raw
    table = bytearray(4096)
    table[:21] = b'\x02' + struct.pack('<IIIHIH', 16, 0, count, 0, 1024, 0)
    fat = [free] * 128
    fat[0] = end
    for first in (1, 9):
        for i in range(first, first + 7): fat[i] = i + 1
        fat[first + 7] = end
    fat[17] = 0xFFFFFFFD
    return bytes(header + directory + word + table + struct.pack('<128I', *fat))
