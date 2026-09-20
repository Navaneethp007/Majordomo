"""Tests for reading documents that are not text files.

The floor matters more than the feature. `read_text(errors="replace")` does not
fail on a PDF — it returns the bytes as replacement characters, the model
summarises the noise, and you get a confident answer about a document nobody
read. Most of what is here is about that never happening again.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from unittest import mock

import pytest

from majordomo import documents, tools


def minimal_pdf(text: str, pages: int = 1) -> bytes:
    """A hand-built PDF with a real text layer. No dependency to write one."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF"
    ).encode()
    return bytes(out)


def write_docx(path: Path, paragraphs, table=None) -> None:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)
    if table:
        added = document.add_table(rows=len(table), cols=len(table[0]))
        for row, values in zip(added.rows, table):
            for cell, value in zip(row.cells, values):
                cell.text = value
    document.save(str(path))


# ---------------------------------------------------------------------------
# The floor: nothing returns garbage as though it were a document
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,content,expected",
    [
        ("img.png", b"\x89PNG\r\n\x1a\n rest", "a PNG image"),
        ("photo.jpg", b"\xff\xd8\xff\xe0 rest", "a JPEG image"),
        ("app.exe", b"MZ\x90\x00 rest", "a Windows executable"),
        ("data.db", b"SQLite format 3\x00 rest", "a SQLite database"),
        ("blob.bin", b"nothing familiar \x00\x01\x02", "a binary file"),
    ],
)
def test_binary_files_are_named_not_decoded(tmp_path, name, content, expected):
    """Naming the format leads somewhere; "not text" invites a second attempt
    at the same file."""
    (tmp_path / name).write_bytes(content)

    result = tools.read_file(tmp_path, path=name)

    assert "ERROR" in result
    assert expected in result


@pytest.mark.parametrize(
    "name,text,encoding",
    [
        ("plain.txt", "ordinary text", "utf-8"),
        ("bom.txt", "text with a byte-order mark", "utf-8-sig"),
        ("wide.txt", "text in sixteen bits", "utf-16"),
        ("accents.py", "# café, naïve, 日本語", "utf-8"),
    ],
)
def test_text_is_still_text(tmp_path, name, text, encoding):
    """UTF-16 trips the NUL-byte test, so the encoding check has to come first
    — and then it has to be *decoded* as UTF-16, or it reads as letters with
    spaces between them."""
    (tmp_path / name).write_text(text, encoding=encoding)

    assert text in tools.read_file(tmp_path, path=name)


def test_grep_never_reports_a_match_inside_a_binary(tmp_path):
    """Worse than read_file: grep prints matching *lines*, so a binary file
    contributes mangled bytes that look like findings."""
    (tmp_path / "notes.txt").write_text("SECRET in text\n", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"SECRET\x00\x01\x02 in binary")

    out = tools.grep(tmp_path, pattern="SECRET")

    assert "notes.txt" in out
    assert "blob.bin" not in out


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_a_pdf_with_a_text_layer_reads_as_text(tmp_path):
    pytest.importorskip("pypdf")
    (tmp_path / "report.pdf").write_bytes(minimal_pdf("Quarterly Report: revenue up"))

    assert "Quarterly Report" in tools.read_file(tmp_path, path="report.pdf")


def test_a_docx_reads_as_text_including_its_tables(tmp_path):
    """Tables hold much of what people actually want out of a .docx, and
    `paragraphs` does not include them."""
    write_docx(
        tmp_path / "notes.docx",
        ["Design notes", "The agent is a loop."],
        table=[["Role", "Model"], ["fuser", "nemotron"]],
    )

    out = tools.read_file(tmp_path, path="notes.docx")

    assert "The agent is a loop." in out
    assert "fuser | nemotron" in out


def test_a_scan_is_not_reported_as_an_empty_document(tmp_path):
    """The failure that must never look like success: a scanned page yields
    zero characters, and an empty string returned as "the document" reads
    exactly like an empty document."""
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(tmp_path / "scan.pdf", "wb") as handle:
        writer.write(handle)

    result = tools.read_file(tmp_path, path="scan.pdf")

    assert "ERROR" in result
    assert "scan" in result
    assert "OCR" in result


def test_a_corrupt_pdf_says_so(tmp_path):
    pytest.importorskip("pypdf")
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4\nthis is not a pdf at all")

    result = tools.read_file(tmp_path, path="broken.pdf")

    assert "ERROR" in result
    assert "broken.pdf" in result


# ---------------------------------------------------------------------------
# Without the extra
# ---------------------------------------------------------------------------


def test_a_missing_extra_names_the_extra(tmp_path):
    """One pip command away, so the error should say which one."""
    (tmp_path / "report.pdf").write_bytes(minimal_pdf("anything"))

    with mock.patch.dict("sys.modules", {"pypdf": None}):
        result = tools.read_file(tmp_path, path="report.pdf")

    assert "majordomo[documents]" in result


def test_a_missing_docx_extra_names_it_too(tmp_path):
    with zipfile.ZipFile(tmp_path / "notes.docx", "w") as archive:
        archive.writestr("word/document.xml", "<w:document/>")

    with mock.patch.dict("sys.modules", {"docx": None}):
        result = tools.read_file(tmp_path, path="notes.docx")

    assert "majordomo[documents]" in result


def test_the_three_failures_are_told_apart():
    """They need different words because the remedies differ: install a
    package, find the password, or run OCR elsewhere."""
    assert issubclass(documents.ExtractorMissing, documents.ExtractionError)
    assert documents.can_extract(Path("a.pdf"))
    assert documents.can_extract(Path("A.DOCX"))
    assert not documents.can_extract(Path("a.txt"))
    assert not documents.can_extract(Path("a.epub"))


def test_an_unknown_format_is_refused_by_name():
    with pytest.raises(documents.ExtractionError, match=".epub"):
        documents.extract(Path("a.epub"))


# ---------------------------------------------------------------------------
# Signatures that are also ordinary text
# ---------------------------------------------------------------------------


def test_a_note_about_executables_is_not_refused_as_one(tmp_path):
    """`MZ` is a DOS header *and* two printable letters. The refusal tells the
    model not to retry, so it could not recover from the mistake."""
    (tmp_path / "note.txt").write_text(
        "MZ is the prefix used by PE binaries on Windows.\n", encoding="utf-8"
    )

    assert "MZ is the prefix" in tools.read_file(tmp_path, path="note.txt")


def test_a_real_executable_is_still_refused(tmp_path):
    (tmp_path / "app.exe").write_bytes(b"MZ\x90\x00\x03\x00\x00\x00PE\x00\x00")

    assert "a Windows executable" in tools.read_file(tmp_path, path="app.exe")


def test_the_unambiguous_signatures_need_no_corroboration(tmp_path):
    """Long printable magic is distinctive enough on its own — a text file
    opening "SQLite format 3" is not a thing that happens."""
    (tmp_path / "data.db").write_text("SQLite format 3 and then some text", encoding="utf-8")

    assert "a SQLite database" in tools.read_file(tmp_path, path="data.db")


# ---------------------------------------------------------------------------
# Encryption, reachably reported
# ---------------------------------------------------------------------------


def test_a_password_protected_pdf_says_so(tmp_path):
    """`decrypt` returns a PasswordType and does not raise on a wrong password,
    so the `except Exception` around it caught nothing: execution fell through
    to `reader.pages` and an encrypted PDF was reported as "could not be read
    as a PDF"."""
    pypdf = pytest.importorskip("pypdf")
    (tmp_path / "plain.pdf").write_bytes(minimal_pdf("Quarterly Report: revenue up"))

    writer = pypdf.PdfWriter(clone_from=str(tmp_path / "plain.pdf"))
    writer.encrypt("hunter2")
    with open(tmp_path / "locked.pdf", "wb") as handle:
        writer.write(handle)

    result = tools.read_file(tmp_path, path="locked.pdf")

    assert "password-protected" in result
    assert "could not be read as a PDF" not in result


def test_a_missing_crypto_package_is_named(tmp_path):
    """DependencyError is not a PdfReadError, so it escaped this module
    entirely and surfaced as a generic read failure."""
    pypdf = pytest.importorskip("pypdf")
    from pypdf.errors import DependencyError

    (tmp_path / "aes.pdf").write_bytes(minimal_pdf("anything"))

    with mock.patch("pypdf.PdfReader") as reader:
        reader.return_value.is_encrypted = True
        reader.return_value.decrypt.side_effect = DependencyError("needs cryptography")
        result = tools.read_file(tmp_path, path="aes.pdf")

    assert "cryptography" in result


# ---------------------------------------------------------------------------
# grep reads each file once, and decodes it properly
# ---------------------------------------------------------------------------


def test_grep_finds_matches_in_utf16_text(tmp_path):
    """The sniff deliberately lets BOM-marked UTF-16 through as text; grep then
    decoded it as UTF-8 and produced garbage."""
    (tmp_path / "wide.txt").write_text("SECRET in sixteen bits\n", encoding="utf-16")

    out = tools.grep(tmp_path, pattern="SECRET")

    assert "SECRET in sixteen bits" in out


def test_grep_opens_each_candidate_once(tmp_path):
    """It sniffed every file and then opened it again to search."""
    for n in range(3):
        (tmp_path / f"f{n}.txt").write_text("needle\n", encoding="utf-8")

    opened = []
    real = Path.read_bytes

    def counting(self, *a, **k):
        opened.append(self.name)
        return real(self, *a, **k)

    with mock.patch.object(Path, "read_bytes", counting):
        tools.grep(tmp_path, pattern="needle")

    assert sorted(opened) == ["f0.txt", "f1.txt", "f2.txt"]
