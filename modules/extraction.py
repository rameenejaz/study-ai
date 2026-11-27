"""Text extraction from PDF and TXT lecture notes.

* PDFs are read with PyMuPDF, page by page, block by block, so paragraph
  boundaries survive. Every page is processed.
* Scanned (image-only) PDFs are detected and reported: OCR is not part of
  this project, so the user is told to use a text-based PDF instead.
* Encrypted, empty and corrupted files raise ``ExtractionError`` with a
  message that can be shown directly in the UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from modules.preprocessing import normalize_whitespace, word_count

try:  # PyMuPDF >= 1.24.3 exposes the "pymupdf" name; older versions only "fitz"
    import pymupdf  # type: ignore
except ImportError:  # pragma: no cover - depends on installed version
    import fitz as pymupdf  # type: ignore


MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB, generous for lecture notes
MIN_CHARS_PER_TEXT_PAGE = 25
_LINE_HYPHEN = "\x00"  # temporary marker for a word split by a hyphen at the end of a line
# Words that very often start a hyphenated compound ("low-priority", "self-attention").
_COMPOUND_LEFT = {"low", "high", "well", "self", "non", "multi", "real", "long", "short", "full", "half", "single",
                  "double", "open", "user", "kernel", "time", "first", "last", "top", "end", "cross", "two", "three",
                  "four", "one", "zero", "read", "write", "fixed", "built", "hard", "soft", "data", "page", "file",
                  "large", "small", "fine", "pre", "post", "co", "semi", "sub", "inter", "intra", "anti", "auto"}
_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
              "\ufb05": "st", "\ufb06": "st", "\u00ad": ""}


class ExtractionError(Exception):
    """Raised when no usable text can be extracted. The message is user-facing."""


@dataclass
class ExtractionResult:
    text: str
    source_type: str  # "pdf" or "txt"
    page_count: int = 0
    pages_with_text: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return word_count(self.text)

    @property
    def char_count(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def _page_text(page) -> str:
    """Join a page's text blocks into paragraphs.

    Lines inside a block are joined with spaces (they are visual line-wraps),
    and blocks are separated by blank lines (they are paragraphs/headings).
    Words hyphenated across a line break are re-joined.
    """
    paragraphs: List[str] = []
    flags = pymupdf.TEXTFLAGS_BLOCKS & ~pymupdf.TEXT_PRESERVE_LIGATURES  # "ﬁle" -> "file"
    for block in page.get_text("blocks", flags=flags, sort=True):
        # block = (x0, y0, x1, y1, text, block_no, block_type); type 0 = text
        if len(block) >= 7 and block[6] != 0:
            continue
        raw = block[4]
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not lines:
            continue
        joined = ""
        for line in lines:
            if joined.endswith("-") and re.match(r"[a-z]", line):
                joined = joined[:-1] + _LINE_HYPHEN + line  # resolved later, once the vocabulary is known
            elif re.match(r"^\s*(?:[-–•▪●◦*·]|\d{1,2}[.)])\s+", line) and joined:
                joined += "\n" + line  # keep bullet lists on separate lines
            else:
                joined = f"{joined} {line}" if joined else line
        paragraphs.append(joined)
    return "\n\n".join(paragraphs)


def _fix_ligatures(text: str) -> str:
    for ligature, letters in _LIGATURES.items():
        text = text.replace(ligature, letters)
    return text


def _resolve_line_hyphens(text: str) -> str:
    """Decide for each "word-<line break>word" whether the hyphen is real.

    "memory-" + "management" -> "memory-management" (both parts are words used elsewhere
    in the document), but "algo-" + "rithm" -> "algorithm" (a word broken by typesetting).
    """
    # Vocabulary of the document *without* the split fragments themselves.
    unsplit = re.sub(r"[A-Za-z]*" + _LINE_HYPHEN + r"[A-Za-z]*", " ", text)
    vocabulary = {w.lower() for w in re.findall(r"[A-Za-z]+", unsplit)}

    def decide(match: re.Match) -> str:
        left, right = match.group(1), match.group(2)
        right_is_word = right.lower() in vocabulary
        compound = right_is_word and (left.lower() in vocabulary or left.lower() in _COMPOUND_LEFT)
        return f"{left}-{right}" if compound and len(left) > 1 else f"{left}{right}"

    text = re.sub(r"([A-Za-z]+)" + _LINE_HYPHEN + r"([a-z]+)", decide, text)
    return text.replace(_LINE_HYPHEN, "-")


def extract_pdf(data: bytes) -> ExtractionResult:
    if not data:
        raise ExtractionError("The PDF file is empty (0 bytes).")
    if len(data) > MAX_FILE_BYTES:
        raise ExtractionError("The PDF is larger than 25 MB. Please split it into smaller files.")
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several different types
        raise ExtractionError(
            "This file could not be opened as a PDF. It may be corrupted or not a real PDF "
            f"(details: {exc.__class__.__name__})."
        ) from exc

    with document:
        if document.needs_pass:
            raise ExtractionError("This PDF is password-protected. Remove the password and upload it again.")
        page_count = document.page_count
        if page_count == 0:
            raise ExtractionError("The PDF has no pages.")

        page_texts: List[str] = []
        image_only_pages = 0
        pages_with_text = 0
        for page in document:
            try:
                text = _page_text(page)
            except Exception:  # a single damaged page should not lose the rest
                text = ""
            if len(text.strip()) >= MIN_CHARS_PER_TEXT_PAGE:
                pages_with_text += 1
                page_texts.append(text)
            elif page.get_images(full=False):
                image_only_pages += 1

    text = normalize_whitespace(_resolve_line_hyphens(_fix_ligatures("\n\n".join(page_texts))))
    warnings: List[str] = []
    if not text:
        if image_only_pages:
            raise ExtractionError(
                "This PDF appears to be scanned (its pages are images, not text). "
                "StudyMate AI does not include OCR, so it cannot read scanned pages. "
                "Export a text-based PDF, or copy the notes and paste them into the text box."
            )
        raise ExtractionError("No text was found in this PDF. It appears to be empty.")
    if image_only_pages:
        warnings.append(
            f"{image_only_pages} of {page_count} page(s) contain only images (probably scanned). "
            "Their content could not be read because OCR is not supported."
        )
    blank_pages = page_count - pages_with_text - image_only_pages
    if blank_pages:
        warnings.append(f"{blank_pages} page(s) had no text and were skipped.")
    return ExtractionResult(text, "pdf", page_count, pages_with_text, warnings)


# ---------------------------------------------------------------------------
# TXT
# ---------------------------------------------------------------------------


def _decode(data: bytes) -> tuple[str, str]:
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace"), "utf-8-sig"
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace"), "utf-16"
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace"), "cp1252"


def extract_txt(data: bytes) -> ExtractionResult:
    if not data or not data.strip():
        raise ExtractionError("The text file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise ExtractionError("The text file is larger than 25 MB.")
    if b"\x00" in data[:4096] and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise ExtractionError("This does not look like a plain-text file (it contains binary data).")
    text, encoding = _decode(data)
    text = normalize_whitespace(text)
    if not text:
        raise ExtractionError("The text file contains only whitespace.")
    warnings = []
    if encoding == "cp1252":
        warnings.append("The file was not UTF-8; it was read using Windows-1252 encoding.")
    if "�" in text:
        warnings.append("Some characters could not be decoded and were replaced with �.")
    return ExtractionResult(text, "txt", warnings=warnings)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def extract_text(filename: str, data: bytes) -> ExtractionResult:
    """Extract text from an uploaded file, choosing the reader by extension/content."""
    name = (filename or "").lower()
    if name.endswith(".pdf") or data[:5] == b"%PDF-":
        return extract_pdf(data)
    if name.endswith(".txt") or name.endswith(".md"):
        return extract_txt(data)
    raise ExtractionError("Unsupported file type. Please upload a PDF or a TXT file.")
