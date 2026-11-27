import pytest

from modules.extraction import ExtractionError, extract_pdf, extract_text, extract_txt, pymupdf
from modules.preprocessing import split_sentences


def make_pdf(pages):
    """Build a real PDF in memory; each item is the text of one page (None = blank page)."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def make_image_only_pdf():
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 60), False)
    pix.clear_with(200)
    page.insert_image(pymupdf.Rect(50, 50, 300, 300), pixmap=pix)
    data = doc.tobytes()
    doc.close()
    return data


# --- PDF ------------------------------------------------------------------------

def test_valid_pdf_extracts_every_page():
    pages = [
        "Page one. A process is a program in execution.",
        "Page two. A thread is the smallest unit of execution.",
        "Page three. Paging divides memory into fixed-size frames.",
    ]
    result = extract_pdf(make_pdf(pages))
    assert result.page_count == 3 and result.pages_with_text == 3
    for marker in ("process is a program", "smallest unit of execution", "fixed-size frames"):
        assert marker in result.text
    assert result.word_count > 20 and result.char_count == len(result.text)


def test_pdf_preserves_paragraph_boundaries():
    text = "First paragraph about kernels and system calls.\n\nSecond paragraph about scheduling algorithms."
    result = extract_pdf(make_pdf([text]))
    assert "\n\n" in result.text
    first, second = result.text.split("\n\n")[:2]
    assert "kernels" in first and "scheduling" in second


def test_pdf_with_some_blank_pages_warns():
    result = extract_pdf(make_pdf(["Real content about deadlocks and semaphores in operating systems.", None]))
    assert "deadlocks" in result.text
    assert any("no text" in w for w in result.warnings)


def test_empty_pdf_raises():
    with pytest.raises(ExtractionError, match="empty"):
        extract_pdf(make_pdf([None, None]))


def test_scanned_pdf_explains_no_ocr():
    with pytest.raises(ExtractionError, match="OCR"):
        extract_pdf(make_image_only_pdf())


def test_corrupted_pdf_raises_clear_error():
    with pytest.raises(ExtractionError, match="could not be opened"):
        extract_pdf(b"%PDF-1.7 this is not really a pdf \x00\x01\x02")


def test_zero_byte_pdf_raises():
    with pytest.raises(ExtractionError):
        extract_pdf(b"")


# --- TXT ------------------------------------------------------------------------

def test_valid_txt_utf8(ml_notes):
    result = extract_txt(ml_notes.encode("utf-8"))
    assert result.source_type == "txt"
    assert result.word_count == len(ml_notes.split()) or abs(result.word_count - len(ml_notes.split())) < 10
    assert "Supervised learning" in result.text


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "cp1252"])
def test_txt_other_encodings(encoding):
    text = "Café revenue rose. Profit is revenue minus expenses."
    result = extract_txt(text.encode(encoding))
    assert "Café" in result.text and "Profit" in result.text


@pytest.mark.parametrize("data", [b"", b"   \n\n  \t "])
def test_empty_txt_raises(data):
    with pytest.raises(ExtractionError, match="empty"):
        extract_txt(data)


def test_binary_file_with_txt_name_rejected():
    with pytest.raises(ExtractionError, match="binary"):
        extract_txt(b"\x00\x01\x02\x03binary\x00data")


def test_dispatcher_and_unsupported_type():
    assert extract_text("notes.txt", b"Hello world. This is a note.").source_type == "txt"
    assert extract_text("lecture.PDF", make_pdf(["Some lecture text for the dispatcher test."])).source_type == "pdf"
    with pytest.raises(ExtractionError, match="Unsupported"):
        extract_text("slides.pptx", b"PK\x03\x04")


def test_sentence_splitter_handles_abbreviations_and_bullets():
    text = ("Overview\n\nE.g. values such as 3.5 ms are common, i.e. fast. Dr. Smith agreed.\n"
            "- First bullet point here\n- Second bullet point here")
    sentences = split_sentences(text)
    assert "Overview" in sentences
    assert any(s.startswith("E.g. values such as 3.5 ms are common, i.e. fast.") for s in sentences)
    assert "Dr. Smith agreed." in sentences
    assert "First bullet point here" in sentences and "Second bullet point here" in sentences


def test_sample_pdf_round_trip_matches_txt(root):
    """The 4-page sample PDF (with typographic ligatures such as 'ﬁ') extracts word-for-word like the TXT."""
    pdf = extract_pdf((root / "sample_data/os_lecture_2000_words.pdf").read_bytes())
    txt = (root / "sample_data/os_lecture_2000_words.txt").read_text(encoding="utf-8")
    assert pdf.page_count == 4
    assert pdf.text.split() == txt.split()
    assert "ﬁ" not in pdf.text and "ﬀ" not in pdf.text


def test_line_end_hyphens_are_resolved():
    from modules.extraction import _LINE_HYPHEN, _resolve_line_hyphens

    text = (f"Memory management matters. The memory{_LINE_HYPHEN}management unit maps addresses. "
            f"An algo{_LINE_HYPHEN}rithm is a procedure.")
    result = _resolve_line_hyphens(text)
    assert "memory-management unit" in result  # real compound word keeps its hyphen
    assert "An algorithm is" in result          # typesetting break is joined
