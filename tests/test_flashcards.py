import csv
import io

import pytest

from fakes import FakeQuestionModel
from modules.facts import analyse_notes, find_definition
from modules.flashcards import (
    Flashcard,
    FlashcardError,
    flashcards_to_csv,
    flashcards_to_txt,
    generate_flashcards,
    validate_flashcard,
)
from modules.preprocessing import normalise_for_match, text_similarity, word_count
from modules.question_writer import parse_model_question, validate_question


def _assert_valid_cards(result, notes):
    analysis = analyse_notes(notes)
    norm_notes = normalise_for_match(notes)
    for card in result.cards:
        assert card.question.strip() and card.answer.strip()
        assert validate_flashcard(card, analysis) is None
        assert normalise_for_match(card.answer.rstrip(".")) in norm_notes  # answer comes from the notes
        assert word_count(card.answer) <= 40  # never a paragraph
        assert card.source in notes or normalise_for_match(card.source) in norm_notes


@pytest.mark.parametrize("domain", ["os", "ml", "business", "football"])
@pytest.mark.parametrize("model_kind", ["none", "good"])
def test_acceptance_10_cards_are_valid(all_domain_notes, domain, model_kind):
    """Acceptance 3: 10 requested -> 10 valid Q/A pairs (or a clear explanation)."""
    notes = all_domain_notes[domain]
    model = FakeQuestionModel("good") if model_kind == "good" else None
    result = generate_flashcards(notes, 10, model=model)
    assert result.produced == 10, result.warnings
    _assert_valid_cards(result, notes)


@pytest.mark.parametrize("count", [5, 10, 15, 20])
def test_selectable_counts_on_long_notes(os_notes, count):
    result = generate_flashcards(os_notes, count, model=FakeQuestionModel("good"))
    assert result.produced == count
    assert len({normalise_for_match(c.question) for c in result.cards}) == count


def test_cards_come_from_the_whole_document(os_notes):
    result = generate_flashcards(os_notes, 20)
    sections = {c.section for c in result.cards}
    assert len(sections) >= result.sections - 1


def test_no_duplicate_questions_or_answers(ml_notes):
    result = generate_flashcards(ml_notes, 20, model=FakeQuestionModel("good"))
    questions = [c.question for c in result.cards]
    for i, a in enumerate(questions):
        for b in questions[i + 1:]:
            assert text_similarity(a, b) < 0.75
    answers = [normalise_for_match(c.answer) for c in result.cards]
    assert len(answers) == len(set(answers))


def test_short_notes_report_shortfall_honestly(short_notes):
    result = generate_flashcards(short_notes, 20)
    assert result.produced < 20
    assert not result.complete
    assert any("Only" in w and "not 20" in w for w in result.warnings)


@pytest.mark.parametrize("behaviour", ["garbage", "empty", "reveals_answer", "wrong_answers", "json"])
def test_invalid_model_output_is_never_shown(ml_notes, behaviour):
    model = FakeQuestionModel(behaviour)
    result = generate_flashcards(ml_notes, 10, model=model)
    _assert_valid_cards(result, ml_notes)
    for card in result.cards:
        assert ml_notes.count(card.question) == 0  # a copied paragraph is never a question
        if card.method in ("model", "model-retry"):
            assert behaviour == "json"  # only the recoverable JSON output may be accepted
    if behaviour != "json":
        assert any("failed validation" in w for w in result.warnings)


def test_retry_is_attempted_before_fallback(ml_notes):
    model = FakeQuestionModel("garbage")
    generate_flashcards(ml_notes, 5, model=model)
    prompts = model.all_prompts
    assert any(p.startswith("Write a question") for p in prompts)
    assert any("Ask one short question" in p for p in prompts)


def test_validate_flashcard_rules():
    assert validate_flashcard(Flashcard("", "x", "s", 1, "m")) == "missing question"
    assert validate_flashcard(Flashcard("What is x?", " ", "s", 1, "m")) == "missing answer"
    long_answer = " ".join(["word"] * 60)
    assert "passage" in validate_flashcard(Flashcard("What is x?", long_answer, "s", 1, "m"))
    assert validate_flashcard(Flashcard("Explain the topic", "An answer", "s", 1, "m")) is not None
    assert validate_flashcard(Flashcard("What is a process?", "A program in execution.", "s", 1, "m")) is None


def test_answer_must_be_supported_by_notes(ml_notes):
    analysis = analyse_notes(ml_notes)
    card = Flashcard("What is deep learning?", "A technique invented on Mars.", "s", 1, "model")
    assert validate_flashcard(card, analysis) == "answer is not supported by the notes"


def test_parse_model_question_recovers_formats():
    assert parse_model_question("Question: what is a thread") == "What is a thread?"
    assert parse_model_question('{"question": "Which algorithm avoids deadlock?"}') == \
        "Which algorithm avoids deadlock?"
    assert parse_model_question("{'question': 'Who wrote MINIX?',}") == "Who wrote MINIX?"
    assert parse_model_question("1. Q: When was Linux released? A: 1991") == "When was Linux released?"
    assert parse_model_question("") is None and parse_model_question(None) is None


def test_validate_question_rules():
    vocab = {"thread", "smallest", "unit", "execution", "cpu", "schedul", "manag"}
    sentence = "A thread is the smallest unit of execution that the CPU scheduler can manage."
    assert validate_question("What is the smallest unit of execution?", "thread", sentence, vocab) is None
    assert validate_question("What is a thread?", "thread", sentence, vocab) == "question reveals the answer"
    assert validate_question("Thread smallest unit execution", "x", sentence, vocab) == "not a question"
    assert validate_question("Is a thread small?", "x", sentence, vocab) == "not a wh-question"


def test_definition_detection():
    assert find_definition("A process is a program in execution.")["term"] == "A process"
    assert find_definition("The section of code is called the critical section.")["term"] == "the critical section"
    assert find_definition("It is a program in execution.") is None
    assert find_definition("Football is popular.") is None


def test_exports(ml_notes):
    result = generate_flashcards(ml_notes, 5)
    rows = list(csv.DictReader(io.StringIO(flashcards_to_csv(result.cards))))
    assert len(rows) == 5 and rows[0]["question"] and rows[0]["answer"]
    txt = flashcards_to_txt(result.cards)
    assert txt.count("Q: ") == 5 and txt.count("A: ") == 5


def test_too_short_notes_raise():
    with pytest.raises(FlashcardError):
        generate_flashcards("Too short.", 5)
