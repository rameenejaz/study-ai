import csv
import io
import random
from collections import Counter

import pytest

from fakes import FakeQuestionModel
from modules.facts import analyse_notes
from modules.mcqs import (
    MCQ,
    MCQError,
    answer_key_txt,
    choose_distractors,
    generate_mcqs,
    mcqs_to_csv,
    mcqs_to_txt,
    validate_mcq,
)
from modules.preprocessing import normalise_for_match, text_similarity
from modules.question_writer import contains_phrase


def _assert_valid(result, notes):
    analysis = analyse_notes(notes)
    for q in result.questions:
        assert len(q.options) == 4
        assert len({normalise_for_match(o) for o in q.options}) == 4
        assert 0 <= q.correct_index < 4 and q.correct_letter in "ABCD"
        assert q.options[q.correct_index] == q.correct_option
        assert q.explanation.strip()
        assert validate_mcq(q, analysis) is None
        assert normalise_for_match(q.correct_option) in normalise_for_match(notes)  # grounded answer
        for i, option in enumerate(q.options):
            if i != q.correct_index:
                assert not contains_phrase(q.source, option)  # distractor cannot also be "true" in the fact


@pytest.mark.parametrize("domain", ["os", "ml", "business", "football"])
@pytest.mark.parametrize("model_kind", ["none", "good"])
def test_acceptance_10_mcqs(all_domain_notes, domain, model_kind):
    """Acceptance 4: 10 requested -> 10 valid questions, 4 options, one correct answer."""
    notes = all_domain_notes[domain]
    result = generate_mcqs(notes, 10, model=FakeQuestionModel("good") if model_kind == "good" else None)
    assert result.produced == 10, result.warnings
    _assert_valid(result, notes)


@pytest.mark.parametrize("count", [5, 10, 15, 20])
def test_selectable_counts(os_notes, count):
    result = generate_mcqs(os_notes, count, model=FakeQuestionModel("good"))
    assert result.produced == count
    _assert_valid(result, os_notes)


def test_no_duplicate_questions(os_notes):
    result = generate_mcqs(os_notes, 20)
    qs = [q.question for q in result.questions]
    for i, a in enumerate(qs):
        for b in qs[i + 1:]:
            assert text_similarity(a, b) < 0.7
    assert len({normalise_for_match(q.correct_option) for q in result.questions}) == len(qs)


def test_correct_answers_are_spread_over_letters(os_notes):
    result = generate_mcqs(os_notes, 20)
    counts = Counter(q.correct_letter for q in result.questions)
    assert len(counts) == 4 and max(counts.values()) <= 8


def test_questions_cover_whole_document(os_notes):
    result = generate_mcqs(os_notes, 20)
    assert len({q.section for q in result.questions}) >= result.sections - 2


@pytest.mark.parametrize("behaviour", ["garbage", "empty", "reveals_answer", "wrong_answers"])
def test_invalid_model_outputs_are_replaced(ml_notes, behaviour):
    result = generate_mcqs(ml_notes, 10, model=FakeQuestionModel(behaviour))
    _assert_valid(result, ml_notes)
    assert all(q.method in ("definition", "cloze") for q in result.questions)


def test_shortfall_is_reported(short_notes):
    result = generate_mcqs(short_notes, 20)
    assert result.produced < 20
    assert any("Only" in w and "not 20" in w for w in result.warnings)


def _mcq(**overrides):
    base = dict(question="Which of the following is a program in execution?",
                options=["Process", "Thread", "Kernel", "Semaphore"], correct_index=0,
                explanation="From your notes: A process is a program in execution.",
                source="A process is a program in execution.", section=1, method="definition")
    base.update(overrides)
    return MCQ(**base)


def test_validate_mcq_rules():
    assert validate_mcq(_mcq()) is None
    assert "exactly four" in validate_mcq(_mcq(options=["Process", "Thread", "Kernel"]))
    assert "exactly four" in validate_mcq(_mcq(options=["Process", "Thread", "Kernel", "Pipe", "Socket"]))
    assert validate_mcq(_mcq(options=["Process", "process", "Kernel", "Semaphore"])) == "duplicate options"
    assert "not one of the four" in validate_mcq(_mcq(correct_index=4))
    assert "not one of the four" in validate_mcq(_mcq(correct_index=-1))
    assert validate_mcq(_mcq(explanation="")) == "missing explanation"
    assert "contained in another" in validate_mcq(_mcq(options=["Process", "Process control block", "Kernel", "Pipe"]))
    assert validate_mcq(_mcq(question="Which process is a program in execution?")) == "question reveals the answer"
    assert validate_mcq(_mcq(options=["Process", "Thread", "Program", "Pipe"])) == \
        "a distractor also appears in the source sentence (ambiguous)"


def test_distractors_are_from_notes_and_same_kind(os_notes):
    analysis = analyse_notes(os_notes)
    distractors = choose_distractors("1991", "number", "Linux was first released in 1991.", analysis,
                                     random.Random(1))
    assert distractors and len(distractors) == 3 and "1991" not in distractors
    terms = choose_distractors("Deadlock", "term", "A deadlock is a situation in which processes wait.",
                               analysis, random.Random(1))
    assert terms and all(normalise_for_match(t) in normalise_for_match(os_notes) for t in terms)


def test_exports(ml_notes):
    result = generate_mcqs(ml_notes, 5)
    txt = mcqs_to_txt(result.questions)
    assert txt.count(" A) ") == 5 and txt.count(" D) ") == 5 and "Answer:" not in txt
    assert "Answer:" in mcqs_to_txt(result.questions, include_answers=True)
    key = answer_key_txt(result.questions)
    assert all(f"{i}. {q.correct_letter})" in key for i, q in enumerate(result.questions, 1))
    rows = list(csv.reader(io.StringIO(mcqs_to_csv(result.questions))))
    assert rows[0][:7] == ["number", "question", "A", "B", "C", "D", "correct"] and len(rows) == 6


def test_too_short_notes_raise():
    with pytest.raises(MCQError):
        generate_mcqs("A very short note.", 5)
