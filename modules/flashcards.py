"""Flashcard generation: grounded question–answer pairs from the whole notes.

Why the old version failed: it sent "Create 6 concise study flashcards …" plus
the first 2,500 characters to FLAN-T5-small and displayed whatever came back.
A 77M-parameter model cannot reliably produce a numbered list of Q/A pairs,
the input was cut to the opening of the notes, and nothing checked the output,
so students saw a raw paragraph instead of cards.

The new pipeline (see ``modules.facts`` and ``modules.question_writer``):
definitions and key facts are found in every section of the notes; each card's
answer is text that appears in the notes; questions come from templates
(definitions) or from FLAN-T5, one at a time, parsed, validated, verified and
retried, with a fill-in-the-blank fallback. Cards are de-duplicated and the
result says honestly how many valid cards were produced.
"""

from __future__ import annotations

import csv
import io
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from modules.facts import Candidate, NoteAnalysis, analyse_notes, answer_in_notes, interleave_by_section
from modules.preprocessing import DuplicateFilter, normalise_for_match, split_sentences, word_count
from modules.question_writer import BLANK, definition_question, write_questions

ALLOWED_COUNTS = (5, 10, 15, 20)
MAX_ANSWER_WORDS = 40
MIN_NOTE_WORDS = 30


class FlashcardError(Exception):
    """User-facing flashcard generation failure."""


@dataclass
class Flashcard:
    question: str
    answer: str
    source: str     # the sentence in the notes that supports the answer
    section: int    # 1-based section of the notes
    method: str     # "definition", "model", "model-retry" or "cloze"


@dataclass
class FlashcardResult:
    cards: List[Flashcard]
    requested: int
    sections: int
    warnings: List[str] = field(default_factory=list)
    rejected_model_outputs: int = 0

    @property
    def produced(self) -> int:
        return len(self.cards)

    @property
    def complete(self) -> bool:
        return self.produced >= self.requested


def _sentence_case(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


def validate_flashcard(card: Flashcard, analysis: Optional[NoteAnalysis] = None) -> Optional[str]:
    """Return None if the card is valid, otherwise a reason."""
    if not card.question or not card.question.strip():
        return "missing question"
    if not card.answer or not card.answer.strip():
        return "missing answer"
    q, a = card.question.strip(), card.answer.strip()
    if normalise_for_match(q) == normalise_for_match(a):
        return "question and answer are identical"
    if word_count(a) > MAX_ANSWER_WORDS or len(split_sentences(a)) > 1:
        return "answer is a passage, not a flashcard answer"
    if word_count(q) > 60:
        return "question is a passage"
    if not (q.endswith("?") or BLANK in q):
        return "question is not a question or a fill-in-the-blank"
    if analysis is not None and not answer_in_notes(a.rstrip("."), analysis):
        return "answer is not supported by the notes"
    return None


def _card_from_definition(cand: Candidate, notes: str) -> Flashcard:
    return Flashcard(definition_question(cand.term, cand.verb, notes), _sentence_case(cand.definition),
                     cand.sentence, cand.section + 1, "definition")


def generate_flashcards(text: str, count: int = 10, model=None, verify: bool = True) -> FlashcardResult:
    """Generate up to ``count`` validated flashcards from the full notes.

    ``model`` is a ``modules.models.Seq2SeqModel`` (FLAN-T5) or None. With
    None, definition cards and fill-in-the-blank cards are still produced.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if word_count(text) < MIN_NOTE_WORDS:
        raise FlashcardError(f"The notes are too short to make flashcards (need at least {MIN_NOTE_WORDS} words).")

    analysis = analyse_notes(text)
    ordered = interleave_by_section(analysis.candidates)
    if not ordered:
        raise FlashcardError(
            "No definitions or key facts were found. Flashcards need complete sentences, for example "
            "'A process is a program in execution.'"
        )

    cards: List[Flashcard] = []
    seen_questions = DuplicateFilter(threshold=0.75)
    used_answers: set = set()
    used_positions: dict = {}
    rejected_outputs = 0
    warnings: List[str] = []

    # Work through candidates in batches so the model is only called for
    # roughly as many questions as are still needed.
    index = 0
    while len(cards) < count and index < len(ordered):
        needed = count - len(cards)
        batch = ordered[index:index + max(needed * 2, 4)]
        index += len(batch)
        statements = [c for c in batch if c.style == "statement"]
        written = write_questions(statements, analysis.vocabulary, model=model, verify=verify) if statements else {}
        statement_index = {id(c): i for i, c in enumerate(statements)}

        for cand in batch:
            if len(cards) >= count:
                break
            if used_positions.get(cand.position, 0) >= 2:
                continue  # at most two cards from one sentence
            if cand.style == "definition":
                card = _card_from_definition(cand, text)
            else:
                result = written.get(statement_index[id(cand)])
                if result is None:
                    continue
                rejected_outputs += result.rejected_outputs
                card = Flashcard(result.question, cand.answer, cand.sentence, cand.section + 1, result.method)
            answer_key = normalise_for_match(card.answer)
            if answer_key in used_answers:
                continue
            if validate_flashcard(card, analysis) is not None:
                continue
            if not seen_questions.add(card.question):
                continue
            used_answers.add(answer_key)
            used_positions[cand.position] = used_positions.get(cand.position, 0) + 1
            cards.append(card)

    # Present cards in the order of the notes.
    cards.sort(key=lambda c: (c.section, analysis.sentences.index(c.source) if c.source in analysis.sentences else 0))
    if len(cards) < count:
        warnings.append(
            f"Only {len(cards)} valid flashcard(s) could be made from these notes, not {count}. The notes contain "
            "too few distinct definitions and facts; add more material or choose a smaller number."
        )
    if model is None:
        warnings.append("The AI question writer was not available, so fact cards use fill-in-the-blank questions.")
    if rejected_outputs:
        warnings.append(f"{rejected_outputs} model output(s) failed validation and were retried or replaced.")
    return FlashcardResult(cards, count, analysis.sections, warnings, rejected_outputs)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def flashcards_to_csv(cards: List[Flashcard]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["number", "question", "answer", "source", "section", "method"])
    writer.writeheader()
    for number, card in enumerate(cards, 1):
        writer.writerow({"number": number, **asdict(card)})
    return buffer.getvalue()


def flashcards_to_txt(cards: List[Flashcard]) -> str:
    lines = ["StudyMate AI — Flashcards", "=" * 26, ""]
    for number, card in enumerate(cards, 1):
        lines += [f"Card {number}", f"Q: {card.question}", f"A: {card.answer}",
                  f"Source (section {card.section}): {card.source}", ""]
    return "\n".join(lines)
