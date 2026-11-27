"""Multiple-choice question generation with validated, grounded options.

Each MCQ is built from one fact in the notes:

* the correct option is a phrase/term that literally appears in that fact;
* the question comes from a template (definitions: "Which of the following
  is …?") or from FLAN-T5 via ``modules.question_writer`` (validated, verified,
  retried, with a fill-in-the-blank fallback);
* the three distractors are other concepts *from the same notes* and of the
  same kind (term ↔ term, name ↔ name, number ↔ number), chosen so they do not
  appear in the fact sentence and do not overlap with the correct answer –
  this keeps them plausible but unambiguous;
* options are shuffled with a fixed seed, labelled A–D, and validated
  (exactly four, all different, exactly one correct, explanation present).
"""

from __future__ import annotations

import csv
import io
import random
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from modules.facts import Candidate, NoteAnalysis, Phrase, analyse_notes, answer_in_notes, interleave_by_section
from modules.preprocessing import DuplicateFilter, content_words, normalise_for_match, word_count
from modules.question_writer import BLANK, contains_phrase, reverse_definition_question, write_questions

ALLOWED_COUNTS = (5, 10, 15, 20)
LETTERS = ("A", "B", "C", "D")
MIN_NOTE_WORDS = 40
MAX_OPTION_WORDS = 12


class MCQError(Exception):
    """User-facing MCQ generation failure."""


@dataclass
class MCQ:
    question: str
    options: List[str]
    correct_index: int
    explanation: str
    source: str
    section: int
    method: str

    @property
    def correct_letter(self) -> str:
        return LETTERS[self.correct_index]

    @property
    def correct_option(self) -> str:
        return self.options[self.correct_index]


@dataclass
class MCQResult:
    questions: List[MCQ]
    requested: int
    sections: int
    warnings: List[str] = field(default_factory=list)
    rejected_model_outputs: int = 0

    @property
    def produced(self) -> int:
        return len(self.questions)

    @property
    def complete(self) -> bool:
        return self.produced >= self.requested


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_mcq(mcq: MCQ, analysis: Optional[NoteAnalysis] = None) -> Optional[str]:
    """Return None if valid, otherwise the reason."""
    if not mcq.question or not mcq.question.strip():
        return "missing question"
    if not isinstance(mcq.options, list) or len(mcq.options) != 4:
        return "an MCQ must have exactly four options"
    if any(not isinstance(o, str) or not o.strip() for o in mcq.options):
        return "empty option"
    normalised = [normalise_for_match(o) for o in mcq.options]
    if len(set(normalised)) != 4:
        return "duplicate options"
    for i, a in enumerate(normalised):
        for j, b in enumerate(normalised):
            if i != j and f" {a} " in f" {b} ":
                return "one option is contained in another (ambiguous)"
    if not isinstance(mcq.correct_index, int) or not 0 <= mcq.correct_index < 4:
        return "correct answer is not one of the four options"
    if not mcq.explanation or not mcq.explanation.strip():
        return "missing explanation"
    if not (mcq.question.strip().endswith("?") or BLANK in mcq.question):
        return "question is not a question"
    correct = mcq.options[mcq.correct_index]
    if BLANK not in mcq.question and contains_phrase(mcq.question, correct):
        return "question reveals the answer"
    if analysis is not None and not answer_in_notes(correct, analysis):
        return "correct answer is not supported by the notes"
    for i, option in enumerate(mcq.options):
        if i != mcq.correct_index and contains_phrase(mcq.source, option):
            return "a distractor also appears in the source sentence (ambiguous)"
    return None


# ---------------------------------------------------------------------------
# Distractors
# ---------------------------------------------------------------------------

_COMPATIBLE = {
    "number": ("number",),
    "name": ("name", "acronym"),
    "acronym": ("acronym", "name"),
    "concept": ("concept", "term"),
    "term": ("term", "concept"),
    "definition": ("term", "concept"),
}


def _number_variants(answer: str, rng: random.Random) -> List[str]:
    match = re.search(r"\d+(?:[.,]\d+)?", answer)
    if not match:
        return []
    raw = match.group(0)
    value = float(raw.replace(",", ""))
    is_year = raw.isdigit() and 1000 <= value <= 2100 and len(raw) == 4
    if is_year:
        deltas = [d for d in (-10, -5, -3, -2, 2, 3, 5, 10)]
        values = [int(value + d) for d in deltas]
    elif value.is_integer():
        values = sorted({int(value * f) for f in (0.5, 2, 3, 1.5) if int(value * f) not in (0, int(value))} |
                        {int(value + d) for d in (-2, -1, 1, 2) if value + d > 0})
    else:
        values = [round(value * f, 2) for f in (0.5, 1.5, 2, 0.75)]
    rng.shuffle(values)
    return [answer[:match.start()] + str(v) + answer[match.end():] for v in values]


def choose_distractors(answer: str, answer_kind: str, sentence: str, analysis: NoteAnalysis,
                       rng: random.Random, extra_pool: Sequence[str] = ()) -> Optional[List[str]]:
    """Pick three wrong options from the notes, or None if that is impossible."""
    answer_norm = normalise_for_match(answer)
    answer_words = set(content_words(answer))
    kinds = _COMPATIBLE.get(answer_kind, ("concept", "term"))
    pool: List[Phrase] = analysis.phrases_of_kind(kinds)
    options = [(p.text, p.score) for p in pool] + [(t, 3.0) for t in extra_pool]

    def acceptable(text: str, chosen: List[str]) -> bool:
        norm = normalise_for_match(text)
        if not norm or norm == answer_norm or word_count(text) > MAX_OPTION_WORDS:
            return False
        if f" {norm} " in f" {answer_norm} " or f" {answer_norm} " in f" {norm} ":
            return False
        if answer_words & set(content_words(text)):
            return False
        if contains_phrase(sentence, text):
            return False
        for other in chosen:
            other_norm = normalise_for_match(other)
            if f" {norm} " in f" {other_norm} " or f" {other_norm} " in f" {norm} " or \
                    set(content_words(other)) & set(content_words(text)):
                return False
        return True

    target_len = word_count(answer)
    preferred = {normalise_for_match(t) for t in extra_pool}
    # Same-kind concepts first (e.g. other defined terms), then similar length, then importance.
    ranked = sorted(options, key=lambda item: (normalise_for_match(item[0]) not in preferred,
                                               abs(word_count(item[0]) - target_len), -item[1]))
    shortlist = ranked[:10]
    rng.shuffle(shortlist)
    shortlist.sort(key=lambda item: (normalise_for_match(item[0]) not in preferred,
                                     abs(word_count(item[0]) - target_len)))
    chosen: List[str] = []
    for text, _ in shortlist + ranked[10:]:
        if acceptable(text, chosen):
            chosen.append(_match_case(text, answer))
        if len(chosen) == 3:
            return chosen
    if answer_kind == "number":
        for variant in _number_variants(answer, rng):
            if acceptable(variant, chosen):
                chosen.append(variant)
            if len(chosen) == 3:
                return chosen
    return None


def _match_case(option: str, answer: str) -> str:
    """Use the answer's capitalisation style so the correct option doesn't stand out."""
    if answer[:1].isupper() and option[:1].islower():
        return option[0].upper() + option[1:]
    if answer[:1].islower() and option[:1].isupper() and not option.split()[0].isupper() \
            and not re.match(r"[A-Z][a-z]+\s+[A-Z]", option):
        return option[0].lower() + option[1:]
    return option


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _strip_article(term: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", term.strip(), flags=re.IGNORECASE)


def _assemble(question: str, answer: str, distractors: List[str], cand: Candidate, method: str,
              position: int, rng: random.Random) -> MCQ:
    others = list(distractors)
    rng.shuffle(others)
    options = others[:position] + [answer] + others[position:]
    explanation = f"From your notes (section {cand.section + 1}): “{cand.sentence}”"
    return MCQ(question, options, position, explanation, cand.sentence, cand.section + 1, method)


def generate_mcqs(text: str, count: int = 10, model=None, verify: bool = True, seed: int = 7) -> MCQResult:
    """Generate up to ``count`` validated MCQs covering the whole notes."""
    if count <= 0:
        raise ValueError("count must be positive")
    if word_count(text) < MIN_NOTE_WORDS:
        raise MCQError(f"The notes are too short for a quiz (need at least {MIN_NOTE_WORDS} words).")

    analysis = analyse_notes(text)
    ordered = interleave_by_section(analysis.candidates)
    if not ordered:
        raise MCQError("No definitions or key facts were found in the notes to build questions from.")

    defined_terms = [_strip_article(c.term) for c in analysis.candidates if c.style == "definition"]
    rng = random.Random(seed)
    positions = [0, 1, 2, 3] * ((count // 4) + 2)
    rng.shuffle(positions)

    questions: List[MCQ] = []
    seen = DuplicateFilter(threshold=0.7)
    used_answers: set = set()
    used_positions: dict = {}
    rejected_outputs = 0
    index = 0
    while len(questions) < count and index < len(ordered):
        needed = count - len(questions)
        batch = ordered[index:index + max(needed * 2, 4)]
        index += len(batch)
        statements = [c for c in batch if c.style == "statement"]
        written = write_questions(statements, analysis.vocabulary, model=model, verify=verify) if statements else {}
        statement_index = {id(c): i for i, c in enumerate(statements)}

        for cand in batch:
            if len(questions) >= count:
                break
            if used_positions.get(cand.position, 0) >= 1:
                continue  # one question per sentence keeps the quiz varied
            item_rng = random.Random(seed * 1000 + cand.position * 10 + len(questions))
            if cand.style == "definition":
                answer = _strip_article(cand.term)
                answer = answer[0].upper() + answer[1:]
                question = reverse_definition_question(cand.definition, cand.verb)
                method = "definition"
                distractors = choose_distractors(answer, "term", cand.sentence, analysis, item_rng,
                                                 extra_pool=[t for t in defined_terms
                                                             if normalise_for_match(t) != normalise_for_match(answer)])
            else:
                result = written.get(statement_index[id(cand)])
                if result is None:
                    continue
                rejected_outputs += result.rejected_outputs
                answer, question, method = cand.answer, result.question, result.method
                distractors = choose_distractors(answer, cand.answer_kind, cand.sentence, analysis, item_rng)
            if distractors is None:
                continue
            key = normalise_for_match(answer)
            if key in used_answers:
                continue
            mcq = _assemble(question, answer, distractors, cand, method, positions[len(questions)], item_rng)
            if validate_mcq(mcq, analysis) is not None:
                continue
            if not seen.add(mcq.question):
                continue
            used_answers.add(key)
            used_positions[cand.position] = 1
            questions.append(mcq)

    questions.sort(key=lambda q: (q.section, analysis.sentences.index(q.source) if q.source in analysis.sentences else 0))
    warnings: List[str] = []
    if len(questions) < count:
        warnings.append(
            f"Only {len(questions)} valid question(s) could be made, not {count}. Each question needs a distinct "
            "fact plus three different concepts from the notes as wrong options; longer notes give more questions."
        )
    if model is None:
        warnings.append("The AI question writer was not available, so fact questions use fill-in-the-blank form.")
    if rejected_outputs:
        warnings.append(f"{rejected_outputs} model output(s) failed validation and were retried or replaced.")
    return MCQResult(questions, count, analysis.sections, warnings, rejected_outputs)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def mcqs_to_txt(questions: List[MCQ], include_answers: bool = False) -> str:
    title = "StudyMate AI — Practice MCQs" + (" (with answers)" if include_answers else "")
    lines = [title, "=" * len(title), ""]
    for number, q in enumerate(questions, 1):
        lines.append(f"{number}. {q.question}")
        lines += [f"   {letter}) {option}" for letter, option in zip(LETTERS, q.options)]
        if include_answers:
            lines.append(f"   Answer: {q.correct_letter}) {q.correct_option}")
            lines.append(f"   Explanation: {q.explanation}")
        lines.append("")
    return "\n".join(lines)


def answer_key_txt(questions: List[MCQ]) -> str:
    lines = ["StudyMate AI — Answer key", "=" * 25, ""]
    for number, q in enumerate(questions, 1):
        lines.append(f"{number}. {q.correct_letter}) {q.correct_option}")
        lines.append(f"   {q.explanation}")
    return "\n".join(lines)


def mcqs_to_csv(questions: List[MCQ]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["number", "question", "A", "B", "C", "D", "correct", "explanation"])
    for number, q in enumerate(questions, 1):
        writer.writerow([number, q.question, *q.options, q.correct_letter, q.explanation])
    return buffer.getvalue()
