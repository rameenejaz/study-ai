"""Turning a (sentence, answer) pair into a well-formed question.

FLAN-T5 is good at one narrow task – "write a question about this sentence
whose answer is X" – and unreliable at producing long structured lists. So
the model is only ever asked for *one* question at a time, and its output is:

1. parsed      – prefixes such as "Question:", quotes, JSON wrappers, extra
                 lines and missing question marks are repaired;
2. validated   – must be a real wh-question, must not reveal the answer, must
                 not be a copy of the sentence, must use the notes' vocabulary;
3. verified    – (optional) the model answers its own question from the
                 sentence; the answer must match the expected answer;
4. retried     – with a corrective prompt if any check fails;
5. replaced    – by a deterministic fill-in-the-blank question if the model
                 still fails, so malformed output is never shown as a question.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from modules.facts import Candidate
from modules.preprocessing import content_words, normalise_for_match, support_ratio, text_similarity, word_count

BLANK = "_____"
_WH_START = re.compile(
    r"^(what|which|who|whom|whose|when|where|why|how|in what|in which|on what|on which|at what|for what|"
    r"by what|from what|to what|under what|during what|according to)\b",
    re.IGNORECASE,
)
_PREFIX_RE = re.compile(r"^\s*(?:question|q|generated question|output|answer)\s*[:.)-]\s*", re.IGNORECASE)
_LIST_PREFIX_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")


@dataclass
class WrittenQuestion:
    question: str
    method: str          # "model", "model-retry", "template" or "cloze"
    rejected_outputs: int = 0


# ---------------------------------------------------------------------------
# Parsing model output
# ---------------------------------------------------------------------------


def parse_model_question(raw: Optional[str]) -> Optional[str]:
    """Recover a single question from raw model text, or return None."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    # JSON-ish output: {"question": "..."} (possibly with a trailing comma or single quotes)
    if text.startswith("{") or '"question"' in text.lower():
        try:
            data = json.loads(text)
            if isinstance(data, dict) and isinstance(data.get("question"), str):
                text = data["question"]
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"[\"']question[\"']\s*:\s*[\"']([^\"']+)[\"']", text, re.IGNORECASE)
            if match:
                text = match.group(1)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    line = lines[0]
    for _ in range(2):
        line = _PREFIX_RE.sub("", _LIST_PREFIX_RE.sub("", line)).strip()
    line = line.strip("\"'“”‘’ ")
    if "?" in line:
        line = line[: line.index("?") + 1]
    elif _WH_START.match(line):
        line = line.rstrip(" .!;:") + "?"
    line = re.sub(r"\s+", " ", line)
    line = re.sub(r"\s+\?", "?", line)
    if not line or line == "?":
        return None
    return line[0].upper() + line[1:]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def contains_phrase(text: str, phrase: str) -> bool:
    norm_phrase = normalise_for_match(phrase)
    return bool(norm_phrase) and f" {norm_phrase} " in f" {normalise_for_match(text)} "


def validate_question(question: Optional[str], answer: str, sentence: str, vocabulary: set) -> Optional[str]:
    """Return None if valid, otherwise the reason for rejection."""
    if not question:
        return "empty output"
    n_words = word_count(question)
    if n_words < 4:
        return "too short"
    if n_words > 35:
        return "too long"
    if not question.endswith("?"):
        return "not a question"
    if not _WH_START.match(question):
        return "not a wh-question"
    if contains_phrase(question, answer):
        return "question reveals the answer"
    if text_similarity(question, sentence) >= 0.9:
        return "copy of the source sentence"
    tokens = content_words(question)
    if tokens and max(tokens.count(t) for t in set(tokens)) > 2:
        return "repetitive output"
    if support_ratio(question, vocabulary) < 0.6:
        return "uses words that are not in the notes"
    return None


def answers_match(predicted: str, expected: str) -> bool:
    """Token-overlap check used to verify that a question leads to the expected answer."""
    pred, exp = normalise_for_match(predicted), normalise_for_match(expected)
    if not pred or not exp:
        return False
    if exp in pred or pred in exp:
        return True
    p_tokens, e_tokens = pred.split(), exp.split()
    common = sum(min(p_tokens.count(t), e_tokens.count(t)) for t in set(e_tokens))
    if common == 0:
        return False
    precision, recall = common / len(p_tokens), common / len(e_tokens)
    return 2 * precision * recall / (precision + recall) >= 0.5


# ---------------------------------------------------------------------------
# Templates (no model needed)
# ---------------------------------------------------------------------------


def make_cloze(sentence: str, answer: str) -> Optional[str]:
    """Replace the first whole-word occurrence of ``answer`` with a blank."""
    pattern = re.compile(r"(?<![\w])" + re.escape(answer.strip()) + r"(?![\w])", re.IGNORECASE)
    if not pattern.search(sentence):
        return None
    cloze = pattern.sub(BLANK, sentence.strip(), count=1)
    if cloze.count(BLANK) != 1 or word_count(cloze) < 5:
        return None
    return cloze


def natural_case(term: str, notes: str = "") -> str:
    """Lower-case a term's first letter when it was only capitalised because it began a sentence.

    "A process" -> "a process"; "Elasticity" -> "elasticity" if the notes also
    write "elasticity" in lower case; "Linux", "SWOT analysis" and
    "Belady's anomaly" keep their capitals.
    """
    words = term.strip().split()
    if not words:
        return term
    first = words[0]
    if first.lower() in {"a", "an", "the"}:
        return " ".join([first.lower()] + words[1:])
    if first[:1].isupper() and first[1:].islower():
        # Capitalised in the middle of a sentence somewhere -> a proper noun such as "Linux".
        proper = notes and re.search(r"[a-z,;][ \t]+" + re.escape(first) + r"(?![\w])", notes)
        if not proper:
            return " ".join([first.lower()] + words[1:])
    return term.strip()


def definition_question(term: str, verb: str, notes: str = "") -> str:
    plural = verb.startswith("are") or verb == "refer to"
    return f"What {'are' if plural else 'is'} {natural_case(term, notes)}?"


def reverse_definition_question(definition: str, verb: str) -> str:
    """'Which of the following is a program in execution?' style question."""
    definition = definition.strip().rstrip(".")
    lead = {"is": "is", "are": "are", "refers to": "refers to", "refer to": "refer to", "means": "means",
            "is defined as": "is defined as", "are defined as": "are defined as",
            "can be defined as": "can be defined as", "is described as": "is described as",
            "is a term for": "is a term for"}.get(verb, "is")
    return f"Which of the following {lead} {definition[0].lower() + definition[1:]}?"


# ---------------------------------------------------------------------------
# Prompts and the main routine
# ---------------------------------------------------------------------------


def primary_prompt(sentence: str, answer: str) -> str:
    return (f'Write a question about the text below. The answer to the question must be "{answer}".\n\n'
            f"Text: {sentence}\n\nQuestion:")


def corrective_prompt(sentence: str, answer: str) -> str:
    return (f"{sentence}\n\nAsk one short question that starts with What, Which, Who, When, Where or How. "
            f'Do not include the words "{answer}" in the question, because "{answer}" is the answer.')


def verification_prompt(sentence: str, question: str) -> str:
    return f"Read this and answer the question.\n\n{sentence}\n\n{question}"


def write_questions(
    candidates: Sequence[Candidate],
    vocabulary: set,
    model=None,
    verify: bool = True,
    batch_size: int = 8,
) -> Dict[int, WrittenQuestion]:
    """Write a question for each statement candidate (index -> result).

    Candidates that cannot get a valid model question fall back to a cloze
    (fill-in-the-blank) question; if even that is impossible they are absent
    from the returned dict and the caller skips them.
    """
    results: Dict[int, WrittenQuestion] = {}
    rejected: Dict[int, int] = {i: 0 for i in range(len(candidates))}
    pending = list(range(len(candidates)))

    if model is not None:
        for attempt, build in (("model", primary_prompt), ("model-retry", corrective_prompt)):
            if not pending:
                break
            prompts = [build(candidates[i].sentence, candidates[i].answer) for i in pending]
            outputs = model.generate(prompts, max_new_tokens=40, min_new_tokens=4, num_beams=4,
                                     batch_size=batch_size, no_repeat_ngram_size=3)
            accepted: Dict[int, str] = {}
            for i, raw in zip(pending, outputs):
                question = parse_model_question(raw)
                reason = validate_question(question, candidates[i].answer, candidates[i].sentence, vocabulary)
                if reason is None:
                    accepted[i] = question  # type: ignore[assignment]
                else:
                    rejected[i] += 1
            if verify and accepted:
                order = list(accepted)
                checks = model.generate([verification_prompt(candidates[i].sentence, accepted[i]) for i in order],
                                        max_new_tokens=24, min_new_tokens=1, num_beams=1, batch_size=batch_size,
                                        no_repeat_ngram_size=0)
                for i, predicted in zip(order, checks):
                    if not answers_match(predicted, candidates[i].answer):
                        rejected[i] += 1
                        del accepted[i]
            for i, question in accepted.items():
                results[i] = WrittenQuestion(question, attempt, rejected[i])
            pending = [i for i in pending if i not in accepted]

    for i in pending:
        cloze = make_cloze(candidates[i].sentence, candidates[i].answer)
        if cloze:
            results[i] = WrittenQuestion(f"Fill in the blank: {cloze}", "cloze", rejected[i])
    return results
