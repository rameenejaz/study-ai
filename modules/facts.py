"""Finding the facts that flashcards and MCQs are built from.

Everything here is deterministic and works directly on the student's notes,
which is what keeps the study material grounded: every answer is a phrase or
definition that literally appears in the notes.

Steps
-----
1. Split the notes into sentences and group them into sections (~250 words)
   so that facts are picked from the whole document, not just the opening.
2. Build a table of key phrases (recurring 1–3 word concepts, names,
   acronyms and numbers) for the whole document.
3. For every usable sentence create *candidates*:
   * definition candidates – "X is a/an/the …", "X refers to …",
     "… is called X", "Term: definition" glossary lines;
   * statement candidates – the sentence with one key phrase as the answer.
4. Rank candidates inside each section and interleave the sections
   (round-robin) so a request for 10 items samples the entire document.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from modules.preprocessing import (
    STOPWORDS,
    chunk_sentences,
    content_words,
    looks_like_heading,
    normalise_for_match,
    split_sentences,
    word_count,
)

SECTION_WORDS = 250

_PRONOUN_STARTS = {
    "it", "this", "that", "these", "those", "they", "them", "he", "she", "we", "you", "i", "there", "here",
    "such", "its", "their", "our", "his", "her", "each", "every", "both", "all", "some", "many", "most",
    "another", "other", "others", "one", "however", "also", "then", "so", "thus", "therefore", "but", "and",
    "for", "if", "when", "while", "although", "because", "as",
}

# Common verbs (and their inflections) – used to keep verbs out of answer phrases.
_BASE_VERBS = """
accept access achieve act add affect allow apply arrange assign avoid begin believe belong break bring build
buy calculate call carry cause change check choose claim combine come compare compete complete compute connect
consider consist contain continue control convert copy cost count create cut deal decide decrease define
delete deliver depend describe design detect determine develop differ divide do drive earn ensure enter
estimate evaluate exist expand explain express fail fall feel fill find finish fit focus follow form gain
generate get give go grow handle happen help hold identify implement improve increase influence involve
keep know lead learn leave let lie limit link listen live load look lose maintain make manage map mean measure
meet move need occur offer open operate organise organize own pass pay perform place plan play point predict
prefer prepare present prevent process produce protect provide pull push put raise reach read receive reduce
refer reflect relate release remain remove repeat replace represent require reserve respond rest result
return reveal run save say score see seek select sell send separate serve set share show sign solve sort
speak spend split start stay stop store study suggest supply support switch take talk teach tell tend test
think train transfer treat try turn understand update use vary view wait want win work write yield
rent matter close become intervene examine rank propose award commit review govern found popularise popularize
compress approximate stripe mirror spread record repair satisfy acquire introduce grant allocate shuffle
eliminate translate organise direct motivate communicate subtract identify execute stay touch request last
switch perform release assign ensure solve range behave approximate reduce add migrate deploy build buy
""".split()

_IRREGULAR = """is are was were be been being has have had does did done made took taken gave given went gone
became become began begun brought built bought chose chosen came felt found got gotten grew grown held kept knew
known led left lost meant met paid put ran read said saw seen sent set shown sold spent split stood taught told
thought understood won wrote written""".split()


def _inflect(verb: str) -> set:
    forms = {verb, verb + "s", verb + "es", verb + "ed", verb + "d", verb + "ing"}
    if verb.endswith("e"):
        forms |= {verb[:-1] + "ing", verb + "d"}
    if verb.endswith("y"):
        forms |= {verb[:-1] + "ies", verb[:-1] + "ied"}
    return forms


VERB_FORMS = frozenset({form for v in _BASE_VERBS for form in _inflect(v)} | set(_IRREGULAR))
# Nouns that look like verb forms and should stay usable in phrases.
_NOUN_EXCEPTIONS = {"learning", "processing", "scheduling", "computing", "programming", "training", "testing",
                    "marketing", "accounting", "budgeting", "planning", "networking", "paging", "caching",
                    "swapping", "threading", "heading", "pricing", "branding", "financing", "set", "process",
                    "processes", "control", "design", "result", "results", "support", "return", "returns",
                    "cost", "costs", "score", "change", "changes", "work", "point", "points", "plan", "plans",
                    "limit", "limits", "map", "maps", "test", "tests", "view", "views", "sort", "share",
                    "shares", "supply", "switch", "store", "stores", "form", "forms", "need", "needs", "use",
                    "access", "transfer", "load", "release", "update", "split", "pass", "offer", "sign",
                    "study", "studies", "model", "models", "yield", "influence", "rest", "play", "turn", "operating",
                    "waiting", "running", "computing", "record", "records", "request", "requests", "range",
                    "review", "award", "mirror", "spread", "rank", "ranks", "direct", "found"}

_GENERIC = {"way", "ways", "thing", "things", "lot", "kind", "kinds", "part", "parts", "case", "cases", "time",
            "times", "fact", "idea", "ideas", "number", "numbers", "example", "examples", "type", "types",
            "level", "levels", "important", "different", "various", "certain", "main", "new", "good", "large",
            "small", "high", "low", "first", "second", "third", "able", "possible", "common", "key", "basic",
            "same", "other", "own", "general", "specific", "particular", "whole", "similar", "simple", "lecture",
            "notes", "chapter", "section", "topic", "student", "students", "following", "instead", "rather"}

_PHRASE_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'’\-/]*|\d+(?:[.,]\d+)?%?")
_NUMBER_RE = re.compile(
    r"\b(?:1[0-9]{3}|20[0-9]{2})s?\b|\b\d+(?:[.,]\d+)?\s?(?:%|percent|per cent|kb|mb|gb|tb|ms|seconds?|minutes?|"
    r"hours?|days?|years?|players?|teams?|bits?|bytes?|million|billion|thousand)\b",
    re.IGNORECASE,
)
_NAME_RE = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:of\s+|de\s+|the\s+)?(?:[A-Z][a-z]+|[A-Z]{2,}))+\b")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}s?\b")

_DEF_VERBS = r"is defined as|are defined as|can be defined as|is described as|refers to|refer to|is a term for|" \
             r"means|is|are"
_DEF_RE = re.compile(
    rf"^(?P<term>[A-Za-z][\w\-/'’ ]{{1,60}}?)(?:\s*\((?P<abbr>[A-Za-z0-9\-]{{2,10}})\))?\s+"
    rf"(?P<verb>{_DEF_VERBS})\s+(?P<definition>.+?)[.!]?$"
)
_CALLED_RE = re.compile(
    r"^(?P<definition>.{10,}?)\s+(?P<verb>is called|are called|is known as|are known as|is termed)\s+"
    r"(?P<term>[A-Za-z][\w\-/'’ ]{1,50}?)[.!]?$"
)
_GLOSSARY_RE = re.compile(r"^(?P<term>[A-Z][\w\-/'’ ()]{1,50}?)\s*(?::|\s[–—-])\s+(?P<definition>.{15,}?)[.!]?$")
_IS_DEF_START = re.compile(r"^(?:a|an|the|one|any|used|responsible|defined|known|considered|made|composed|"
                           r"based|essentially|simply|basically)\b", re.IGNORECASE)


@dataclass
class Phrase:
    text: str          # display form as it appears in the notes
    kind: str          # concept | name | acronym | number | term
    score: float
    count: int


@dataclass
class Candidate:
    """One possible flashcard/MCQ: a sentence plus the answer it supports."""

    sentence: str
    section: int
    position: int
    style: str                 # "definition" or "statement"
    answer: str                # phrase/definition that literally occurs in the notes
    answer_kind: str
    score: float
    term: str = ""
    definition: str = ""
    verb: str = ""

    @property
    def key(self) -> tuple:
        return (self.position, normalise_for_match(self.answer))


@dataclass
class NoteAnalysis:
    sentences: List[str]
    section_of: Dict[int, int]
    sections: int
    phrases: Dict[str, Phrase]
    candidates: List[Candidate] = field(default_factory=list)
    vocabulary: set = field(default_factory=set)
    normalised_text: str = ""

    def phrases_of_kind(self, kinds: Iterable[str]) -> List[Phrase]:
        kinds = set(kinds)
        return [p for p in self.phrases.values() if p.kind in kinds]


# ---------------------------------------------------------------------------
# Key phrases
# ---------------------------------------------------------------------------


def _is_breaker(word: str) -> bool:
    lower = word.lower()
    if lower in STOPWORDS or lower in _GENERIC:
        return True
    if lower in VERB_FORMS and lower not in _NOUN_EXCEPTIONS:
        return True
    return lower.endswith("ly") and len(lower) > 4


_GERUND_AFTER = {"for", "before", "after", "by", "of", "in", "without", "while", "when", "and", "or", "to", "from",
                 "on", "about", "into", "through", "than"}


def _candidate_runs(sentence: str) -> List[str]:
    """Runs of non-stop, non-verb words (later cut into 1–3 word n-grams).

    Runs never cross punctuation, so list items ("head, body or feet") and
    parenthesised abbreviations are not glued into nonsense phrases.
    """
    runs: List[List[str]] = []
    for segment in re.split(r"[,;:()\[\]\"“”—–]|\s-\s|\.\s", sentence):
        words = _PHRASE_WORD_RE.findall(segment)
        run: List[str] = []
        previous = ""
        for word in words:
            lower = word.lower()
            gerund = lower.endswith("ing") and len(lower) > 5 and lower not in _NOUN_EXCEPTIONS \
                and previous in _GERUND_AFTER
            if _is_breaker(word) or gerund or re.fullmatch(r"\d+(?:[.,]\d+)?%?", word):
                if run:
                    runs.append(run)
                run = []
            else:
                run.append(word.strip("'’-/"))
                if re.search(r"['’]s$", word):  # a possessive ends the phrase
                    runs.append(run[:-1])
                    run = []
            previous = lower
        if run:
            runs.append(run)
    return [" ".join(r) for r in runs if r]


def _candidate_ngrams(sentence: str) -> List[str]:
    grams: List[str] = []
    runs = [r.split() for r in _candidate_runs(sentence)]
    for run in runs:
        for n in (3, 2, 1):
            for i in range(len(run) - n + 1):
                gram = run[i:i + n]
                if all(len(w) >= 2 for w in gram) and any(len(w) >= 4 for w in gram):
                    grams.append(" ".join(gram))
    return grams


def _run_ends(sentence: str) -> List[str]:
    return [run.split()[-1] for run in _candidate_runs(sentence)]


def extract_key_phrases(sentences: Sequence[str]) -> Dict[str, Phrase]:
    counts: Counter = Counter()
    heads: Counter = Counter()  # how often a single word ends a phrase (i.e. is used as a noun head)
    display: Dict[str, str] = {}
    sentences = [s for s in sentences if not looks_like_heading(s)]
    for sentence in sentences:
        grams = _candidate_ngrams(sentence)
        for gram in set(grams):
            key = gram.lower()
            counts[key] += 1
            display.setdefault(key, gram)
        for run_end in _run_ends(sentence):
            heads[run_end.lower()] += 1
    phrases: Dict[str, Phrase] = {}
    for key, count in counts.items():
        n_words = len(key.split())
        if n_words == 1 and count < 2 and not display[key].isupper():
            continue  # one-off single words are rarely concepts
        if n_words == 3 and count < 2:
            continue  # long one-off word runs are usually fragments of a clause
        if n_words == 1 and heads[key] == 0:
            continue  # always modifies another word ("operating" system, "virtual" memory)
        score = (1.0 + math.log(count)) * (1.0 + 0.8 * (n_words - 1))
        phrases[key] = Phrase(display[key], "concept", score, count)
    # Drop a shorter phrase that only ever appears inside a longer one
    # ("operating" inside "operating system").
    for key in list(phrases):
        if any(other != key and f" {key} " in f" {other} " and phrases[other].count >= phrases[key].count
               for other in phrases):
            del phrases[key]

    for sentence in sentences:
        for match in _NUMBER_RE.finditer(sentence):
            text = match.group(0).strip()
            phrases.setdefault(text.lower(), Phrase(text, "number", 1.5, 0)).count += 1
        for match in _NAME_RE.finditer(sentence):
            text = match.group(0)
            first = text.split()[0].lower()
            if match.start() == 0 and (first in STOPWORDS or first in _PRONOUN_STARTS):
                continue
            if all(w.lower() in STOPWORDS for w in text.split()):
                continue
            entry = phrases.setdefault(text.lower(), Phrase(text, "name", 2.0, 0))
            entry.kind, entry.count = "name", entry.count + 1
        for match in _ACRONYM_RE.finditer(sentence):
            text = match.group(0)
            if text.rstrip("s") in {"I", "A"}:
                continue
            entry = phrases.setdefault(text.lower(), Phrase(text, "acronym", 1.8, 0))
            entry.kind, entry.count = "acronym", entry.count + 1
    return phrases


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------


def _clean_term(term: str) -> str:
    return re.sub(r"\s+", " ", term).strip(" ,;:-–—\"'“”")


def _valid_term(term: str) -> bool:
    words = term.split()
    if not 1 <= len(words) <= 6:
        return False
    if words[0].lower() in _PRONOUN_STARTS:
        return False
    if not content_words(term):
        return False
    # Without an article, a leading verb form ("Using X", "Makes …") is not a term.
    if words[0].lower() not in {"a", "an", "the"} and words[0].lower() in VERB_FORMS \
            and words[0].lower() not in _NOUN_EXCEPTIONS:
        return False
    return not any(w.lower() in {"which", "who", "that", "where", "when"} for w in words)


def _trim_definition(definition: str) -> str:
    """Cut a contrasting second clause ("…, while a physical address is …")."""
    return re.split(r",\s+(?:while|whereas|but)\s|;\s|,\s+and\s+(?:a|an|the|it|its|they|this|these)\s", definition,
                    maxsplit=1)[0].strip().rstrip(".")


def find_definition(sentence: str) -> Optional[dict]:
    """Return {term, definition, verb} if the sentence defines something."""
    text = sentence.strip()
    match = _CALLED_RE.match(text)
    if match:
        term, definition = _clean_term(match.group("term")), match.group("definition").strip()
        if _valid_term(term) and word_count(definition) >= 4 and normalise_for_match(term) not in \
                normalise_for_match(definition):
            return {"term": term, "definition": definition, "verb": "is"}
    match = _DEF_RE.match(text)
    if match:
        term = _clean_term(match.group("term"))
        definition = _trim_definition(match.group("definition"))
        verb = match.group("verb")
        if verb in {"is", "are"} and not _IS_DEF_START.match(definition):
            return None
        if _valid_term(term) and word_count(definition) >= 4 and \
                normalise_for_match(term) not in normalise_for_match(definition):
            return {"term": term, "definition": definition, "verb": verb}
    match = _GLOSSARY_RE.match(text)
    if match:
        term, definition = _clean_term(match.group("term")), match.group("definition").strip()
        def_words = definition.split()
        title_case = sum(1 for w in def_words if w[:1].isupper()) / max(1, len(def_words)) >= 0.5
        if _valid_term(term) and word_count(definition) >= 5 and len(term.split()) <= 4 and not title_case:
            return {"term": term, "definition": definition, "verb": "is"}
    return None


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


def _usable_sentence(sentence: str) -> bool:
    n = word_count(sentence)
    if n < 6 or n > 50 or looks_like_heading(sentence) or sentence.rstrip().endswith("?"):
        return False
    first = re.sub(r"[^a-z]", "", sentence.split()[0].lower()) if sentence.split() else ""
    return first not in _PRONOUN_STARTS


def _answer_candidates(sentence: str, phrases: Dict[str, Phrase]) -> List[Phrase]:
    """Key phrases that occur in this sentence, best first."""
    normal = f" {normalise_for_match(sentence)} "
    found: List[Phrase] = []
    for key, phrase in phrases.items():
        needle = f" {normalise_for_match(phrase.text)} "
        if needle.strip() and needle in normal:
            found.append(phrase)
    # Drop phrases contained in a longer found phrase (keep the most specific).
    found.sort(key=lambda p: (-len(p.text.split()), -p.score))
    kept: List[Phrase] = []
    for phrase in found:
        norm = normalise_for_match(phrase.text)
        if any(f" {norm} " in f" {normalise_for_match(k.text)} " for k in kept):
            continue
        kept.append(phrase)
    kind_bonus = {"name": 1.0, "number": 0.8, "acronym": 0.6, "concept": 0.0}
    kept.sort(key=lambda p: -(p.score + kind_bonus.get(p.kind, 0)))
    return kept


def analyse_notes(text: str, section_words: int = SECTION_WORDS) -> NoteAnalysis:
    sentences = split_sentences(text)
    chunks = chunk_sentences(sentences, max_tokens=section_words, count_tokens=lambda s: max(1, len(s.split())))
    section_of = {i: chunk.index for chunk in chunks for i in chunk.sentence_indices}
    sentences = [s for chunk in chunks for s in chunk.sentences]  # identical unless a sentence was split
    phrases = extract_key_phrases(sentences)
    for sentence in sentences:  # defined terms are the strongest concepts in the notes
        definition = find_definition(sentence) if _usable_sentence(sentence) or _GLOSSARY_RE.match(sentence) else None
        if definition:
            term = re.sub(r"^(?:a|an|the)\s+", "", definition["term"], flags=re.IGNORECASE)
            entry = phrases.get(term.lower())
            if entry is None or entry.kind == "concept":
                phrases[term.lower()] = Phrase(term, "term", 5.0 + (entry.score if entry else 0), max(1, entry.count if entry else 1))
    analysis = NoteAnalysis(sentences, section_of, len(chunks), phrases,
                            vocabulary=set(content_words(text)), normalised_text=normalise_for_match(text))

    for position, sentence in enumerate(sentences):
        if not _usable_sentence(sentence) and not _GLOSSARY_RE.match(sentence):
            continue
        section = section_of.get(position, 0)
        length_penalty = 0.0 if word_count(sentence) <= 35 else 0.5
        definition = find_definition(sentence)
        if definition:
            analysis.candidates.append(Candidate(
                sentence, section, position, "definition", definition["definition"], "definition",
                score=4.0 - length_penalty, term=definition["term"], definition=definition["definition"],
                verb=definition["verb"],
            ))
        if not _usable_sentence(sentence):
            continue
        answers = _answer_candidates(sentence, phrases)
        for rank, phrase in enumerate(answers[:3]):
            surface = surface_form(sentence, phrase.text)
            if definition and normalise_for_match(phrase.text) == normalise_for_match(definition["term"]):
                continue  # already covered by the definition candidate
            if not surface or word_count(surface) >= word_count(sentence) - 3:
                continue
            analysis.candidates.append(Candidate(
                sentence, section, position, "statement", surface, phrase.kind,
                score=1.0 + min(phrase.score, 6.0) / 2 - rank * 0.8 - length_penalty,
            ))
    return analysis


def interleave_by_section(candidates: Sequence[Candidate]) -> List[Candidate]:
    """Best candidate of section 1, of section 2, …, then second-best of each, etc.

    Within a pass, the first candidate per sentence is preferred, so one sentence
    does not produce several items before every sentence has produced one.
    """
    by_section: Dict[int, List[Candidate]] = defaultdict(list)
    for cand in candidates:
        by_section[cand.section].append(cand)
    queues = []
    for section in sorted(by_section):
        items = sorted(by_section[section], key=lambda c: -c.score)
        first_per_sentence, rest, seen = [], [], set()
        for item in items:
            (rest if item.position in seen else first_per_sentence).append(item)
            seen.add(item.position)
        queues.append(first_per_sentence + rest)
    ordered: List[Candidate] = []
    while any(queues):
        for queue in queues:
            if queue:
                ordered.append(queue.pop(0))
    return ordered


def surface_form(sentence: str, phrase: str) -> Optional[str]:
    """The phrase exactly as written in ``sentence`` (keeps that sentence's capitalisation)."""
    pattern = re.compile(r"(?<![\w])" + r"\W+".join(re.escape(w) for w in phrase.split()) + r"(?![\w])", re.IGNORECASE)
    match = pattern.search(sentence)
    return match.group(0) if match else None


def answer_in_notes(answer: str, analysis: NoteAnalysis) -> bool:
    norm = normalise_for_match(answer)
    return bool(norm) and f" {norm} " in f" {analysis.normalised_text} "
