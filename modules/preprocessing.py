"""Text preprocessing shared by every StudyMate AI feature.

Contents
--------
* ``clean_text``          – the exact normalisation used when the TF-IDF
                            classifier was trained (copied from the notebook).
* ``split_sentences``     – sentence segmentation that respects common
                            abbreviations, decimals, bullet points and headings.
* ``chunk_sentences``     – groups whole sentences into chunks that fit a
                            model's token limit (never cuts mid-sentence unless a
                            single sentence is longer than the limit).
* similarity helpers      – content-word overlap / Jaccard used for duplicate
                            detection and for checking that generated text is
                            supported by the original notes.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Sequence

# ---------------------------------------------------------------------------
# Classifier preprocessing (must stay identical to the notebook / train.py)
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Normalise text exactly as it was normalised for TF-IDF training.

    Copied verbatim from ``notebooks/StudyMate_AI.ipynb`` (section 5).
    Changing this function would silently change the classifier's inputs.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Basic counts
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")


def word_count(text: str) -> int:
    """Number of words in ``text`` (punctuation is ignored)."""
    return len(_WORD_RE.findall(text or ""))


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs but keep paragraph breaks (blank lines)."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ").replace("\t", " ")
    text = re.sub(r"[ \f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------

_ABBREVIATIONS = {
    "e.g", "i.e", "etc", "vs", "dr", "mr", "mrs", "ms", "prof", "sr", "jr",
    "fig", "figs", "no", "nos", "approx", "dept", "est", "inc", "ltd", "co",
    "corp", "st", "mt", "eq", "eqs", "ch", "sec", "vol", "pp", "p", "al",
    "cf", "min", "max", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec", "u.s", "u.k", "a.m", "p.m", "ph.d",
}

_BULLET_RE = re.compile(r"^\s*(?:[-–—•▪●◦*·]|\(?\d{1,2}[.)]|\(?[a-zA-Z][.)])\s+")
_BOUNDARY_RE = re.compile(r"([.!?]+[\"'”’)\]]*)\s+(?=[\"'“‘(\[]?[A-Z0-9])")


def _is_abbreviation(text_before: str) -> bool:
    """True if the full stop that ends ``text_before`` belongs to an abbreviation."""
    match = re.search(r"([A-Za-z][A-Za-z.]*)\.$", text_before)
    if not match:
        return False
    token = match.group(1).lower().rstrip(".")
    if token in _ABBREVIATIONS:
        return True
    # Single capital initial such as "J. K. Rowling" or "A. Turing"
    return len(token) == 1 and match.group(1).isupper()


def _split_unit(unit: str) -> List[str]:
    """Split one paragraph/bullet into sentences."""
    sentences: List[str] = []
    start = 0
    for match in _BOUNDARY_RE.finditer(unit):
        end = match.end(1)
        candidate = unit[start:end]
        if candidate.endswith(".") and _is_abbreviation(candidate):
            continue
        piece = candidate.strip()
        if piece:
            sentences.append(piece)
        start = match.end()
    tail = unit[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _paragraph_units(text: str) -> List[str]:
    """Turn raw notes into units: joined paragraphs, bullets and headings."""
    units: List[str] = []
    for block in re.split(r"\n\s*\n", normalize_whitespace(text)):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        current: List[str] = []
        for line in lines:
            is_bullet = bool(_BULLET_RE.match(line))
            if is_bullet:
                if current:
                    units.append(" ".join(current))
                    current = []
                units.append(_BULLET_RE.sub("", line, count=1).strip())
                continue
            # A short capitalised line without final punctuation, followed by a line
            # that starts with a capital letter, is a heading – not a wrapped line.
            if len(current) == 1 and re.search(r"[.!?:;,]$", current[0]) is None \
                    and len(current[0].split()) <= 8 and current[0][:1].isupper() and line[:1].isupper():
                units.append(current[0])
                current = []
            current.append(line)
        if current:
            units.append(" ".join(current))
    # Re-join words hyphenated across line breaks ("compu- ter" -> "computer")
    return [re.sub(r"(\w)- (\w)", r"\1\2", u) for u in units if u]


def split_sentences(text: str) -> List[str]:
    """Split notes into sentences.

    Bullets and headings become their own units, abbreviations such as
    "e.g." or "Dr." and decimals such as "3.5" do not end a sentence.
    """
    sentences: List[str] = []
    for unit in _paragraph_units(text):
        sentences.extend(s for s in _split_unit(unit) if re.search(r"[A-Za-z0-9]", s))
    return sentences


def split_paragraphs(text: str) -> List[str]:
    """Paragraphs separated by blank lines."""
    return [p.strip() for p in re.split(r"\n\s*\n", normalize_whitespace(text)) if p.strip()]


def looks_like_heading(sentence: str) -> bool:
    """A title or heading rather than a sentence.

    Either a short line without final punctuation ("3. Scheduling") or a line
    without final punctuation in which most words are capitalised
    ("Operating Systems — Lecture 3: Processes, Scheduling and Files").
    """
    text = sentence.strip()
    if re.search(r"[.!?][\"'”’)]*$", text):
        return False
    words = [w for w in re.findall(r"[A-Za-z][\w'’-]*", text)]
    if len(words) <= 8:
        return True
    long_words = [w for w in words if w.lower() not in STOPWORDS]
    capitalised = sum(1 for w in long_words if w[:1].isupper())
    return capitalised / max(1, len(long_words)) >= 0.5


# ---------------------------------------------------------------------------
# Token counting and chunking
# ---------------------------------------------------------------------------

TokenCounter = Callable[[str], int]


def approximate_token_count(text: str) -> int:
    """Conservative estimate of sub-word tokens for BART/T5 tokenizers.

    Used only when the real tokenizer is not available (e.g. in unit tests).
    English prose averages ~1.3 BPE tokens per word; technical notes run
    higher, so 1.5 tokens/word plus punctuation keeps estimates on the safe side.
    """
    words = len((text or "").split())
    punctuation = len(re.findall(r"[^\w\s]", text or ""))
    return int(math.ceil(words * 1.5 + punctuation * 0.5)) + 2


@dataclass
class Chunk:
    """A group of consecutive whole sentences."""

    index: int
    sentences: List[str]
    first_sentence: int  # index of the first sentence in the full document
    token_count: int = 0
    split_long_sentence: bool = False

    @property
    def text(self) -> str:
        return " ".join(self.sentences)

    @property
    def sentence_indices(self) -> range:
        return range(self.first_sentence, self.first_sentence + len(self.sentences))


def _split_long_sentence(sentence: str, max_tokens: int, count_tokens: TokenCounter) -> List[str]:
    """Split a single over-long sentence into word windows that fit the limit."""
    words = sentence.split()
    pieces: List[str] = []
    current: List[str] = []
    for word in words:
        trial = " ".join(current + [word])
        if current and count_tokens(trial) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_sentences(
    sentences: Sequence[str],
    max_tokens: int,
    count_tokens: TokenCounter = approximate_token_count,
    min_chunks: int = 1,
) -> List[Chunk]:
    """Group sentences into chunks of at most ``max_tokens`` tokens.

    * Sentences are never cut unless one sentence alone exceeds the limit.
    * ``min_chunks`` lets the caller force a document into several sections
      (used to stop a summariser from only covering the opening sentences).
    * Every input sentence appears in exactly one chunk, in order.
    """
    if max_tokens < 16:
        raise ValueError("max_tokens is too small to build meaningful chunks")

    units: List[tuple[str, bool]] = []
    for sentence in sentences:
        if count_tokens(sentence) > max_tokens:
            units.extend((piece, True) for piece in _split_long_sentence(sentence, max_tokens, count_tokens))
        else:
            units.append((sentence, False))

    if not units:
        return []

    # If the caller wants more sections than the token limit would produce,
    # lower the per-chunk budget so the text is spread evenly.
    total_tokens = sum(count_tokens(u) for u, _ in units)
    budget = max_tokens
    if min_chunks > 1 and len(units) >= min_chunks:
        budget = min(max_tokens, max(16, math.ceil(total_tokens / min_chunks)))

    chunks: List[Chunk] = []
    current: List[str] = []
    current_tokens = 0
    current_long = False
    first_index = 0
    sentence_pointer = 0
    for unit, was_split in units:
        unit_tokens = count_tokens(unit)
        if current and current_tokens + unit_tokens > budget:
            chunks.append(Chunk(len(chunks), current, first_index, current_tokens, current_long))
            first_index = sentence_pointer
            current, current_tokens, current_long = [], 0, False
        current.append(unit)
        current_tokens += unit_tokens
        current_long = current_long or was_split
        sentence_pointer += 1
    if current:
        chunks.append(Chunk(len(chunks), current, first_index, current_tokens, current_long))
    return chunks


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are aren't as at be because been before being
    below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down
    during each either etc few for from further had hadn't has hasn't have haven't having he he'd he'll he's
    her here here's hers herself him himself his how how's however i i'd i'll i'm i've if in into is isn't it
    it's its itself just let's may me might more most much must mustn't my myself no nor not now of off on
    once only or other ought our ours ourselves out over own same shall shan't she she'd she'll she's should
    shouldn't so some such than that that's the their theirs them themselves then there there's these they
    they'd they'll they're they've this those through thus to too under until up upon us use used uses using
    very via was wasn't we we'd we'll we're we've were weren't what what's when when's where where's whether
    which while who who's whom why why's will with within without won't would wouldn't yet you you'd you'll
    you're you've your yours yourself yourselves one two three many several often usually also e.g i.e
    called known example examples include includes including like well within across among per therefore
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")


def _stem(token: str) -> str:
    """Very small suffix stripper so that 'processes' ~ 'process'."""
    for suffix in ("ies", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3 and not token.endswith("ss"):
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def content_words(text: str) -> List[str]:
    """Lower-cased, stemmed, stop-word-free tokens."""
    return [_stem(t) for t in _TOKEN_RE.findall((text or "").lower()) if t not in STOPWORDS and len(t) > 1]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def text_similarity(a: str, b: str) -> float:
    """Jaccard similarity of content words."""
    return jaccard(content_words(a), content_words(b))


def support_ratio(generated: str, source_vocabulary: set) -> float:
    """Share of the generated text's content words that occur in the source.

    A low ratio means the generated sentence introduces many words that are
    not in the notes – a strong signal of invented content.
    """
    words = content_words(generated)
    if not words:
        return 0.0
    return sum(1 for w in words if w in source_vocabulary) / len(words)


def normalise_for_match(text: str) -> str:
    """Lower-case and strip punctuation, for substring checks."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


@dataclass
class DuplicateFilter:
    """Keeps items whose text is not too similar to anything already kept."""

    threshold: float = 0.7
    _kept: List[List[str]] = field(default_factory=list)

    def is_duplicate(self, text: str) -> bool:
        words = content_words(text)
        return any(jaccard(words, other) >= self.threshold for other in self._kept)

    def add(self, text: str) -> bool:
        """Add ``text``; return False (and do not add) if it is a duplicate."""
        if self.is_duplicate(text):
            return False
        self._kept.append(content_words(text))
        return True
