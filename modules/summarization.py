"""Coverage-preserving summarisation of lecture notes (map -> check -> reduce).

Why the old version failed
--------------------------
* It cut the notes every 550 *words* (not sentences) and used only the first
  10 chunks, so anything after ~5,500 words was silently dropped.
* It used the same small ``max_length`` for every chunk and again for the
  final pass, so a long document was squeezed into ~120 tokens in one go.
* DistilBART-CNN was trained on news articles, where the first sentences
  carry the story ("lead bias"). Given a whole short note as one input it
  tends to copy the opening sentences, which is why a 118-word note came back
  as its first paragraph.

What this module does
---------------------
1. Split the notes into sentences and group whole sentences into sections that
   fit the model's real token limit (measured with the model's tokenizer).
   Short notes are still split into at least 2–3 sections so the model must
   summarise every part, not just the opening.
2. Summarise **every** section with a length budget derived from the chosen
   length (Short / Medium / Detailed) and the section's size.
3. Check each generated sentence: drop fragments, drop sentences whose words
   are mostly absent from the notes (likely invented), and drop repeats.
   A section whose summary is entirely rejected gets its most representative
   original sentence instead, so no section is ever silently lost.
4. If the combined summary is still well over the target length, run a second
   (reduce) pass section-group by section-group, which shortens the text
   without collapsing it to the opening lines.

Length control: ``max_new_tokens`` is an upper bound, not a promise. The model
may stop earlier. Therefore each preset also sets ``min_new_tokens``, the
section size and a target word budget enforced by the reduce pass, and the UI
reports the word count actually produced.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence

from modules.preprocessing import (
    Chunk,
    DuplicateFilter,
    chunk_sentences,
    content_words,
    looks_like_heading,
    split_sentences,
    support_ratio,
    text_similarity,
    word_count,
)

MIN_INPUT_WORDS = 40


class SummarizationError(Exception):
    """User-facing summarisation failure."""


class SummaryModel(Protocol):
    name: str
    max_input_tokens: int

    def count_tokens(self, text: str) -> int: ...

    def generate(self, prompts: Sequence[str], max_new_tokens: int, min_new_tokens: int = 0,
                 num_beams: int = 4, batch_size: int = 4, no_repeat_ngram_size: int = 3,
                 length_penalty: float = 1.0) -> List[str]: ...


@dataclass(frozen=True)
class LengthPreset:
    name: str
    target_ratio: float     # target summary length as a share of the input words
    min_words: int          # lower bound for the target
    max_words: int          # upper bound for the target
    section_tokens: int     # preferred section size (smaller = more detail kept)
    section_ratio: float    # per-section output budget relative to section words
    min_sections: int       # minimum number of sections for short notes
    description: str


LENGTH_PRESETS: Dict[str, LengthPreset] = {
    "Short": LengthPreset("Short", 0.15, 35, 180, 700, 0.22, 2,
                          "Key ideas only — roughly 15% of the notes."),
    "Medium": LengthPreset("Medium", 0.28, 60, 400, 480, 0.35, 2,
                           "Main ideas with supporting points — roughly 25–30% of the notes."),
    "Detailed": LengthPreset("Detailed", 0.45, 90, 900, 320, 0.50, 3,
                             "Definitions, explanations and details — roughly 40–50% of the notes."),
}


@dataclass
class SectionSummary:
    section: int
    source_words: int
    max_new_tokens: int
    min_new_tokens: int
    raw_output: str
    kept_sentences: List[str]
    rejected: List[str] = field(default_factory=list)
    used_fallback: bool = False


@dataclass
class SummaryResult:
    summary: str
    length: str
    input_words: int
    target_words: int
    sections_total: int
    sections_summarised: int
    sections_covered: int
    passes: int
    model_name: str
    section_details: List[SectionSummary] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return word_count(self.summary)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def get_preset(length: str) -> LengthPreset:
    try:
        return LENGTH_PRESETS[length]
    except KeyError as exc:
        raise ValueError(f"Unknown summary length '{length}'. Choose one of {list(LENGTH_PRESETS)}.") from exc


def target_word_count(input_words: int, preset: LengthPreset) -> int:
    target = round(input_words * preset.target_ratio)
    target = max(preset.min_words, min(preset.max_words, target))
    return max(15, min(target, int(input_words * 0.6)))  # a summary must be clearly shorter than the notes


def section_budget(section_words: int, section_tokens: int, ratio: float) -> tuple[int, int]:
    """(max_new_tokens, min_new_tokens) for one section.

    ~1.4 tokens per output word. The minimum is never more than 80% of the
    section itself, so the model is never forced to pad a tiny section.
    """
    target_words = max(12, section_words * ratio)
    max_new = int(min(320, max(28, target_words * 1.4 * 1.35 + 8)))
    min_new = int(min(target_words * 1.4 * 0.75, section_tokens * 0.8, max_new - 8))
    return max_new, max(8, min_new)


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def tidy_generated_text(text: str) -> str:
    """Fix spacing artefacts typical of BART output (" ." / "( x )")."""
    text = re.sub(r"\s+([.,;:!?%)])", r"\1", text or "")
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _is_fragment(sentence: str) -> bool:
    return word_count(sentence) < 4 or not re.search(r"[.!?][\"'”’)]*$", sentence.strip())


def _representative_sentence(chunk: Chunk) -> str:
    """The sentence most similar to the rest of its section (a centroid pick)."""
    candidates = [s for s in chunk.sentences if word_count(s) >= 6] or list(chunk.sentences)
    if len(candidates) == 1:
        return candidates[0]
    section_words = content_words(chunk.text)
    best, best_score = candidates[0], -1.0
    for sentence in candidates:
        words = set(content_words(sentence))
        if not words:
            continue
        score = sum(1 for w in section_words if w in words) / (len(words) ** 0.5)
        if score > best_score:
            best, best_score = sentence, score
    return best


def filter_generated_sentences(
    generated: str,
    source_vocabulary: set,
    duplicate_filter: DuplicateFilter,
    min_support: float = 0.7,
) -> tuple[List[str], List[str]]:
    """Return (kept, rejected) sentences from one model output."""
    kept: List[str] = []
    rejected: List[str] = []
    sentences = split_sentences(tidy_generated_text(generated))
    for i, sentence in enumerate(sentences):
        is_last = i == len(sentences) - 1
        if _is_fragment(sentence):
            # A cut-off final sentence is a token-limit artefact: drop it if others exist.
            if is_last and kept and word_count(sentence) >= 4:
                rejected.append(sentence)
                continue
            if word_count(sentence) < 4:
                rejected.append(sentence)
                continue
            sentence = sentence.rstrip(",;: ") + "."
        if support_ratio(sentence, source_vocabulary) < min_support:
            rejected.append(sentence)  # mostly words that are not in the notes
            continue
        if not duplicate_filter.add(sentence):
            rejected.append(sentence)
            continue
        kept.append(sentence)
    return kept, rejected


def _covered_sections(summary_sentences: Sequence[str], chunks: Sequence[Chunk]) -> int:
    """How many source sections have at least one summary sentence that matches them best."""
    covered = set()
    for sentence in summary_sentences:
        best_chunk, best_score = None, 0.0
        for chunk in chunks:
            for source in chunk.sentences:
                score = text_similarity(sentence, source)
                if score > best_score:
                    best_chunk, best_score = chunk.index, score
        if best_chunk is not None and best_score >= 0.15:
            covered.add(best_chunk)
    return len(covered)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _prompt(model: SummaryModel, text: str) -> str:
    return f"summarize: {text}" if "t5" in model.name.lower() else text


def _plan_sections(sentences: List[str], model: SummaryModel, preset: LengthPreset) -> List[Chunk]:
    limit = model.max_input_tokens - 16  # room for special tokens / prompt prefix
    section_tokens = min(preset.section_tokens, limit)
    total_words = sum(word_count(s) for s in sentences)
    # Force short notes into a few sections (each >= ~35 words) to counter lead bias.
    min_sections = max(1, min(preset.min_sections, total_words // 35, len(sentences) // 2))
    count = (lambda s: model.count_tokens(_prompt(model, s)))
    return chunk_sentences(sentences, section_tokens, count, min_chunks=min_sections)


def _summarise_sections(
    model: SummaryModel,
    chunks: Sequence[Chunk],
    ratio: float,
    vocabulary: set,
    duplicate_filter: DuplicateFilter,
    allow_fallback: bool,
) -> List[SectionSummary]:
    details: List[SectionSummary] = []
    # Sections with similar budgets are batched together for speed.
    budgets = [section_budget(word_count(c.text), c.token_count, ratio) for c in chunks]
    outputs: List[Optional[str]] = [None] * len(chunks)
    by_budget: Dict[tuple, List[int]] = {}
    for i, budget in enumerate(budgets):
        by_budget.setdefault(budget, []).append(i)
    for (max_new, min_new), indices in by_budget.items():
        prompts = [_prompt(model, chunks[i].text) for i in indices]
        results = model.generate(prompts, max_new_tokens=max_new, min_new_tokens=min_new,
                                 num_beams=4, no_repeat_ngram_size=3, length_penalty=1.0)
        if len(results) != len(prompts):
            raise SummarizationError("The summariser returned the wrong number of outputs.")
        for i, out in zip(indices, results):
            outputs[i] = out

    for chunk, (max_new, min_new), raw in zip(chunks, budgets, outputs):
        kept, rejected = filter_generated_sentences(raw or "", vocabulary, duplicate_filter)
        detail = SectionSummary(chunk.index + 1, word_count(chunk.text), max_new, min_new, raw or "", kept, rejected)
        if not kept and allow_fallback:
            fallback = _representative_sentence(chunk)
            if duplicate_filter.add(fallback):
                detail.kept_sentences = [fallback]
                detail.used_fallback = True
        details.append(detail)
    return details


def summarize(text: str, length: str, model: SummaryModel, max_reduce_passes: int = 2) -> SummaryResult:
    """Summarise ``text`` with the chosen length preset. Every section is processed."""
    preset = get_preset(length)
    input_words = word_count(text)
    if input_words < MIN_INPUT_WORDS:
        raise SummarizationError(
            f"The notes are too short to summarise ({input_words} words). Add at least {MIN_INPUT_WORDS} words."
        )
    # Headings ("3. CPU Scheduling") carry no statements and would run into the
    # next sentence without punctuation, so only real sentences are summarised.
    sentences = [s for s in split_sentences(text) if not looks_like_heading(s)]
    if not sentences:
        raise SummarizationError("No complete sentences were found in the notes.")

    vocabulary = set(content_words(text))
    target = target_word_count(input_words, preset)
    chunks = _plan_sections(sentences, model, preset)
    warnings: List[str] = []
    if any(c.split_long_sentence for c in chunks):
        warnings.append("A very long sentence (or text without punctuation) had to be split by words.")

    try:
        details = _summarise_sections(model, chunks, preset.section_ratio, vocabulary,
                                      DuplicateFilter(0.6), allow_fallback=True)
    except SummarizationError:
        raise
    except Exception as exc:
        raise SummarizationError(f"The summarisation model failed: {exc}") from exc

    fallback_sections = [d.section for d in details if d.used_fallback]
    if fallback_sections:
        warnings.append(
            "For section(s) " + ", ".join(map(str, fallback_sections)) + " the model's output was rejected "
            "(unsupported or empty), so the most representative original sentence was used instead."
        )
    paragraphs = [d.kept_sentences for d in details if d.kept_sentences]
    if not paragraphs:
        raise SummarizationError("The model did not produce a usable summary for any section.")

    passes = 1
    current_words = sum(word_count(s) for p in paragraphs for s in p)
    while current_words > target * 1.35 and passes <= max_reduce_passes:
        # Reduce pass: compress groups of section summaries separately so that
        # later sections are not squeezed out by the opening ones.
        all_sentences = [s for p in paragraphs for s in p]
        groups_wanted = max(1, min(len(paragraphs), math.ceil(target / 70)))
        limit = model.max_input_tokens - 16
        count = (lambda s: model.count_tokens(_prompt(model, s)))
        groups = chunk_sentences(all_sentences, limit, count, min_chunks=groups_wanted)
        ratio = max(0.15, min(0.9, target / current_words))
        try:
            reduce_details = _summarise_sections(model, groups, ratio, vocabulary, DuplicateFilter(0.6),
                                                 allow_fallback=True)
        except Exception as exc:
            raise SummarizationError(f"The summarisation model failed in the reduce pass: {exc}") from exc
        new_paragraphs = [d.kept_sentences for d in reduce_details if d.kept_sentences]
        new_words = sum(word_count(s) for p in new_paragraphs for s in p)
        passes += 1
        if not new_paragraphs or new_words >= current_words:
            warnings.append("The second pass could not shorten the summary further; the section summaries are shown.")
            break
        paragraphs, current_words = new_paragraphs, new_words

    final_sentences = [s for p in paragraphs for s in p]
    covered = _covered_sections(final_sentences, chunks)
    if covered < len(chunks):
        warnings.append(
            f"The final summary draws on {covered} of {len(chunks)} sections. Choose a longer summary "
            "length to keep more of the notes."
        )
    if current_words < target * 0.5:
        warnings.append(
            f"The summary is shorter than the target (~{target} words) because the model stopped early or "
            "some generated sentences were rejected as unsupported."
        )

    return SummaryResult(
        summary="\n\n".join(" ".join(p) for p in paragraphs),
        length=preset.name,
        input_words=input_words,
        target_words=target,
        sections_total=len(chunks),
        sections_summarised=len(details),
        sections_covered=covered,
        passes=passes,
        model_name=model.name,
        section_details=details,
        warnings=warnings,
    )
