import pytest

from fakes import FailingModel, LeadBiasedSummarizer
from modules.preprocessing import chunk_sentences, looks_like_heading, split_sentences, text_similarity, word_count
from modules.summarization import (
    LENGTH_PRESETS,
    SummarizationError,
    filter_generated_sentences,
    section_budget,
    summarize,
    target_word_count,
)
from modules.preprocessing import DuplicateFilter, content_words


def _paragraphs(text):
    return [p for p in text.split("\n\n") if p.strip()]


def test_short_note_summary(short_notes):
    result = summarize(short_notes, "Short", LeadBiasedSummarizer())
    assert result.summary.strip()
    assert result.word_count < word_count(short_notes)
    assert result.sections_summarised == result.sections_total


def test_acceptance_118_word_note_is_not_just_first_paragraph(short_notes):
    """Acceptance 1: even a lead-biased model must not reduce the note to its first paragraph."""
    assert word_count(short_notes) == 118
    first_paragraph = _paragraphs(short_notes)[0]
    later_text = " ".join(_paragraphs(short_notes)[1:])
    for length in LENGTH_PRESETS:
        result = summarize(short_notes, length, LeadBiasedSummarizer())
        summary_sentences = split_sentences(result.summary)
        from_later = [s for s in summary_sentences if s in later_text]
        assert from_later, f"{length}: summary only covers the opening paragraph"
        assert result.summary.strip() != first_paragraph.strip()
        assert result.sections_total >= 2


def test_acceptance_2000_word_note_uses_every_chunk(os_notes):
    """Acceptance 2: every sentence of a ~2,000-word note is sent to the model in some section."""
    assert word_count(os_notes) >= 2000
    model = LeadBiasedSummarizer()
    result = summarize(os_notes, "Medium", model)
    first_pass_prompts = model.calls[0]["prompts"] if len(model.calls) == 1 else \
        [p for call in model.calls[: result.sections_total] for p in call["prompts"]]
    sent = " ".join(model.all_prompts)
    body = [s for s in split_sentences(os_notes) if not looks_like_heading(s)]
    assert len(body) > 100
    for sentence in body:
        assert sentence in sent, f"sentence never reached the model: {sentence[:60]}"
    assert result.sections_summarised == result.sections_total >= 5
    assert len(first_pass_prompts) >= 1


def test_summary_covers_sections_across_the_document(os_notes):
    result = summarize(os_notes, "Detailed", LeadBiasedSummarizer())
    assert result.sections_covered >= result.sections_total - 1
    # Content from the end of the notes (file systems / RAID) must be present.
    assert "RAID" in result.summary or "file" in result.summary.lower()


def test_chunks_respect_token_limit_and_sentence_boundaries(os_notes):
    model = LeadBiasedSummarizer(max_input_tokens=300)
    summarize(os_notes, "Short", model)  # FakeModelBase asserts every prompt fits the limit
    sentences = set(split_sentences(os_notes))
    for call in model.calls[:1]:
        for prompt in call["prompts"]:
            for s in split_sentences(prompt):
                assert s in sentences  # whole sentences only


def test_acceptance_length_changes_configuration_and_output(os_notes):
    """Acceptance 7: Short/Medium/Detailed change the budgets and the result length."""
    results, budgets = {}, {}
    for length in ("Short", "Medium", "Detailed"):
        model = LeadBiasedSummarizer()
        results[length] = summarize(os_notes, length, model)
        budgets[length] = model.calls[0]["max_new_tokens"]
    assert results["Short"].target_words < results["Medium"].target_words < results["Detailed"].target_words
    assert results["Short"].word_count < results["Medium"].word_count < results["Detailed"].word_count
    assert results["Short"].sections_total < results["Detailed"].sections_total
    assert len({LENGTH_PRESETS[k].section_ratio for k in LENGTH_PRESETS}) == 3


def test_target_word_count_bounds():
    for preset in LENGTH_PRESETS.values():
        assert target_word_count(118, preset) <= 0.6 * 118
        assert preset.min_words <= target_word_count(10_000, preset) <= preset.max_words


def test_section_budget_never_forces_padding():
    max_new, min_new = section_budget(section_words=20, section_tokens=30, ratio=0.5)
    assert min_new <= 30 * 0.8 and min_new < max_new


def test_invented_sentences_are_removed(ml_notes):
    invented = "Quantum blockchain spaceships revolutionised medieval agriculture forever."
    result = summarize(ml_notes, "Medium", LeadBiasedSummarizer(inject=invented))
    assert "Quantum" not in result.summary
    assert any(invented in rejected for d in result.section_details for rejected in d.rejected)


def test_repeated_sentences_are_removed(ml_notes):
    result = summarize(ml_notes, "Medium", LeadBiasedSummarizer(repeat=True))
    sentences = split_sentences(result.summary)
    for i, a in enumerate(sentences):
        for b in sentences[i + 1:]:
            assert text_similarity(a, b) < 0.6


def test_filter_rejects_fragments_and_unsupported():
    vocab = set(content_words("A process is a program in execution. Threads share memory."))
    kept, rejected = filter_generated_sentences(
        "A process is a program in execution . Bananas grow on volcanic islands. Threads share", vocab,
        DuplicateFilter(0.6))
    assert kept == ["A process is a program in execution."]
    assert len(rejected) == 2


def test_too_short_input_raises():
    with pytest.raises(SummarizationError, match="too short"):
        summarize("Just a few words here.", "Short", LeadBiasedSummarizer())


def test_model_failure_is_reported(ml_notes):
    with pytest.raises(SummarizationError, match="failed"):
        summarize(ml_notes, "Short", FailingModel())


def test_unknown_length_rejected(ml_notes):
    with pytest.raises(ValueError):
        summarize(ml_notes, "Tiny", LeadBiasedSummarizer())


def test_chunker_keeps_every_sentence_once(os_notes):
    sentences = split_sentences(os_notes)
    chunks = chunk_sentences(sentences, 200)
    flattened = [s for c in chunks for s in c.sentences]
    assert flattened == sentences
    forced = chunk_sentences(split_sentences("One two three four. Five six seven eight. Nine ten eleven twelve. "
                                             "Thirteen fourteen fifteen sixteen."), 500, min_chunks=2)
    assert len(forced) == 2
