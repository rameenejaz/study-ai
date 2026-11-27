"""Runs the real Hugging Face code path with *tiny, randomly initialised* models.

No download is needed: a word-level tokenizer and 2-layer T5/BART models are
built in memory. Their output is random text, which is exactly what we want
here – it proves that ``Seq2SeqModel.generate`` works with the installed
transformers/torch versions and that random (invalid) model output is
rejected by validation and replaced by fallbacks, never shown as a result.
Quality of the real DistilBART / FLAN-T5 is checked separately with
``scripts/check_real_models.py`` on a machine that can download them.
"""

import re

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from modules.flashcards import generate_flashcards, validate_flashcard  # noqa: E402
from modules.facts import analyse_notes  # noqa: E402
from modules.mcqs import generate_mcqs, validate_mcq  # noqa: E402
from modules.models import InputTooLongError, ModelLoadError, Seq2SeqModel, _input_limit, load_seq2seq  # noqa: E402
from modules.summarization import summarize  # noqa: E402


def _tokenizer(texts):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, "<s>": 3}
    for text in texts:
        for token in re.findall(r"\w+|[^\w\s]", text.lower()):
            vocab.setdefault(token, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="</s>",
                                   unk_token="<unk>", bos_token="<s>")


@pytest.fixture(scope="module")
def tokenizer(all_domain_notes, short_notes):
    prompts = ("write a question about the text below . the answer to the question must be \" text : question : "
               "ask one short question that starts with what , which , who , when , where or how . do not include "
               "the words because is answer read this and answer summarize")
    return _tokenizer(list(all_domain_notes.values()) + [short_notes, prompts])


@pytest.fixture(scope="module")
def tiny_t5(tokenizer):
    from transformers import T5Config, T5ForConditionalGeneration

    torch.manual_seed(0)
    config = T5Config(vocab_size=len(tokenizer), d_model=32, d_ff=64, d_kv=16, num_layers=2, num_heads=2,
                      decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    return Seq2SeqModel("tiny-random-t5", tokenizer, T5ForConditionalGeneration(config).eval(), 512)


@pytest.fixture(scope="module")
def tiny_bart(tokenizer):
    from transformers import BartConfig, BartForConditionalGeneration

    torch.manual_seed(0)
    config = BartConfig(vocab_size=len(tokenizer), d_model=32, encoder_layers=1, decoder_layers=1,
                        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=64,
                        decoder_ffn_dim=64, max_position_embeddings=1024, pad_token_id=0, eos_token_id=1,
                        bos_token_id=3, decoder_start_token_id=1, forced_bos_token_id=None, forced_eos_token_id=1)
    model = BartForConditionalGeneration(config).eval()
    assert _input_limit(model) == 1024
    return Seq2SeqModel("tiny-random-bart", tokenizer, model, 1024)


def test_generate_respects_budget(tiny_t5):
    outputs = tiny_t5.generate(["a process is a program in execution .", "threads share memory ."],
                               max_new_tokens=6, min_new_tokens=2, num_beams=2)
    assert len(outputs) == 2 and all(isinstance(o, str) for o in outputs)
    for text in outputs:
        assert len(tiny_t5.tokenizer(text, add_special_tokens=False)["input_ids"]) <= 6


def test_overlong_input_is_refused_not_truncated(tiny_t5):
    with pytest.raises(InputTooLongError):
        tiny_t5.generate(["word " * 600], max_new_tokens=5)


def test_summarize_with_real_generate(tiny_bart, os_notes):
    result = summarize(os_notes, "Short", tiny_bart)
    assert result.summary.strip()
    assert result.sections_summarised == result.sections_total
    # Random text is rejected, so the representative original sentences are used and this is reported.
    assert any("rejected" in w for w in result.warnings)


def test_flashcards_and_mcqs_with_real_generate(tiny_t5, ml_notes):
    analysis = analyse_notes(ml_notes)
    cards = generate_flashcards(ml_notes, 10, model=tiny_t5)
    assert cards.produced == 10
    assert all(validate_flashcard(c, analysis) is None for c in cards.cards)
    assert all(c.method in ("definition", "cloze") for c in cards.cards)  # random text never accepted
    quiz = generate_mcqs(ml_notes, 10, model=tiny_t5)
    assert quiz.produced == 10 and all(validate_mcq(q, analysis) is None for q in quiz.questions)


def test_unavailable_model_gives_clear_error(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(ModelLoadError, match="Could not download"):
        load_seq2seq("this-org/does-not-exist-model")
