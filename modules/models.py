"""Loading and running the pretrained Transformer models (CPU only).

Two Hugging Face sequence-to-sequence models are used:

* Summariser:   ``sshleifer/distilbart-cnn-12-6`` (306M parameters, 1,024-token input)
                – the same model the notebook's ROUGE evaluation used.
* Generator:    ``google/flan-t5-base`` (248M parameters, 512-token input)
                – used for question writing and answer checking.

Both can be overridden with environment variables
``STUDYMATE_SUMMARY_MODEL`` / ``STUDYMATE_QG_MODEL`` (for example
``google/flan-t5-small`` on a very memory-constrained host).

``generate`` is a thin, testable wrapper around ``model.generate`` that
refuses inputs longer than the model's limit instead of truncating them
silently – callers are responsible for chunking.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

SUMMARY_MODEL = os.getenv("STUDYMATE_SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6")
QG_MODEL = os.getenv("STUDYMATE_QG_MODEL", "google/flan-t5-base")

# Input limits. BART has 1,024 learned positions. T5 uses relative positions
# but was trained on 512-token inputs; quality drops beyond that.
DEFAULT_INPUT_LIMITS = {"bart": 1024, "t5": 512}


class ModelLoadError(Exception):
    """A model could not be downloaded or loaded. Message is user-facing."""


class GenerationError(Exception):
    """Model inference failed. Message is user-facing."""


class InputTooLongError(GenerationError):
    """An input exceeded the model's token limit (a chunking bug, never truncated silently)."""


@dataclass
class Seq2SeqModel:
    """Tokenizer + model pair with a safe ``generate`` method."""

    name: str
    tokenizer: object
    model: object
    max_input_tokens: int

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True)["input_ids"])

    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: int,
        min_new_tokens: int = 0,
        num_beams: int = 4,
        batch_size: int = 4,
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 1.0,
    ) -> List[str]:
        import torch

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        min_new_tokens = max(0, min(min_new_tokens, max_new_tokens - 1))
        outputs: List[str] = []
        for start in range(0, len(prompts), batch_size):
            batch = list(prompts[start:start + batch_size])
            for prompt in batch:
                n_tokens = self.count_tokens(prompt)
                if n_tokens > self.max_input_tokens:
                    raise InputTooLongError(
                        f"Internal error: an input of {n_tokens} tokens exceeds the {self.max_input_tokens}-token "
                        f"limit of {self.name}. Inputs must be chunked before generation."
                    )
            encoded = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=False)
            try:
                with torch.inference_mode():
                    generated = self.model.generate(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        num_beams=num_beams,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        length_penalty=length_penalty,
                        early_stopping=num_beams > 1,
                        do_sample=False,
                    )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() or "alloc" in str(exc).lower():
                    raise GenerationError(
                        "The server ran out of memory while generating. Try shorter notes or fewer items."
                    ) from exc
                raise GenerationError(f"The model failed while generating text: {exc}") from exc
            outputs.extend(
                text.strip() for text in self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        return outputs


def _input_limit(model) -> int:
    config = model.config
    model_type = getattr(config, "model_type", "")
    if model_type in DEFAULT_INPUT_LIMITS:
        return DEFAULT_INPUT_LIMITS[model_type]
    return int(getattr(config, "max_position_embeddings", 512) or 512)


def load_seq2seq(model_name: str, max_input_tokens: Optional[int] = None) -> Seq2SeqModel:
    """Download (first time only) and load a seq2seq model for CPU inference."""
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ModelLoadError(
            "PyTorch and Transformers are not installed. Run: python3 -m pip install -r requirements.txt"
        ) from exc

    try:
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True)
        model.eval()
    except MemoryError as exc:
        raise ModelLoadError(
            f"Not enough memory to load {model_name}. On Streamlit Community Cloud, set "
            "STUDYMATE_QG_MODEL=google/flan-t5-small or STUDYMATE_SUMMARY_MODEL=sshleifer/distilbart-cnn-6-6."
        ) from exc
    except OSError as exc:
        raise ModelLoadError(
            f"Could not download or open the model '{model_name}' from Hugging Face. Check the internet "
            f"connection and the model name, then try again. ({exc.__class__.__name__})"
        ) from exc
    except Exception as exc:
        raise ModelLoadError(f"Loading '{model_name}' failed: {exc}") from exc

    return Seq2SeqModel(model_name, tokenizer, model, max_input_tokens or _input_limit(model))
