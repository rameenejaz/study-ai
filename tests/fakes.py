"""Stand-in models for unit tests.

They follow the same interface as ``modules.models.Seq2SeqModel`` (``name``,
``max_input_tokens``, ``count_tokens``, ``generate``) but run instantly and
deterministically, which lets the tests check the *pipeline logic* –
chunking, coverage, length budgets, parsing, validation and fallbacks.
They say nothing about the writing quality of the real DistilBART/FLAN-T5.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence

from modules.preprocessing import approximate_token_count, split_sentences


class FakeModelBase:
    def __init__(self, name: str, max_input_tokens: int):
        self.name = name
        self.max_input_tokens = max_input_tokens
        self.calls: List[dict] = []

    def count_tokens(self, text: str) -> int:
        return approximate_token_count(text)

    def _record(self, prompts, **kwargs):
        for prompt in prompts:
            assert self.count_tokens(prompt) <= self.max_input_tokens, "input exceeds the model limit"
        self.calls.append({"prompts": list(prompts), **kwargs})

    @property
    def all_prompts(self) -> List[str]:
        return [p for call in self.calls for p in call["prompts"]]


class LeadBiasedSummarizer(FakeModelBase):
    """Mimics DistilBART-CNN's worst habit: it copies the opening sentences.

    It copies sentences from the start of each input until the word budget
    implied by ``max_new_tokens`` is reached (but at least ``min_new_tokens``).
    """

    def __init__(self, max_input_tokens: int = 1024, inject: Optional[str] = None, repeat: bool = False):
        super().__init__("fake-distilbart-cnn", max_input_tokens)
        self.inject = inject      # an invented sentence appended to every output
        self.repeat = repeat      # repeat the first sentence (tests de-duplication)

    def generate(self, prompts: Sequence[str], max_new_tokens: int, min_new_tokens: int = 0, **kwargs) -> List[str]:
        self._record(prompts, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens, **kwargs)
        outputs = []
        max_words = max(1, int(max_new_tokens / 1.4))
        min_words = int(min_new_tokens / 1.4)
        for prompt in prompts:
            words_out: List[str] = []
            for sentence in split_sentences(prompt):
                n = len(sentence.split())
                if words_out and len(words_out) + n > max_words and len(words_out) >= min_words:
                    break
                words_out.extend(sentence.split())
                if len(words_out) >= max_words:
                    words_out = words_out[:max_words]
                    break
            text = " ".join(words_out)
            if self.repeat:
                first = split_sentences(prompt)[0]
                text = f"{first} {first} {text}"
            if self.inject:
                text = f"{text} {self.inject}"
            outputs.append(text)
        return outputs


class FailingModel(FakeModelBase):
    def __init__(self):
        super().__init__("broken-model", 512)

    def generate(self, prompts, max_new_tokens, min_new_tokens=0, **kwargs):
        raise RuntimeError("simulated inference crash")


_QG_RE = re.compile(r'The answer to the question must be "(?P<answer>.+?)"\.\n\nText: (?P<sentence>.+?)\n\nQuestion:', re.S)
_RETRY_RE = re.compile(r'^(?P<sentence>.+?)\n\nAsk one short question.*?because "(?P<answer>.+?)" is the answer\.$', re.S)
_VERIFY_RE = re.compile(r"Read this and answer the question\.\n\n(?P<sentence>.+?)\n\n(?P<question>.+)$", re.S)


class FakeQuestionModel(FakeModelBase):
    """Behaves like a well-behaved FLAN-T5 for question writing.

    * QG prompt -> "Which ... ?" question built from the sentence without the answer.
    * Verification prompt -> returns the answer it was asked to target.
    ``behaviour`` can make it misbehave to test parsing/validation/fallbacks.
    """

    def __init__(self, behaviour: str = "good", raw_output: Optional[Callable[[str, str], str]] = None):
        super().__init__("fake-flan-t5", 512)
        self.behaviour = behaviour
        self.raw_output = raw_output
        self.targets = {}

    def _question(self, sentence: str, answer: str) -> str:
        pattern = re.compile(r"(?<![\w])" + re.escape(answer) + r"(?![\w])", re.IGNORECASE)
        rest = pattern.sub("which concept", sentence, count=1).rstrip(".")
        return f"Question: According to the notes, {rest}?"

    def generate(self, prompts: Sequence[str], max_new_tokens: int, min_new_tokens: int = 0, **kwargs) -> List[str]:
        self._record(prompts, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens, **kwargs)
        outputs = []
        for prompt in prompts:
            verify = _VERIFY_RE.match(prompt)
            qg = _QG_RE.search(prompt) or _RETRY_RE.match(prompt)
            if verify and not qg:
                sentence = verify.group("sentence")
                outputs.append("something unrelated" if self.behaviour == "wrong_answers"
                               else self.targets.get(sentence, ""))
                continue
            assert qg, f"unexpected prompt: {prompt[:80]}"
            sentence, answer = qg.group("sentence").strip(), qg.group("answer")
            self.targets[sentence] = answer
            if self.raw_output is not None:
                outputs.append(self.raw_output(sentence, answer))
            elif self.behaviour == "good":
                outputs.append(self._question(sentence, answer))
            elif self.behaviour == "json":
                outputs.append('{"question": "%s"}' % self._question(sentence, answer).replace("Question: ", ""))
            elif self.behaviour == "garbage":
                outputs.append(sentence + " " + sentence)  # a paragraph, not a question
            elif self.behaviour == "empty":
                outputs.append("")
            elif self.behaviour == "reveals_answer":
                outputs.append(f"What is {answer}?")
            elif self.behaviour == "wrong_answers":
                outputs.append(self._question(sentence, answer))
            else:
                raise ValueError(self.behaviour)
        return outputs
