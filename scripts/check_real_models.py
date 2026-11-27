"""Check the REAL DistilBART and FLAN-T5 models on the sample notes.

The automated tests use stand-in models (the test machine could not download
from Hugging Face). Run this once on your own computer to see what the real
models produce and confirm that everything passes validation:

    python3 scripts/check_real_models.py            # all checks (downloads ~2 GB the first time)
    python3 scripts/check_real_models.py --quick    # short note only

It prints the summaries, flashcards and MCQs, the time taken, and a PASS/FAIL
line for each acceptance check.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.facts import analyse_notes  # noqa: E402
from modules.flashcards import generate_flashcards, validate_flashcard  # noqa: E402
from modules.mcqs import generate_mcqs, validate_mcq  # noqa: E402
from modules.models import QG_MODEL, SUMMARY_MODEL, load_seq2seq  # noqa: E402
from modules.preprocessing import split_sentences, text_similarity  # noqa: E402
from modules.summarization import summarize  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    short = (ROOT / "sample_data/short_note_118_words.txt").read_text()
    long = (ROOT / "sample_data/os_lecture_2000_words.txt").read_text()
    ml = (ROOT / "sample_data/machine_learning_notes.txt").read_text()

    print(f"Loading {SUMMARY_MODEL} …")
    t = time.time()
    summarizer = load_seq2seq(SUMMARY_MODEL)
    print(f"  loaded in {time.time() - t:.0f}s")

    texts = [("118-word note", short)] + ([] if args.quick else [("2,000-word note", long)])
    for label, text in texts:
        lengths = {}
        for length in ("Short", "Medium", "Detailed"):
            t = time.time()
            res = summarize(text, length, summarizer)
            lengths[length] = res.word_count
            print(f"\n=== {label} — {length} ({res.word_count} words, target ~{res.target_words}, "
                  f"{res.sections_covered}/{res.sections_total} sections, {time.time() - t:.0f}s) ===")
            print(res.summary)
            for w in res.warnings:
                print("  warning:", w)
        if label.startswith("118"):
            later = split_sentences(" ".join(short.split("\n\n")[1:]))
            first = split_sentences(short.split("\n\n")[0])
            res = summarize(short, "Medium", summarizer)

            def from_later(sentence):  # best-matching source sentence is outside the first paragraph
                return max(later, key=lambda s: text_similarity(sentence, s)) and \
                    max(text_similarity(sentence, s) for s in later) > max(text_similarity(sentence, s) for s in first)

            check("118-word summary is not just the first paragraph",
                  any(from_later(s) for s in split_sentences(res.summary)))
        check(f"{label}: Short < Medium < Detailed", lengths["Short"] < lengths["Medium"] < lengths["Detailed"],
              str(lengths))
    del summarizer

    print(f"\nLoading {QG_MODEL} …")
    generator = load_seq2seq(QG_MODEL)
    notes = ml if args.quick else long
    analysis = analyse_notes(notes)
    t = time.time()
    cards = generate_flashcards(notes, 10, model=generator)
    print(f"\n=== 10 flashcards ({time.time() - t:.0f}s) ===")
    for i, c in enumerate(cards.cards, 1):
        print(f"{i:>2}. [{c.method}] Q: {c.question}\n    A: {c.answer}")
    check("10 valid flashcards", cards.produced == 10 and all(validate_flashcard(c, analysis) is None
                                                               for c in cards.cards), str(cards.warnings))
    t = time.time()
    quiz = generate_mcqs(notes, 10, model=generator)
    print(f"\n=== 10 MCQs ({time.time() - t:.0f}s) ===")
    for i, q in enumerate(quiz.questions, 1):
        print(f"{i:>2}. [{q.method}] {q.question}")
        for letter, option in zip("ABCD", q.options):
            print(f"      {letter}) {option}{'   <- correct' if letter == q.correct_letter else ''}")
    check("10 valid MCQs", quiz.produced == 10 and all(validate_mcq(q, analysis) is None for q in quiz.questions),
          str(quiz.warnings))
    model_written = sum(1 for c in cards.cards if c.method.startswith("model"))
    print(f"\nFLAN-T5 wrote {model_written} flashcard question(s); the rest used definitions or fill-in-the-blank.")
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed.")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
