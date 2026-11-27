"""StudyMate AI — Streamlit front end.

Run locally:   python3 -m streamlit run app.py

All study logic lives in ``modules/``; this file only handles the interface
and ``st.session_state`` so that results survive Streamlit's reruns.
"""

from __future__ import annotations

import hashlib
import html
import os
from pathlib import Path

import streamlit as st

from modules import classification, flashcards, mcqs, summarization
from modules.extraction import ExtractionError, extract_text
from modules.models import QG_MODEL, SUMMARY_MODEL, GenerationError, ModelLoadError, load_seq2seq
from modules.preprocessing import word_count

APP_DIR = Path(__file__).parent
ARTIFACTS_DIR = Path(os.getenv("STUDYMATE_ARTIFACTS_DIR", APP_DIR / "artifacts"))
SAMPLES_DIR = APP_DIR / "sample_data"
SAMPLES = {
    "Operating Systems (2,017 words)": "os_lecture_2000_words.txt",
    "Machine Learning (549 words)": "machine_learning_notes.txt",
    "Business Management (406 words)": "business_management_notes.txt",
    "Football & Sports (332 words)": "football_sports_notes.txt",
    "Cloud Computing — short note (118 words)": "short_note_118_words.txt",
}

st.set_page_config(page_title="StudyMate AI", page_icon="📚", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:wght@600&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextArea textarea, button { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 1.6rem; max-width: 1180px; }
.sm-hero { padding: 1.6rem 1.8rem; border-radius: 16px; color: #fff;
  background: linear-gradient(120deg, #18234a 0%, #2c3f86 55%, #4656c9 100%); margin-bottom: 1.2rem; }
.sm-hero h1 { font-family: 'Source Serif 4', Georgia, serif; font-size: 2.2rem; margin: 0 0 .25rem 0; color: #fff; }
.sm-hero p { margin: 0; color: #dfe5ff; font-size: 1.02rem; }
.sm-step { font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #4656c9; margin: 1.2rem 0 .2rem 0; }
.sm-card { border: 1px solid #dfe3f0; border-radius: 16px; padding: 1.6rem 1.8rem; background: #fff;
  box-shadow: 0 2px 10px rgba(24,35,74,.06); min-height: 190px; }
.sm-card .label { font-size: .75rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #6b7392; }
.sm-card .q { font-size: 1.28rem; font-weight: 600; color: #18234a; margin: .5rem 0 0 0; line-height: 1.45; }
.sm-card .a { margin-top: 1.1rem; padding-top: 1rem; border-top: 1px dashed #cdd3e6; }
.sm-card .a .text { font-size: 1.12rem; color: #0f5132; font-weight: 600; margin-top: .35rem; line-height: 1.45; }
.sm-card .src { font-size: .86rem; color: #5b6380; margin-top: .7rem; line-height: 1.45; }
.sm-card .hidden { color: #8a91ad; font-style: italic; margin-top: 1.1rem; }
.sm-summary { background: #fff; border: 1px solid #dfe3f0; border-left: 4px solid #4656c9; border-radius: 12px;
  padding: 1.1rem 1.3rem; line-height: 1.65; color: #1e2235; }
.sm-summary p { margin: 0 0 .8rem 0; }
.sm-topic { display: inline-block; padding: .45rem 1rem; border-radius: 999px; background: #e9ecff; color: #2c3f86;
  font-weight: 700; font-size: 1.15rem; }
.sm-muted { color: #6b7392; font-size: .9rem; }
.sm-option-correct { color: #0f5132; font-weight: 600; }
div[data-testid="stMetricValue"] { font-size: 1.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

DEFAULTS = {
    "notes": "",
    "upload_signature": None,
    "extraction_info": None,
    "original_text": "",
    "classification": None,
    "summary": None,
    "flashcards": None,
    "fc_index": 0,
    "fc_revealed": False,
    "mcqs": None,
    "quiz_id": 0,
    "quiz_index": 0,
    "quiz_answers": {},
    "quiz_finished": False,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, dict) else value


def notes_hash(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


def stale_notice(result_entry: dict | None) -> None:
    if result_entry and result_entry.get("notes_hash") != notes_hash(st.session_state.notes):
        st.info("Your notes have changed since this was generated. Generate again to update it.", icon="✏️")


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_classifier():
    return classification.load_classifier(ARTIFACTS_DIR)


@st.cache_resource(show_spinner=False, max_entries=1)
def get_transformer(model_name: str):
    """Only one Transformer is kept in memory at a time (free-tier RAM limit)."""
    return load_seq2seq(model_name)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📚 StudyMate AI")
    st.caption("Machine Learning + Transformer study assistant for university lecture notes.")
    st.markdown("---")
    st.markdown("**Tools**")
    st.markdown(
        "- [Upload notes](#upload-your-study-material)\n"
        "- [Topic classification](#choose-a-study-tool) — TF-IDF + Logistic Regression\n"
        "- [Smart summary](#choose-a-study-tool) — DistilBART\n"
        "- [Flashcards](#choose-a-study-tool) — fact extraction + FLAN-T5\n"
        "- [Practice MCQs](#choose-a-study-tool) — interactive quiz"
    )
    st.markdown("**How to use**")
    st.markdown(
        "1. Upload a PDF/TXT file or paste your notes.\n"
        "2. Check and edit the extracted text.\n"
        "3. Open a tool tab and press its Generate button.\n"
        "4. Download what you need."
    )
    st.markdown("---")
    st.warning(
        "AI-generated content can be incomplete or wrong. Always check summaries, flashcards and "
        "quiz answers against your original notes.",
        icon="⚠️",
    )
    st.caption(f"Models: {SUMMARY_MODEL} · {QG_MODEL}")

# ---------------------------------------------------------------------------
# Header + upload
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="sm-hero"><h1>Welcome to StudyMate AI</h1>'
    "<p>Turn your lecture notes into a topic label, a summary, flashcards and a practice quiz — "
    "all generated from your own material.</p></div>",
    unsafe_allow_html=True,
)

st.markdown('<div class="sm-step">Step 1</div>', unsafe_allow_html=True)
st.subheader("Upload your study material")

upload_col, sample_col = st.columns([3, 2], gap="large")
with upload_col:
    uploaded = st.file_uploader("PDF or TXT file", type=["pdf", "txt"], help="Text-based PDFs only (no OCR).")
with sample_col:
    sample_choice = st.selectbox("…or try sample notes", ["—"] + list(SAMPLES), index=0)
    if st.button("Load sample", disabled=sample_choice == "—", use_container_width=True):
        text = (SAMPLES_DIR / SAMPLES[sample_choice]).read_text(encoding="utf-8")
        st.session_state.notes = text
        st.session_state.original_text = text
        st.session_state.extraction_info = {"name": SAMPLES[sample_choice], "warnings": [], "pages": None}
        st.session_state.upload_signature = None

if uploaded is not None:
    signature = (uploaded.name, uploaded.size, getattr(uploaded, "file_id", ""))
    if signature != st.session_state.upload_signature:
        # A *new* file: extract once. Reruns with the same file keep the user's edits.
        st.session_state.upload_signature = signature
        try:
            with st.spinner(f"Extracting text from {uploaded.name}…"):
                result = extract_text(uploaded.name, uploaded.getvalue())
            st.session_state.notes = result.text
            st.session_state.original_text = result.text
            st.session_state.extraction_info = {"name": uploaded.name, "warnings": result.warnings,
                                                "pages": result.page_count if result.source_type == "pdf" else None,
                                                "pages_with_text": result.pages_with_text}
        except ExtractionError as exc:
            st.session_state.extraction_info = {"name": uploaded.name, "error": str(exc), "warnings": []}

info = st.session_state.extraction_info
if info:
    if info.get("error"):
        st.error(f"**{info['name']}:** {info['error']}")
    else:
        pages = f" from {info['pages_with_text']} of {info['pages']} page(s)" if info.get("pages") else ""
        st.success(f"Loaded **{info['name']}**{pages}. You can edit the text below before generating.")
        for warning in info.get("warnings", []):
            st.warning(warning)

st.text_area(
    "Lecture notes (editable)",
    key="notes",
    height=280,
    placeholder="Upload a document, load a sample, or paste your lecture notes here…",
)
notes = st.session_state.notes
n_words, n_chars = word_count(notes), len(notes)
count_col, restore_col = st.columns([4, 1])
count_col.caption(f"**{n_words:,}** words · **{n_chars:,}** characters")
def restore_original():
    st.session_state.notes = st.session_state.original_text


if st.session_state.original_text and notes != st.session_state.original_text:
    restore_col.button("Restore original", help="Undo your edits and restore the extracted text",
                       on_click=restore_original)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

st.markdown('<div class="sm-step">Step 2</div>', unsafe_allow_html=True)
st.subheader("Choose a study tool")
tab_topic, tab_summary, tab_cards, tab_quiz = st.tabs(
    ["🧠 Topic Classification", "✨ Smart Summary", "🗂️ Flashcards", "📝 Practice MCQs"]
)


def require_notes(min_words: int) -> bool:
    if word_count(st.session_state.notes) < min_words:
        st.warning(f"Please add at least {min_words} words of notes first (currently {word_count(st.session_state.notes)}).")
        return False
    return True


# ----- Topic classification -------------------------------------------------
with tab_topic:
    st.write("Predicts the broad topic of your notes with the TF-IDF + classical ML model trained in the notebook.")
    st.caption("The model was trained on AG News, so it can only answer **World, Sports, Business or Sci/Tech** — "
               "these are broad news topics, not university course names.")
    if st.button("Predict topic", type="primary", key="btn_topic") and require_notes(5):
        try:
            clf = get_classifier()
            prediction = clf.predict(st.session_state.notes)
            sections = clf.predict_sections(st.session_state.notes) if word_count(st.session_state.notes) > 400 else []
            st.session_state.classification = {"prediction": prediction, "sections": sections,
                                               "model": clf.model_name, "metadata": clf.metadata,
                                               "warnings": clf.load_warnings,
                                               "notes_hash": notes_hash(st.session_state.notes)}
        except classification.ModelArtifactsError as exc:
            st.error(f"**Model files problem:** {exc}")
        except ValueError as exc:
            st.warning(str(exc))

    entry = st.session_state.classification
    if entry:
        stale_notice(entry)
        pred = entry["prediction"]
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown("Predicted topic")
            st.markdown(f'<span class="sm-topic">{html.escape(pred.label)}</span>', unsafe_allow_html=True)
            if pred.confidence is not None:
                st.metric("Confidence", f"{pred.confidence:.0%}")
            st.caption(pred.confidence_note)
            for warning in entry["warnings"]:
                st.warning(warning)
        with right:
            if pred.probabilities:
                st.markdown("Probability for each topic")
                for label, prob in sorted(pred.probabilities.items(), key=lambda kv: -kv[1]):
                    st.progress(min(1.0, prob), text=f"{label}: {prob:.1%}")
        if entry["sections"]:
            with st.expander(f"Topic by section ({len(entry['sections'])} sections)"):
                st.dataframe(
                    [{"Section": s.section, "Words": s.words, "Topic": s.label,
                      "Confidence": f"{s.confidence:.0%}" if s.confidence is not None else "n/a"}
                     for s in entry["sections"]],
                    hide_index=True, use_container_width=True,
                )
        meta = entry["metadata"]
        if meta.get("results"):
            with st.expander("About the model"):
                st.write(f"Selected model: **{entry['model']}** (best macro-F1 of the five algorithms compared in "
                         f"the notebook). Trained on {meta.get('training_records_used', '?'):,} AG News articles, "
                         f"tested on {meta.get('test_records', '?'):,}.")
                st.dataframe(meta["results"], hide_index=True, use_container_width=True)

# ----- Smart summary --------------------------------------------------------
with tab_summary:
    st.write("Summarises **all** of your notes: the text is split into sections at sentence boundaries, every section "
             "is summarised, unsupported or repeated sentences are removed, and long results get a second pass.")
    length = st.segmented_control("Summary length", list(summarization.LENGTH_PRESETS), default="Medium",
                                  key="summary_length")
    length = length or "Medium"
    preset = summarization.get_preset(length)
    target = summarization.target_word_count(max(1, word_count(notes)), preset)
    st.caption(f"**{length}:** {preset.description} For your notes that is a target of about **{target} words**. "
               "Note: the model's maximum generation length is an upper limit, not a guaranteed length — the model "
               "can stop earlier, so the final word count is approximate and is reported below.")
    if st.button("Generate summary", type="primary", key="btn_summary") and \
            require_notes(summarization.MIN_INPUT_WORDS):
        try:
            with st.status("Preparing summary…", expanded=False) as status:
                status.update(label=f"Loading {SUMMARY_MODEL} (the first run downloads it, ~1 GB)…")
                model = get_transformer(SUMMARY_MODEL)
                status.update(label="Summarising every section of your notes… (CPU: about 5–20 s per section)")
                result = summarization.summarize(st.session_state.notes, length, model)
                status.update(label="Summary ready", state="complete")
            st.session_state.summary = {"result": result, "notes_hash": notes_hash(st.session_state.notes)}
        except (ModelLoadError, GenerationError, summarization.SummarizationError) as exc:
            st.error(f"**Summary could not be generated:** {exc}")
        except Exception as exc:  # never fail silently or show a half-finished result
            st.error(f"**Unexpected error while summarising:** {exc.__class__.__name__}: {exc}")

    entry = st.session_state.summary
    if entry:
        stale_notice(entry)
        res: summarization.SummaryResult = entry["result"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Summary words", res.word_count)
        m2.metric("Target", f"~{res.target_words}")
        m3.metric("Sections covered", f"{res.sections_covered}/{res.sections_total}")
        m4.metric("Length setting", res.length)
        paragraphs = "".join(f"<p>{html.escape(p)}</p>" for p in res.summary.split("\n\n"))
        st.markdown(f'<div class="sm-summary">{paragraphs}</div>', unsafe_allow_html=True)
        for warning in res.warnings:
            st.warning(warning)
        st.download_button("⬇️ Download summary (.txt)", res.summary, "studymate_summary.txt", "text/plain",
                           on_click="ignore")
        with st.expander("How this summary was produced"):
            st.write(f"Model: `{res.model_name}` · input {res.input_words} words · {res.sections_total} section(s) "
                     f"summarised · {res.passes} pass(es).")
            st.dataframe(
                [{"Section": d.section, "Source words": d.source_words, "min/max new tokens":
                  f"{d.min_new_tokens}/{d.max_new_tokens}", "Sentences kept": len(d.kept_sentences),
                  "Rejected": len(d.rejected), "Fallback used": "yes" if d.used_fallback else ""}
                 for d in res.section_details],
                hide_index=True, use_container_width=True,
            )


def load_question_model(use_ai: bool, status):
    """Return the FLAN-T5 wrapper, or None (with a visible warning) if it cannot be loaded."""
    if not use_ai:
        return None
    try:
        status.update(label=f"Loading {QG_MODEL} (the first run downloads it, ~1 GB)…")
        return get_transformer(QG_MODEL)
    except ModelLoadError as exc:
        st.warning(f"The AI question writer could not be loaded, so extraction-based questions are used. ({exc})")
        return None


# ----- Flashcards -----------------------------------------------------------
def fc_prev():
    st.session_state.fc_index = max(0, st.session_state.fc_index - 1)
    st.session_state.fc_revealed = False


def fc_next():
    total = len(st.session_state.flashcards["result"].cards)
    st.session_state.fc_index = min(total - 1, st.session_state.fc_index + 1)
    st.session_state.fc_revealed = False


def fc_toggle():
    st.session_state.fc_revealed = not st.session_state.fc_revealed


with tab_cards:
    st.write("Builds question-and-answer cards from definitions and key facts across **all** sections of your notes. "
             "Every answer is taken from your notes.")
    c1, c2 = st.columns([2, 3])
    n_cards = c1.segmented_control("Number of flashcards", flashcards.ALLOWED_COUNTS, default=10, key="n_cards")
    use_ai_cards = c2.toggle("Use FLAN-T5 to write fact questions", value=True, key="ai_cards",
                             help="Off = faster, uses definition questions and fill-in-the-blank only.")
    if st.button("Generate flashcards", type="primary", key="btn_cards") and require_notes(flashcards.MIN_NOTE_WORDS):
        try:
            with st.status("Preparing flashcards…", expanded=False) as status:
                model = load_question_model(use_ai_cards, status)
                status.update(label="Finding definitions and key facts, writing and checking questions…")
                result = flashcards.generate_flashcards(st.session_state.notes, int(n_cards or 10), model=model)
                status.update(label="Flashcards ready", state="complete")
            st.session_state.flashcards = {"result": result, "notes_hash": notes_hash(st.session_state.notes)}
            st.session_state.fc_index, st.session_state.fc_revealed = 0, False
        except (flashcards.FlashcardError, GenerationError) as exc:
            st.error(f"**Flashcards could not be generated:** {exc}")
        except Exception as exc:
            st.error(f"**Unexpected error while making flashcards:** {exc.__class__.__name__}: {exc}")

    entry = st.session_state.flashcards
    if entry:
        stale_notice(entry)
        res: flashcards.FlashcardResult = entry["result"]
        if res.cards:
            if res.complete:
                st.success(f"{res.produced} flashcards created from {res.sections} section(s) of your notes.")
            for warning in res.warnings:
                st.warning(warning)
            idx = min(st.session_state.fc_index, len(res.cards) - 1)
            card = res.cards[idx]
            st.progress((idx + 1) / len(res.cards), text=f"Card {idx + 1} of {len(res.cards)}")
            if st.session_state.fc_revealed:
                shown_answer = card.answer[:1].upper() + card.answer[1:]
                back = (f'<div class="a"><div class="label">Answer</div><div class="text">{html.escape(shown_answer)}</div>'
                        f'<div class="src">From your notes (section {card.section}): “{html.escape(card.source)}”</div></div>')
            else:
                back = '<div class="hidden">Think of your answer, then press “Show answer”.</div>'
            st.markdown(f'<div class="sm-card"><div class="label">Question {idx + 1}</div>'
                        f'<div class="q">{html.escape(card.question)}</div>{back}</div>', unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            b1.button("◀ Previous", on_click=fc_prev, disabled=idx == 0, use_container_width=True, key="fc_prev")
            b2.button("Hide answer" if st.session_state.fc_revealed else "Show answer", on_click=fc_toggle,
                      type="primary", use_container_width=True, key="fc_toggle")
            b3.button("Next ▶", on_click=fc_next, disabled=idx >= len(res.cards) - 1, use_container_width=True,
                      key="fc_next")
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Download flashcards (.csv)", flashcards.flashcards_to_csv(res.cards),
                               "studymate_flashcards.csv", "text/csv", on_click="ignore", use_container_width=True)
            d2.download_button("⬇️ Download flashcards (.txt)", flashcards.flashcards_to_txt(res.cards),
                               "studymate_flashcards.txt", "text/plain", on_click="ignore", use_container_width=True)
            with st.expander("See all cards in a table"):
                st.dataframe([{"#": i + 1, "Question": c.question, "Answer": c.answer, "Section": c.section,
                               "Method": c.method} for i, c in enumerate(res.cards)],
                             hide_index=True, use_container_width=True)
        else:
            st.error("No valid flashcards could be made from these notes. " + " ".join(res.warnings))


# ----- Practice MCQs --------------------------------------------------------
def quiz_submit(choice_key: str, index: int, options: list):
    choice = st.session_state.get(choice_key)
    if choice is not None:
        st.session_state.quiz_answers[index] = options.index(choice)


def quiz_next(total: int):
    if st.session_state.quiz_index >= total - 1:
        st.session_state.quiz_finished = True
    else:
        st.session_state.quiz_index += 1


def quiz_restart():
    st.session_state.quiz_id += 1  # new widget keys -> cleared radio buttons
    st.session_state.quiz_index = 0
    st.session_state.quiz_answers = {}
    st.session_state.quiz_finished = False


with tab_quiz:
    st.write("Generates factual multiple-choice questions from across your notes. Each has four options (A–D), "
             "exactly one correct answer and an explanation quoted from your notes.")
    c1, c2 = st.columns([2, 3])
    n_mcq = c1.segmented_control("Number of questions", mcqs.ALLOWED_COUNTS, default=10, key="n_mcq")
    use_ai_mcq = c2.toggle("Use FLAN-T5 to write fact questions", value=True, key="ai_mcq",
                           help="Off = faster, uses definition questions and fill-in-the-blank only.")
    if st.button("Generate quiz", type="primary", key="btn_mcq") and require_notes(mcqs.MIN_NOTE_WORDS):
        try:
            with st.status("Preparing quiz…", expanded=False) as status:
                model = load_question_model(use_ai_mcq, status)
                status.update(label="Selecting facts, writing questions and choosing distractors…")
                result = mcqs.generate_mcqs(st.session_state.notes, int(n_mcq or 10), model=model)
                status.update(label="Quiz ready", state="complete")
            st.session_state.mcqs = {"result": result, "notes_hash": notes_hash(st.session_state.notes)}
            quiz_restart()
        except (mcqs.MCQError, GenerationError) as exc:
            st.error(f"**The quiz could not be generated:** {exc}")
        except Exception as exc:
            st.error(f"**Unexpected error while making the quiz:** {exc.__class__.__name__}: {exc}")

    entry = st.session_state.mcqs
    if entry:
        stale_notice(entry)
        res: mcqs.MCQResult = entry["result"]
        questions = res.questions
        if not questions:
            st.error("No valid questions could be made from these notes. " + " ".join(res.warnings))
        else:
            for warning in res.warnings:
                st.warning(warning)
            answers = st.session_state.quiz_answers
            score = sum(1 for i, a in answers.items() if a == questions[i].correct_index)
            if st.session_state.quiz_finished:
                pct = score / len(questions)
                st.subheader("Quiz complete 🎉")
                s1, s2 = st.columns(2)
                s1.metric("Score", f"{score} / {len(questions)}")
                s2.metric("Percentage", f"{pct:.0%}")
                st.progress(pct)
                with st.expander("Review your answers", expanded=True):
                    for i, q in enumerate(questions):
                        chosen = answers.get(i)
                        mark = "✅" if chosen == q.correct_index else "❌"
                        your = f"{mcqs.LETTERS[chosen]}) {q.options[chosen]}" if chosen is not None else "—"
                        st.markdown(f"{mark} **{i + 1}. {q.question}**  \nYour answer: {your} · Correct: "
                                    f"**{q.correct_letter}) {q.correct_option}**")
                st.button("🔄 Restart quiz", on_click=quiz_restart, type="primary", key="quiz_restart_end")
            else:
                i = min(st.session_state.quiz_index, len(questions) - 1)
                q = questions[i]
                st.progress(i / len(questions), text=f"Question {i + 1} of {len(questions)} · Score so far: {score}")
                st.markdown(f"#### {i + 1}. {q.question}")
                labelled = [f"{letter}) {option}" for letter, option in zip(mcqs.LETTERS, q.options)]
                choice_key = f"quiz_{st.session_state.quiz_id}_{i}"
                submitted = i in answers
                st.radio("Choose one answer", labelled, index=None, key=choice_key, disabled=submitted,
                         label_visibility="collapsed")
                if not submitted:
                    st.button("Submit answer", type="primary", key=f"submit_{choice_key}",
                              disabled=st.session_state.get(choice_key) is None,
                              on_click=quiz_submit, args=(choice_key, i, labelled))
                else:
                    if answers[i] == q.correct_index:
                        st.success(f"Correct! **{q.correct_letter}) {q.correct_option}**")
                    else:
                        st.error(f"Not quite. The correct answer is **{q.correct_letter}) {q.correct_option}**.")
                    st.caption(q.explanation)
                    st.button("Finish quiz" if i == len(questions) - 1 else "Next question ▶", type="primary",
                              on_click=quiz_next, args=(len(questions),), key=f"next_{choice_key}")
                st.button("🔄 Restart quiz", on_click=quiz_restart, key="quiz_restart")
            d1, d2, d3 = st.columns(3)
            d1.download_button("⬇️ Questions (.txt)", mcqs.mcqs_to_txt(questions), "studymate_mcqs.txt",
                               "text/plain", on_click="ignore", use_container_width=True)
            d2.download_button("⬇️ Answer key (.txt)", mcqs.answer_key_txt(questions), "studymate_answer_key.txt",
                               "text/plain", on_click="ignore", use_container_width=True)
            d3.download_button("⬇️ All (.csv)", mcqs.mcqs_to_csv(questions), "studymate_mcqs.csv", "text/csv",
                               on_click="ignore", use_container_width=True)

st.markdown("---")
st.caption("StudyMate AI · TF-IDF + Logistic Regression (AG News) · DistilBART summarisation · FLAN-T5 question "
           "writing · Built with Streamlit")
