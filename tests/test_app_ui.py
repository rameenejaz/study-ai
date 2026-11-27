"""End-to-end UI tests using Streamlit's headless AppTest runner.

The Transformer loader is replaced by the fast stand-in models from
``tests/fakes.py``; everything else (session state, widgets, callbacks,
classifier artifacts) is the real application.
"""

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import modules.models
from fakes import FakeQuestionModel, LeadBiasedSummarizer


@pytest.fixture(autouse=True)
def fake_transformers(monkeypatch):
    def fake_loader(name, max_input_tokens=None):
        return LeadBiasedSummarizer() if name == modules.models.SUMMARY_MODEL else FakeQuestionModel("good")

    monkeypatch.setattr(modules.models, "load_seq2seq", fake_loader)
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def start(root, sample="Operating Systems (2,017 words)"):
    at = AppTest.from_file(str(root / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    if sample:
        at.selectbox[0].set_value(sample)
        at.run()
        next(b for b in at.button if b.label == "Load sample").click()
        at.run()
        assert not at.exception
    return at


def markdown_text(at):
    return "\n".join(m.value for m in at.markdown)


def test_app_renders_layout(root):
    at = start(root, sample=None)
    assert len(at.tabs) == 4
    assert [t.label for t in at.tabs] == ["🧠 Topic Classification", "✨ Smart Summary", "🗂️ Flashcards",
                                          "📝 Practice MCQs"]
    assert any("AI-generated content" in w.value for w in at.sidebar.warning)
    assert "Welcome to StudyMate AI" in markdown_text(at)


def test_sample_loads_and_counts_are_shown(root):
    at = start(root)
    assert at.text_area(key="notes").value.startswith("Operating Systems")
    assert any("2,017" in c.value and "words" in c.value for c in at.caption)


def test_edited_notes_survive_reruns(root):
    at = start(root)
    at.text_area(key="notes").set_value("My own edited notes. A process is a program in execution.")
    at.run()
    at.run()  # unrelated rerun
    assert at.text_area(key="notes").value.startswith("My own edited notes")
    assert any(b.label == "Restore original" for b in at.button)
    next(b for b in at.button if b.label == "Restore original").click()
    at.run()
    assert at.text_area(key="notes").value.startswith("Operating Systems")


def test_topic_classification_tab(root):
    at = start(root, sample="Football & Sports (332 words)")
    at.button(key="btn_topic").click()
    at.run()
    assert not at.exception
    assert "Sports" in markdown_text(at)
    assert at.session_state["classification"]["prediction"].label == "Sports"


def test_missing_artifacts_show_error(root, tmp_path, monkeypatch):
    monkeypatch.setenv("STUDYMATE_ARTIFACTS_DIR", str(tmp_path))
    at = start(root, sample="Football & Sports (332 words)")
    at.button(key="btn_topic").click()
    at.run()
    assert any("best_classifier.joblib" in e.value for e in at.error)


def test_summary_tab_and_length_setting(root):
    """Acceptance 7 (UI level): the selected length changes the summary configuration."""
    at = start(root)
    results = {}
    for length in ("Short", "Detailed"):
        at.session_state["summary_length"] = length
        at.run()
        at.button(key="btn_summary").click()
        at.run()
        assert not at.exception and not at.error
        results[length] = at.session_state["summary"]["result"]
    assert results["Short"].length == "Short" and results["Detailed"].length == "Detailed"
    assert results["Short"].target_words < results["Detailed"].target_words
    assert results["Short"].word_count < results["Detailed"].word_count
    assert any(b.label.startswith("⬇️ Download summary") for b in at.get("download_button"))


def test_flashcards_are_interactive_cards(root):
    """Acceptance 5 + 8: real cards with navigation; reruns keep the place."""
    at = start(root)
    at.session_state["n_cards"] = 10
    at.run()
    at.button(key="btn_cards").click()
    at.run()
    assert not at.exception
    result = at.session_state["flashcards"]["result"]
    assert result.produced == 10
    first = result.cards[0]
    page = markdown_text(at)
    assert "Question 1" in page and first.question.replace("'", "&#x27;") in page
    assert "Think of your answer" in page  # answer hidden at first

    at.button(key="fc_toggle").click()
    at.run()
    assert "Answer" in markdown_text(at) and at.session_state["fc_revealed"] is True

    at.button(key="fc_next").click()
    at.run()
    assert at.session_state["fc_index"] == 1 and at.session_state["fc_revealed"] is False
    assert "Question 2" in markdown_text(at)

    # Unrelated interactions must not lose the place or the cards.
    at.button(key="btn_topic").click()
    at.run()
    at.run()
    assert at.session_state["fc_index"] == 1
    assert at.session_state["flashcards"]["result"].produced == 10

    at.button(key="fc_prev").click()
    at.run()
    assert at.session_state["fc_index"] == 0
    assert at.button(key="fc_prev").disabled


def test_mcq_quiz_flow(root):
    """Acceptance 6: select, submit, feedback, next, final score, restart."""
    at = start(root, sample="Machine Learning (549 words)")
    at.session_state["n_mcq"] = 5
    at.run()
    at.button(key="btn_mcq").click()
    at.run()
    assert not at.exception
    questions = at.session_state["mcqs"]["result"].questions
    assert len(questions) == 5

    for i, q in enumerate(questions):
        quiz_id = at.session_state["quiz_id"]
        radio = at.radio(key=f"quiz_{quiz_id}_{i}")
        assert len(radio.options) == 4
        # Answer the first two correctly, the rest with a wrong option.
        choice = q.correct_index if i < 2 else (q.correct_index + 1) % 4
        radio.set_value(radio.options[choice])
        at.run()
        at.button(key=f"submit_quiz_{quiz_id}_{i}").click()
        at.run()
        assert not at.exception
        if i < 2:
            assert any("Correct!" in s.value for s in at.success)
        else:
            assert any("correct answer is" in e.value for e in at.error)
        at.button(key=f"next_quiz_{quiz_id}_{i}").click()
        at.run()

    assert at.session_state["quiz_finished"] is True
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Score"] == "2 / 5" and metrics["Percentage"] == "40%"

    at.button(key="quiz_restart_end").click()
    at.run()
    assert at.session_state["quiz_finished"] is False and at.session_state["quiz_index"] == 0
    assert at.session_state["quiz_answers"] == {}
    assert at.radio(key=f"quiz_{at.session_state['quiz_id']}_0").value is None
