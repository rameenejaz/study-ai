import json
import shutil

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from modules.classification import ModelArtifactsError, load_classifier
from modules.preprocessing import clean_text


@pytest.fixture(scope="module")
def classifier(root):
    return load_classifier(root / "artifacts")


def test_real_artifacts_load_and_are_the_notebook_model(classifier):
    assert classifier.model_name == "Logistic Regression"
    assert classifier.labels == ["World", "Sports", "Business", "Sci/Tech"]
    assert classifier.supports_probability
    assert classifier.load_warnings == []  # requirements pin the training version of scikit-learn


@pytest.mark.parametrize("fixture, expected", [
    ("os_notes", "Sci/Tech"),
    ("ml_notes", "Sci/Tech"),
    ("business_notes", "Business"),
    ("football_notes", "Sports"),
])
def test_domain_notes_are_classified(request, classifier, fixture, expected):
    prediction = classifier.predict(request.getfixturevalue(fixture))
    assert prediction.label == expected
    assert isinstance(prediction.label, str)
    assert 0.0 <= prediction.confidence <= 1.0
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-6
    assert prediction.confidence == max(prediction.probabilities.values())


def test_world_news_sentence(classifier):
    text = "The prime minister met foreign leaders at the United Nations to discuss the peace agreement and refugees."
    assert classifier.predict(text).label == "World"


def test_section_predictions_cover_long_notes(classifier, os_notes):
    sections = classifier.predict_sections(os_notes)
    assert len(sections) >= 6
    assert all(s.label in classifier.labels for s in sections)


def test_clean_text_matches_training_preprocessing():
    assert clean_text("Visit https://x.com NOW!! C++ & Python") == "visit now c python"


def test_empty_text_raises(classifier):
    with pytest.raises(ValueError):
        classifier.predict("!!! ???")


def test_missing_artifacts_give_clear_error(tmp_path):
    with pytest.raises(ModelArtifactsError, match="best_classifier.joblib"):
        load_classifier(tmp_path)


def test_corrupted_model_file(tmp_path):
    (tmp_path / "best_classifier.joblib").write_bytes(b"not a joblib file")
    with pytest.raises(ModelArtifactsError, match="could not be loaded"):
        load_classifier(tmp_path)


def _tiny_training_data():
    texts = ["football match goal striker league"] * 5 + ["stock market shares profit company"] * 5
    return [clean_text(t) for t in texts], [1] * 5 + [2] * 5


def test_string_labels_are_used_directly(tmp_path):
    texts, ids = _tiny_training_data()
    names = ["Sports" if i == 1 else "Business" for i in ids]
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", LogisticRegression())]).fit(texts, names)
    joblib.dump(pipe, tmp_path / "best_classifier.joblib")
    # Deliberately misleading metadata: must NOT be used for string predictions.
    (tmp_path / "model_metadata.json").write_text(json.dumps({"labels": ["A", "B", "C", "D"]}))
    clf = load_classifier(tmp_path)
    assert clf.predict("the striker scored a goal").label == "Sports"
    assert clf.predict("company shares and profit").label == "Business"


def test_integer_labels_outside_metadata_are_rejected(tmp_path):
    texts, ids = _tiny_training_data()
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", LogisticRegression())]).fit(texts, ids)
    joblib.dump(pipe, tmp_path / "best_classifier.joblib")
    (tmp_path / "model_metadata.json").write_text(json.dumps({"labels": ["World", "Sports"]}))  # no id 2
    with pytest.raises(ModelArtifactsError, match="inconsistent"):
        load_classifier(tmp_path)


def test_classifier_without_probabilities_reports_no_confidence(tmp_path):
    texts, ids = _tiny_training_data()
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", LinearSVC())]).fit(texts, ids)
    joblib.dump(pipe, tmp_path / "best_classifier.joblib")
    clf = load_classifier(tmp_path)  # default AG News labels: 1 -> Sports, 2 -> Business
    prediction = clf.predict("striker goal")
    assert prediction.label == "Sports"
    assert prediction.confidence is None and prediction.probabilities == {}
    assert "does not produce calibrated probabilities" in prediction.confidence_note


def test_bare_classifier_uses_saved_vectorizer(tmp_path):
    texts, ids = _tiny_training_data()
    vec = TfidfVectorizer().fit(texts)
    model = LogisticRegression().fit(vec.transform(texts), ids)
    joblib.dump(model, tmp_path / "best_classifier.joblib")
    with pytest.raises(ModelArtifactsError, match="tfidf_vectorizer.joblib is missing"):
        load_classifier(tmp_path)
    joblib.dump(vec, tmp_path / "tfidf_vectorizer.joblib")
    assert load_classifier(tmp_path).predict("profit shares").label == "Business"


def test_version_mismatch_warning(tmp_path, root):
    for name in ("best_classifier.joblib", "model_metadata.json"):
        shutil.copy(root / "artifacts" / name, tmp_path / name)
    meta = json.loads((tmp_path / "model_metadata.json").read_text())
    meta["sklearn_version"] = "0.0.1"
    (tmp_path / "model_metadata.json").write_text(json.dumps(meta))
    assert any("0.0.1" in w for w in load_classifier(tmp_path).load_warnings)
