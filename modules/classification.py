"""Topic classification with the trained AG News TF-IDF pipeline.

The model is the one trained in ``notebooks/StudyMate_AI.ipynb``: a scikit-learn
``Pipeline(tfidf -> LogisticRegression)`` saved to
``artifacts/best_classifier.joblib``. It is loaded once (never re-trained) and
used with the same ``clean_text`` preprocessing that was used during training.

Label handling
--------------
The saved pipeline predicts integer class ids (0-3) and the names live in
``model_metadata.json``. The code does not assume this: if a classifier
returns string labels they are used as they are, and if it returns integers
they are mapped through the metadata after checking that every id has a name.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from modules.preprocessing import chunk_sentences, clean_text, split_sentences, word_count

DEFAULT_LABELS = ["World", "Sports", "Business", "Sci/Tech"]
CLASSIFIER_FILE = "best_classifier.joblib"
VECTORIZER_FILE = "tfidf_vectorizer.joblib"
METADATA_FILE = "model_metadata.json"


class ModelArtifactsError(Exception):
    """Missing or incompatible model files. The message is user-facing."""


@dataclass
class TopicPrediction:
    label: str
    confidence: Optional[float]  # None when the model has no probability estimate
    probabilities: Dict[str, float] = field(default_factory=dict)  # empty if unsupported
    confidence_note: str = ""


@dataclass
class SectionPrediction:
    section: int
    words: int
    label: str
    confidence: Optional[float]


class TopicClassifier:
    def __init__(self, pipeline, labels: Sequence[str], metadata: dict, load_warnings: List[str]):
        self.pipeline = pipeline
        self.metadata = metadata
        self.load_warnings = load_warnings
        self.model_name = metadata.get("best_model", type(self._final_estimator).__name__)
        self._class_names = self._resolve_class_names(labels)

    # ------------------------------------------------------------------
    @property
    def _final_estimator(self):
        return self.pipeline.steps[-1][1] if hasattr(self.pipeline, "steps") else self.pipeline

    def _resolve_class_names(self, labels: Sequence[str]) -> List[str]:
        classes = getattr(self._final_estimator, "classes_", None)
        if classes is None:
            raise ModelArtifactsError("The saved classifier has no 'classes_' attribute; it may not be fitted.")
        names: List[str] = []
        for cls in classes:
            if isinstance(cls, (str, np.str_)):
                names.append(str(cls))  # already a label – never index with it
            else:
                idx = int(cls)
                if idx < 0 or idx >= len(labels):
                    raise ModelArtifactsError(
                        f"The classifier predicts class id {idx}, but model_metadata.json only lists "
                        f"{len(labels)} label(s): {list(labels)}. The label mapping is inconsistent."
                    )
                names.append(str(labels[idx]))
        return names

    @property
    def labels(self) -> List[str]:
        return list(self._class_names)

    @property
    def supports_probability(self) -> bool:
        return hasattr(self.pipeline, "predict_proba") and hasattr(self._final_estimator, "predict_proba")

    # ------------------------------------------------------------------
    def _label_for(self, raw_prediction) -> str:
        classes = list(self._final_estimator.classes_)
        position = classes.index(raw_prediction)
        return self._class_names[position]

    def predict(self, text: str) -> TopicPrediction:
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("The notes are empty after cleaning; there is nothing to classify.")
        raw = self.pipeline.predict([cleaned])[0]
        label = self._label_for(raw)
        if not self.supports_probability:
            return TopicPrediction(
                label, None, {},
                f"{self.model_name} does not produce calibrated probabilities, so no confidence is shown.",
            )
        probs = self.pipeline.predict_proba([cleaned])[0]
        distribution = {name: float(p) for name, p in zip(self._class_names, probs)}
        confidence = distribution[label]
        note = "Probability from the model's predict_proba (Logistic Regression is probabilistic)."
        if confidence < 0.5:
            note = ("Low confidence: the notes do not clearly match one AG News topic. "
                    "The classifier can only choose between World, Sports, Business and Sci/Tech.")
        return TopicPrediction(label, confidence, distribution, note)

    def predict_sections(self, text: str, max_words: int = 250) -> List[SectionPrediction]:
        """Classify long notes section by section (whole sentences, ~max_words each)."""
        sentences = split_sentences(text)
        chunks = chunk_sentences(sentences, max_tokens=max_words, count_tokens=lambda s: len(s.split()))
        results: List[SectionPrediction] = []
        for chunk in chunks:
            if word_count(chunk.text) < 15:
                continue
            pred = self.predict(chunk.text)
            results.append(SectionPrediction(chunk.index + 1, word_count(chunk.text), pred.label, pred.confidence))
        return results


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_metadata(directory: Path) -> dict:
    path = directory / METADATA_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelArtifactsError(f"{METADATA_FILE} is not valid JSON: {exc}") from exc


def load_classifier(artifacts_dir: str | Path = "artifacts") -> TopicClassifier:
    """Load the trained pipeline. Raises ``ModelArtifactsError`` with a clear message."""
    import joblib
    import sklearn

    directory = Path(artifacts_dir)
    classifier_path = directory / CLASSIFIER_FILE
    vectorizer_path = directory / VECTORIZER_FILE
    if not classifier_path.exists():
        raise ModelArtifactsError(
            f"Model file not found: {classifier_path}. Copy best_classifier.joblib (and model_metadata.json) "
            "from the notebook's artifacts into the 'artifacts/' folder of the repository."
        )

    metadata = _read_metadata(directory)
    load_warnings: List[str] = []
    trained_version = metadata.get("sklearn_version")
    if trained_version and trained_version != sklearn.__version__:
        load_warnings.append(
            f"The model was trained with scikit-learn {trained_version} but {sklearn.__version__} is installed. "
            f"Pin scikit-learn=={trained_version} in requirements.txt to avoid incompatibilities."
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = joblib.load(classifier_path)
        except Exception as exc:
            raise ModelArtifactsError(
                f"{CLASSIFIER_FILE} could not be loaded ({exc.__class__.__name__}: {exc}). "
                "It was probably saved with a different scikit-learn version. Install the version listed in "
                "requirements.txt or retrain with train.py."
            ) from exc
    for warning in caught:
        if "InconsistentVersionWarning" in type(warning.message).__name__ and not load_warnings:
            load_warnings.append(str(warning.message).split(".")[0] + ".")

    if hasattr(model, "steps"):
        pipeline = model  # vectorizer is already inside the pipeline
    else:
        # A bare classifier was saved: it needs the separately saved vectorizer.
        if not vectorizer_path.exists():
            raise ModelArtifactsError(
                f"{CLASSIFIER_FILE} contains only a classifier, and {VECTORIZER_FILE} is missing, "
                "so the TF-IDF features cannot be rebuilt."
            )
        from sklearn.pipeline import Pipeline

        vectorizer = joblib.load(vectorizer_path)
        pipeline = Pipeline([("tfidf", vectorizer), ("classifier", model)])

    if not hasattr(pipeline, "predict"):
        raise ModelArtifactsError(f"{CLASSIFIER_FILE} does not contain a scikit-learn model.")

    labels = metadata.get("labels") or DEFAULT_LABELS
    return TopicClassifier(pipeline, labels, metadata, load_warnings)
