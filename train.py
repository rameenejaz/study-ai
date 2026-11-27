"""Retrain the AG News topic classifier (migration path if the saved artifacts ever become incompatible).

This reproduces notebook sections 5–11 exactly (same cleaning, TF-IDF settings,
five algorithms, selection by macro-F1, 8,000 records per class by default)
and writes the same artifact files. The Streamlit app never trains: it only
loads these files.

Usage (macOS):
    python3 -m pip install datasets pandas
    python3 train.py                       # 8,000 per class, as in the notebook
    python3 train.py --sample-per-class 0  # all 120,000 training records
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from modules.preprocessing import clean_text

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
SEED = 42
MAX_FEATURES = 30000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-per-class", type=int, default=8000,
                        help="Training records per class (notebook default 8000); 0 uses all 120,000.")
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()

    from datasets import load_dataset  # only needed for training, not for the app

    data = load_dataset("fancyzhx/ag_news")
    train_df, test_df = pd.DataFrame(data["train"]), pd.DataFrame(data["test"])
    if args.sample_per_class:
        train_df = (train_df.groupby("label", group_keys=False)
                    .apply(lambda x: x.sample(n=min(len(x), args.sample_per_class), random_state=SEED))
                    .reset_index(drop=True))
    X_train, y_train = train_df.text.map(clean_text), train_df.label.astype(int)
    X_test, y_test = test_df.text.map(clean_text), test_df.label.astype(int)

    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=4.0),
        "Linear SVM": LinearSVC(C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=SEED),
        "k-NN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }
    results, fitted = [], {}
    for name, algorithm in models.items():
        print("Training:", name)
        pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2), min_df=2,
                                                   sublinear_tf=True)),
                         ("classifier", algorithm)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
        results.append({"Model": name, "Accuracy": accuracy_score(y_test, pred),
                        "Macro Precision": p, "Macro Recall": r, "Macro F1": f1})
        fitted[name] = pipe

    results_df = pd.DataFrame(results).sort_values("Macro F1", ascending=False)
    best_name = results_df.iloc[0]["Model"]
    best = fitted[best_name]

    os.makedirs(args.output, exist_ok=True)
    joblib.dump(best, os.path.join(args.output, "best_classifier.joblib"))
    joblib.dump(best.named_steps["tfidf"], os.path.join(args.output, "tfidf_vectorizer.joblib"))
    results_df.to_csv(os.path.join(args.output, "model_comparison.csv"), index=False)
    metadata = {
        "best_model": best_name,
        "labels": LABELS,
        "training_records_used": len(X_train),
        "test_records": len(X_test),
        "dataset": "fancyzhx/ag_news",
        "results": results_df.to_dict(orient="records"),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "label_mapping": {str(i): label for i, label in enumerate(LABELS)},
    }
    with open(os.path.join(args.output, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(results_df.to_string(index=False))
    print(f"Best model: {best_name}. Artifacts saved in ./{args.output}. "
          f"Set scikit-learn=={sklearn.__version__} in requirements.txt.")


if __name__ == "__main__":
    main()
