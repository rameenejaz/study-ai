import os, re, json, argparse, joblib
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

LABELS = ["World", "Sports", "Business", "Sci/Tech"]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-per-class", type=int, default=0,
                        help="Optional training cap per class; 0 uses all records.")
    args = parser.parse_args()
    data = load_dataset("fancyzhx/ag_news")
    train_df, test_df = pd.DataFrame(data["train"]), pd.DataFrame(data["test"])
    if args.sample_per_class:
        train_df = (train_df.groupby("label", group_keys=False)
                    .apply(lambda x: x.sample(min(len(x), args.sample_per_class), random_state=42))
                    .reset_index(drop=True))
    X_train, y_train = train_df.text.map(clean_text), train_df.label.astype(int)
    X_test, y_test = test_df.text.map(clean_text), test_df.label.astype(int)
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=4.0),
        "Linear SVM": LinearSVC(C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=42),
        "k-NN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    }
    results, best, best_f1 = [], None, -1
    for name, clf in models.items():
        pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=2, sublinear_tf=True)),
                         ("classifier", clf)])
        print("Training", name)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
        acc = accuracy_score(y_test, pred)
        results.append({"model": name, "accuracy": acc, "macro_precision": p, "macro_recall": r, "macro_f1": f1})
        if f1 > best_f1:
            best, best_f1, best_name = pipe, f1, name
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(best, "artifacts/best_classifier.joblib")
    joblib.dump(best.named_steps["tfidf"], "artifacts/tfidf_vectorizer.joblib")
    pd.DataFrame(results).to_csv("artifacts/model_comparison.csv", index=False)
    with open("artifacts/model_metadata.json", "w") as f:
        json.dump({"best_model": best_name, "labels": LABELS, "training_records_used": len(train_df),
                   "test_records": len(test_df), "dataset": "fancyzhx/ag_news", "results": results}, f, indent=2)
    print(pd.DataFrame(results).sort_values("macro_f1", ascending=False).to_string(index=False))
    print("Artifacts saved in ./artifacts")

if __name__ == "__main__":
    main()
