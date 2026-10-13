"""Evaluate the original TF-IDF + Random Forest model on the shared splits.

Usage:  python benchmark_baseline.py --setting balanced
        python benchmark_baseline.py --setting full
"""
import argparse, json, os, time

import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score

from data_utils import get_splits

for pkg, path in [("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab"),
                  ("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

from train_model import MentalHealthModelTrainer  # noqa: E402  (needs the NLTK data above)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["balanced", "full"], default="balanced")
    args = ap.parse_args()

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_splits(args.setting)
    prep = MentalHealthModelTrainer().preprocess_text  # same preprocessing as the original model

    t0 = time.time()
    X_train_p, X_val_p, X_test_p = (s.apply(prep) for s in (X_train, X_val, X_test))
    vectorizer = TfidfVectorizer(max_features=5000)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(vectorizer.fit_transform(X_train_p), y_train)
    train_seconds = time.time() - t0

    results = {"model": "TF-IDF + Random Forest", "setting": args.setting,
               "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
               "train_seconds": round(train_seconds, 1)}
    for name, Xp, y in [("val", X_val_p, y_val), ("test", X_test_p, y_test)]:
        pred = clf.predict(vectorizer.transform(Xp))
        results[name] = {
            "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"),
            "weighted_f1": f1_score(y, pred, average="weighted"),
            "report": classification_report(y, pred, output_dict=True),
        }
        print(f"[{args.setting}] {name}: accuracy={results[name]['accuracy']:.4f} "
              f"macro-F1={results[name]['macro_f1']:.4f}")

    os.makedirs("results", exist_ok=True)
    with open(f"results/baseline_{args.setting}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
