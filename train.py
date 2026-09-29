"""Train the SpamSieve classifier.

Downloads the UCI SMS Spam Collection (if not already cached), compares a few
candidate models with cross-validation, evaluates the winner on a held-out
test set and saves the full pipeline to models/spamsieve.joblib.

Usage:
    python train.py
"""

import io
import json
import urllib.request
import warnings
import zipfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from spamsieve.text import normalise

DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "SMSSpamCollection"
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "spamsieve.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"
SEED = 7

# numpy on macOS (Accelerate) emits spurious matmul overflow warnings during
# fitting; the results are unaffected.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")


def fetch_dataset() -> pd.DataFrame:
    if not DATA_FILE.exists():
        print(f"Downloading dataset from {DATASET_URL} ...")
        DATA_DIR.mkdir(exist_ok=True)
        with urllib.request.urlopen(DATASET_URL) as resp:
            archive = zipfile.ZipFile(io.BytesIO(resp.read()))
        DATA_FILE.write_bytes(archive.read("SMSSpamCollection"))

    df = pd.read_csv(DATA_FILE, sep="\t", header=None, names=["label", "message"], quoting=3, encoding="utf-8")
    before = len(df)
    df = df.dropna().drop_duplicates(subset="message").reset_index(drop=True)
    print(f"Loaded {before} messages, {len(df)} after removing duplicates")
    print(df["label"].value_counts().to_string(), "\n")
    return df


def build_features() -> FeatureUnion:
    # Word n-grams catch phrases like "call now"; character n-grams catch
    # obfuscated spellings like "fr33" or "w1n".
    return FeatureUnion([
        ("words", TfidfVectorizer(preprocessor=normalise, ngram_range=(1, 2), min_df=2,
                                  stop_words="english", sublinear_tf=True)),
        ("chars", TfidfVectorizer(preprocessor=normalise, analyzer="char_wb", ngram_range=(3, 5),
                                  min_df=3, sublinear_tf=True)),
    ])


def candidates() -> dict:
    return {
        "complement_nb": ComplementNB(alpha=0.3),
        "logistic_regression": LogisticRegression(C=20, max_iter=2000, class_weight="balanced"),
        "linear_svm": CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=3),
        "random_forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
    }


def main() -> None:
    df = fetch_dataset()
    X = df["message"]
    y = (df["label"] == "spam").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {}
    print(f"{'model':<22}{'precision':>10}{'recall':>10}{'f1':>10}")
    for name, clf in candidates().items():
        pipe = Pipeline([("features", build_features()), ("clf", clf)])
        cv = cross_validate(pipe, X_train, y_train, cv=folds, scoring=["precision", "recall", "f1"])
        scores[name] = {m: float(cv[f"test_{m}"].mean()) for m in ("precision", "recall", "f1")}
        s = scores[name]
        print(f"{name:<22}{s['precision']:>10.3f}{s['recall']:>10.3f}{s['f1']:>10.3f}")

    # F1 balances not flagging real messages (precision) against actually
    # catching spam (recall); precision alone rewards a model that barely
    # flags anything.
    best = max(scores, key=lambda n: scores[n]["f1"])
    print(f"\nBest model by cross-validated F1: {best}\n")

    model = Pipeline([("features", build_features()), ("clf", candidates()[best])])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("Held-out test set:")
    print(classification_report(y_test, pred, target_names=["ham", "spam"], digits=3))
    print("Confusion matrix [[ham→ham, ham→spam], [spam→ham, spam→spam]]:")
    print(confusion_matrix(y_test, pred))

    # Refit on all data for the shipped model now that we've measured it.
    model.fit(X, y)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    METRICS_FILE.write_text(json.dumps({
        "model": best,
        "cv": scores,
        "test": {
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
        },
        "n_messages": len(df),
    }, indent=2))
    print(f"\nSaved {MODEL_FILE} and {METRICS_FILE}")


if __name__ == "__main__":
    main()
