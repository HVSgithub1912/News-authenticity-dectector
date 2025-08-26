import os
import argparse
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def load_dataset(dataset_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    # Try common schema variants
    text_col_candidates = ["text", "title", "content"]
    label_col_candidates = ["label", "labels", "target", "y", "class"]

    text_col = next((c for c in text_col_candidates if c in df.columns), None)
    label_col = next((c for c in label_col_candidates if c in df.columns), None)

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not find text/label columns in CSV. Columns present: {list(df.columns)}"
        )

    return df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})


def train_and_export(
    dataset_path: str,
    output_dir: str = "models",
    max_features: int = 50000,
    ngram_max: int = 2,
    model_type: str = "linear_svc",
):
    os.makedirs(output_dir, exist_ok=True)

    df = load_dataset(dataset_path)
    df = df.dropna(subset=["text", "label"])  # basic cleanup

    # Choose a test size that works even for tiny demo datasets
    n_samples = len(df)
    min_test_frac = max(0.2, min(0.5, 2.0 / max(1, n_samples)))
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"], df["label"], test_size=min_test_frac, random_state=42, stratify=df["label"]
        )
    except ValueError:
        # Fallback: no stratify for very small datasets
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"], df["label"], test_size=min_test_frac, random_state=42, stratify=None
        )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, ngram_max),
    )

    if model_type == "linear_svc":
        base_clf = LinearSVC(class_weight="balanced")
        # Calibrate to get probabilities for UI confidence display
        clf = CalibratedClassifierCV(estimator=base_clf, cv=3)
        pipeline = Pipeline([
            ("tfidf", vectorizer),
            ("clf", clf),
        ])
        # Small param search; safe defaults if dataset is tiny
        param_grid = {
            "tfidf__ngram_range": [(1, 2), (1, 3)],
            "tfidf__max_features": [30000, max_features],
            "clf__estimator__C": [0.5, 1.0, 2.0],
        }
        try:
            grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=0)
            grid.fit(X_train, y_train)
            pipeline = grid.best_estimator_
        except Exception:
            pipeline.fit(X_train, y_train)
    else:
        model = LogisticRegression(max_iter=500, class_weight="balanced")
        pipeline = Pipeline([
            ("tfidf", vectorizer),
            ("clf", model),
        ])
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Persist separate artifacts for the web app
    fitted_vectorizer = pipeline.named_steps["tfidf"]
    fitted_model = pipeline.named_steps["clf"]

    joblib.dump(fitted_vectorizer, os.path.join(output_dir, "vectorizer.pkl"))
    joblib.dump(fitted_model, os.path.join(output_dir, "model.pkl"))
    print(f"Saved artifacts to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and export fake news model")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to CSV file containing columns: text, label (or common variants)",
    )
    parser.add_argument("--out", default="models", help="Output directory for artifacts")
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--model", choices=["linear_svc", "logreg"], default="linear_svc")
    args = parser.parse_args()

    train_and_export(
        dataset_path=args.data,
        output_dir=args.out,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        model_type=args.model,
    )



