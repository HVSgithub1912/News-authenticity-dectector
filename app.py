import os
import re
from typing import Tuple

from flask import Flask, render_template, request, redirect, url_for, flash
import joblib


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_ROOT, "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")

    # Load model artifacts if present
    app.vectorizer, app.model = load_artifacts()

    @app.route("/", methods=["GET"])  # type: ignore[misc]
    def index():
        model_ready = app.vectorizer is not None and app.model is not None
        return render_template("index.html", model_ready=model_ready)

    @app.route("/predict", methods=["POST"])  # type: ignore[misc]
    def predict():
        if app.vectorizer is None or app.model is None:
            flash("Model not found. Please run training first.")
            return redirect(url_for("index"))

        text = request.form.get("news_text", "").strip()
        if not text:
            flash("Please paste some news text to analyze.")
            return redirect(url_for("index"))

        cleaned = basic_clean(text)
        X = app.vectorizer.transform([cleaned])
        pred = app.model.predict(X)[0]
        proba = None
        try:
            proba = float(app.model.predict_proba(X)[0][1])
        except Exception:
            proba = None

        label = "Fake" if int(pred) == 1 else "Not Fake"
        confidence = f"{proba*100:.1f}%" if proba is not None else "N/A"

        return render_template(
            "index.html",
            model_ready=True,
            input_text=text,
            prediction_label=label,
            prediction_confidence=confidence,
        )

    return app


def load_artifacts() -> Tuple[object, object]:
    if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH)):
        return None, None  # type: ignore[return-value]
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)
        return vectorizer, model
    except Exception:
        return None, None  # type: ignore[return-value]


_NON_ALPHA = re.compile(r"[^a-zA-Z\s]")
_MULTI_SPACE = re.compile(r"\s+")


def basic_clean(text: str) -> str:
    text = text.lower()
    text = _NON_ALPHA.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)



