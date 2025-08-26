# Fake News Detector - Flask Web App

This is a minimal Flask web app to serve a scikit-learn fake news classifier inspired by the GitHub project "Fake-News-Detection" by Kapil Singh Negi. See source repo: `https://github.com/kapilsinghnegi/Fake-News-Detection`.

## Quickstart

1. (Optional) Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train and export model artifacts (expects a CSV with `text` and `label` or common variants like `content`/`title` and `target`):

```bash
python train_export.py --data path/to/your_dataset.csv --out models
```

This produces `models/vectorizer.pkl` and `models/model.pkl`.

4. Run the web server:

```bash
python app.py
```

Open `http://localhost:5000` and paste news text to analyze.

## Deploy (Render)

1. Commit your dataset under `data/news.csv` (or adjust `render.yaml`).
2. Push to GitHub.
3. On Render, create a new Web Service from the repo. It will use `render.yaml` to build/train and `gunicorn` to serve `wsgi:app`.

## Deploy (Railway)

1. Push to GitHub and import repo in Railway.
2. Build Command:

```bash
pip install -r requirements.txt && python train_export.py --data data/news.csv --out models --model linear_svc --max_features 100000 --ngram_max 3
```

3. Start Command:

```bash
gunicorn -w 2 -k gthread -t 120 -b 0.0.0.0:$PORT wsgi:app
```

## Windows PowerShell example

```powershell
# From project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python train_export.py --data D:\data\fake_news.csv --out models
python app.py
```

## Notes

- The demo uses TF-IDF + Logistic Regression for speed and reliability.
- If you want to use models from the reference repo, export compatible artifacts (`vectorizer.pkl`, `model.pkl`).
- This tool provides a heuristic prediction and is not a substitute for professional fact-checking.

## Acknowledgment

Original ML project reference: [Fake-News-Detection on GitHub](https://github.com/kapilsinghnegi/Fake-News-Detection).

