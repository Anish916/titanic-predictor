# Titanic Survival Predictor

A Streamlit web app that serves the XGBoost model from the Titanic
survival prediction notebook.

## Files

| File | Purpose |
| --- | --- |
| `train_model.py` | Trains the pipeline and writes `model.joblib` |
| `app.py` | The Streamlit web app |
| `model.joblib` | The trained model (created by `train_model.py`) |
| `requirements.txt` | Python dependencies |
| `train.csv` | Kaggle training data (you supply this) |

## Run locally

```bash
pip install -r requirements.txt
python train_model.py      # writes model.joblib, prints test accuracy
streamlit run app.py       # opens on http://localhost:8501
```

## Deploy to Streamlit Community Cloud (free)

1. Push this folder to a public GitHub repository. Commit `model.joblib`
   so the server does not need to retrain. Leave `train.csv` out if you
   prefer, the app does not read it.
2. Go to share.streamlit.io and sign in with GitHub.
3. Choose "New app", pick your repository and branch, set the main file
   to `app.py`, then choose "Deploy".
4. The first build takes two or three minutes while the dependencies
   install. After that you get a public URL ending in `.streamlit.app`.

Pushing a new commit to the branch redeploys the app automatically.

## Other hosting options

**Hugging Face Spaces**: create a Space, choose the Streamlit SDK, upload
the same files. Works the same way and is also free.

**Render or Railway**: add a start command of
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

**Docker**: see `Dockerfile` in this folder.

## Notes on the model

Test accuracy is about 83 percent on a held-out fifth of the data. It was
trained on 891 passengers, so uncommon input combinations rest on very few
examples and the probability can jump between nearby fares. The features
are ticket class, sex, age, siblings or spouses aboard, parents or
children aboard, fare and port of embarkation, matching the notebook.
