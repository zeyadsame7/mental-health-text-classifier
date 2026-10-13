# Mental Health Text Classifier

An NLP system that analyzes short texts, such as chat messages or social-media posts, for signs of mental-health conditions. It has two parts:

1. **Multi-class classifier.** A machine-learning pipeline trained on about 53,000 labelled statements. It predicts one of 7 categories: *Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality disorder*.
2. **Risk-assessment web app.** A Flask app that combines a pretrained **Hugging Face** sentiment model with keyword detection to give a low, medium or high risk level and matching recommendations.

> ⚠️ This is an educational project. It is not a diagnostic tool and must not replace professional mental-health advice.

## How it works

### Classifier (`train_model.py`)
- **Preprocessing:** lowercasing, removing non-letters, NLTK tokenization, stop-word removal and WordNet lemmatization.
- **Class balancing:** undersamples every class to the size of the smallest one, so the model doesn't just favor the majority classes.
- **Features:** TF-IDF vectors (top 5,000 terms).
- **Model:** Random Forest (100 trees), with a stratified 70 / 15 / 15 train / validation / test split.
- **Evaluation:** per-class precision, recall and F1 (scikit-learn `classification_report`) on the validation and test sets.
- The trained vectorizer and model are saved together with `joblib`.

### Risk-assessment app (`app.py`, `mental_health_detector.py`, `templates/index.html`)
- A Hugging Face `transformers` sentiment-analysis pipeline scores the emotional tone of the message.
- Keyword groups flag indicators of depression, anxiety, suicidal thoughts, self-harm and isolation.
- A weighted score combines the two into a **low, medium or high** risk level with recommendations.
- The app exposes a JSON endpoint (`POST /analyze`) and a Bootstrap web page.

### Desktop GUI (`mental_health_gui.py`)
A Tkinter app that loads the trained classifier and shows the predicted category for any message you type.

## Dataset

This repo doesn't include the dataset, because it's too large for GitHub. The model was trained on the public Kaggle dataset **[Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)** (columns: `statement`, `status`). Download it and save it as `your_new_dataset.csv` in the project folder.

## Run it

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

python train_model.py        # train and evaluate the classifier, which saves mental_health_model.joblib
python mental_health_gui.py  # desktop app that uses the trained classifier
python app.py                # web app at http://localhost:5000
```

## Tech

Python · scikit-learn · NLTK · Hugging Face Transformers · Pandas · NumPy · Flask · Tkinter
