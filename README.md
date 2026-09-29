# 🛡️ SpamSieve

SpamSieve is a small web app that tells you whether a text message looks like spam. Paste in a single message, or upload a CSV of messages to check them all at once and download the results.

## Features

- **Spam probability.** You get a confidence score for each message, not just a yes/no answer.
- **Adjustable threshold.** Raise it to flag fewer messages or lower it to catch more.
- **Batch checking.** Upload a CSV, pick the message column, and download the results with a spam probability for each row.

## How it works

**Data.** The model is trained on the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (5,574 English SMS messages labelled *ham* or *spam*). `train.py` downloads it automatically and drops duplicate messages, which leaves 5,171.

**Preprocessing** ([spamsieve/text.py](spamsieve/text.py)). Spam relies on links, phone numbers and prices, so instead of deleting these the cleaner replaces them with tokens such as `_url_`, `_phone_` and `_money_` that the model can learn from.

**Features.** Two TF-IDF representations are combined:
- word unigrams and bigrams, which catch phrases like "call now" and "claim your"
- character 3–5-grams, which catch obfuscated spellings like "fr33"

**Model selection.** Four classifiers are compared with 5-fold stratified cross-validation on the training split, and the one with the best F1 score is kept. I used F1 rather than precision alone because a model that barely flags anything can still get perfect precision.

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Complement Naive Bayes | 0.871 | 0.965 | 0.916 |
| Logistic Regression | 0.974 | 0.944 | 0.959 |
| **Linear SVM (calibrated)** | **0.976** | **0.944** | **0.960** |
| Random Forest | 0.977 | 0.906 | 0.940 |

On the held-out 20% test set, the chosen model scores **98.3% precision** and **90.1% recall** on spam. It flags 2 of 904 legitimate messages as spam. Once evaluated, the model is refit on the full dataset before it is saved.

## Running it locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py          # downloads data, trains, writes models/
streamlit run app.py
```

## Project structure

```
app.py              Streamlit interface
train.py            data download, model comparison, training
spamsieve/text.py   text normalisation shared by training and the app
models/             trained pipeline and its metrics
```

## Acknowledgements

The dataset is the SMS Spam Collection by Tiago A. Almeida and José María Gómez Hidalgo, available from the UCI Machine Learning Repository under CC BY 4.0.
