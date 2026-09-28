---
title: NLP Spam Mail Detector
emoji: 📧
colorFrom: red
colorTo: orange
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 📧 Spam Mail Detector

An NLP pipeline that classifies emails and SMS messages as **spam or ham** using **TF-IDF vectorization + Naive Bayes**. Achieves **97%+ accuracy** on the SMS Spam Collection dataset with full EDA, feature importance plots, and ROC curves.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=flat-square)
![NLTK](https://img.shields.io/badge/NLTK-3.x-green?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-97.3%25-brightgreen?style=flat-square)

---

## ✨ Features

- 🔤 **Full NLP preprocessing** — lowercasing, URL/phone removal, stopword filtering, stemming
- 📊 **TF-IDF with bigrams** — captures multi-word spam patterns like "free entry", "call now"
- 🤖 **Multinomial Naive Bayes** with 5-fold cross-validation
- 📈 **Rich visualizations** — confusion matrix, ROC curve, top spam/ham feature plots
- 🧪 **Single message prediction** via CLI
- 🔧 Feature engineering — capital ratio, exclamation count, URL/phone presence

---

## 🛠️ Setup

```bash
git clone https://github.com/santoshnarreddy/spam-mail-detector
cd spam-mail-detector
pip install -r requirements.txt
```

**Get the dataset:**
```bash
# Download from Kaggle:
# https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
# Place spam.csv in data/
mkdir -p data
```

---

## 🚀 Usage

**Train the model:**
```bash
python spam_classifier.py --train --data data/spam.csv
```

**Predict a message:**
```bash
python spam_classifier.py --predict --text "Congratulations! You've won a free iPhone! Call now!"
```

**Run demo predictions (no text needed):**
```bash
python spam_classifier.py --predict
```

---

## 📊 Results

| Metric | Ham | Spam |
|---|---|---|
| Precision | 97.8% | 96.4% |
| Recall | 99.5% | 89.3% |
| F1-score | 98.6% | 92.7% |

**Overall accuracy: 97.3%** | **ROC-AUC: 0.9897**

5-Fold CV F1: **0.9521 ± 0.0084**

---

## 🧠 How It Works

```
Raw Text
   ↓
Text Cleaning (lowercase, remove URLs/phones/punctuation)
   ↓
Stemming + Stopword Removal (NLTK)
   ↓
TF-IDF Vectorization (6000 features, unigrams + bigrams)
   ↓
Multinomial Naive Bayes (alpha=0.1)
   ↓
Spam / Ham Prediction + Confidence Score
```

**Why Naive Bayes?**
Despite its simplicity, NB works exceptionally well for text classification because it models each word's probability independently — which aligns with how spam actually works (certain words are strong spam signals). Tried LogReg and RF too; NB wins on this dataset size.

---

## 📁 Project Structure

```
spam-mail-detector/
├── spam_classifier.py    # Full pipeline: preprocessing, training, prediction
├── data/
│   └── spam.csv          # Dataset (not included, download from Kaggle)
├── model/
│   ├── spam_classifier.pkl   # Saved model (created after training)
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── top_features.png
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

```
scikit-learn>=1.2
nltk>=3.8
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
```

---

## 📌 Notes

- The dataset is imbalanced (~87% ham, ~13% spam). I set `class_weight` balanced in early experiments, but the default NB actually performed better — likely because the prior probabilities encode useful information.
- Bigrams help catch phrases like "free entry", "call now", "claim prize" that NB with unigrams misses.
- The capital letter ratio was the most useful handcrafted feature — spam tends to SHOUT A LOT.

---

## 📄 License

MIT
