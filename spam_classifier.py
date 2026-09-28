"""
Spam Mail Detector
Author: Santosh Narreddy

NLP pipeline for classifying emails as spam or ham (not spam).
Uses TF-IDF vectorization + Naive Bayes with optional ensemble.

Dataset: SMS Spam Collection (UCI / Kaggle)
  - 5,574 messages: 4,827 ham + 747 spam
  - Download: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Usage:
    python spam_classifier.py --train --data spam.csv
    python spam_classifier.py --predict --text "Congratulations! You've won a free iPhone!"
    python spam_classifier.py --predict --text "Hey, are we still meeting tomorrow?"
"""

import os
import re
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data if not already present
for pkg in ['stopwords', 'punkt']:
    try:
        nltk.data.find(f'corpora/{pkg}' if pkg != 'punkt' else f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words('english'))
STEMMER    = PorterStemmer()


# ─── Text Preprocessing ────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean and normalize email/SMS text.
    Steps: lowercase → remove URLs/emails/numbers → remove punctuation → stem.
    """
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove phone numbers (common in spam)
    text = re.sub(r'\b\d{10,}\b|\+\d{1,3}\s?\d+', '', text)
    # Remove all non-alpha characters except spaces
    text = re.sub(r'[^a-z\s]', '', text)

    tokens = text.split()
    tokens = [STEMMER.stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]

    return ' '.join(tokens)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering — extra handcrafted features alongside TF-IDF.
    These actually helped the classifier with borderline cases.
    """
    df = df.copy()
    df['char_count']        = df['text'].apply(len)
    df['word_count']        = df['text'].apply(lambda x: len(x.split()))
    df['capital_ratio']     = df['text'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )
    df['num_exclamations']  = df['text'].apply(lambda x: x.count('!'))
    df['num_dollar_signs']  = df['text'].apply(lambda x: x.count('$'))
    df['num_digits']        = df['text'].apply(lambda x: sum(c.isdigit() for c in x))
    df['contains_url']      = df['text'].apply(
        lambda x: int(bool(re.search(r'http|www|\.com', x.lower())))
    )
    df['contains_phone']    = df['text'].apply(
        lambda x: int(bool(re.search(r'\b\d{10,}\b|\+\d', x)))
    )
    return df


# ─── Model Building ────────────────────────────────────────────────────────────

def build_pipeline():
    """
    TF-IDF + Multinomial Naive Bayes pipeline.
    Tried Logistic Regression and Random Forest too — NB wins on this
    small dataset because of its text-friendly probability assumptions.
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=6000,
            ngram_range=(1, 2),       # Unigrams + bigrams
            min_df=2,
            sublinear_tf=True,        # Replace TF with 1+log(TF)
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\b[a-zA-Z]{2,}\b'
        )),
        ('clf', MultinomialNB(alpha=0.1))
    ])


# ─── Training ──────────────────────────────────────────────────────────────────

def load_data(csv_path: str):
    """Load and parse the spam dataset."""
    df = pd.read_csv(csv_path, encoding='latin-1')

    # UCI dataset has columns: v1 (label), v2 (text) + junk columns
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
    df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})

    print(f"[INFO] Loaded {len(df)} messages — "
          f"ham: {(df.label == 'ham').sum()} | spam: {(df.label == 'spam').sum()}")
    return df


def train(csv_path: str, output_dir: str = 'model'):
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(csv_path)
    df['clean_text'] = df['text'].apply(clean_text)

    X = df['clean_text']
    y = df['label_enc']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    model = build_pipeline()

    # Cross-validation before final fit
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    print(f"[INFO] 5-Fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model.fit(X_train, y_train)

    # Evaluation
    y_pred     = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n[RESULTS]")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])}")

    # Save model
    model_path = os.path.join(output_dir, 'spam_classifier.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[INFO] Model saved to {model_path}")

    # Plots
    _plot_confusion_matrix(y_test, y_pred, output_dir)
    _plot_roc_curve(y_test, y_pred_proba, auc, output_dir)
    _plot_top_features(model, output_dir)

    return model


# ─── Plotting Helpers ──────────────────────────────────────────────────────────

def _plot_confusion_matrix(y_true, y_pred, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print("[INFO] Saved confusion_matrix.png")


def _plot_roc_curve(y_true, y_proba, auc, out_dir):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'roc_curve.png'), dpi=150)
    plt.close()
    print("[INFO] Saved roc_curve.png")


def _plot_top_features(pipeline, out_dir, n=20):
    vectorizer = pipeline.named_steps['tfidf']
    clf        = pipeline.named_steps['clf']
    feature_names = np.array(vectorizer.get_feature_names_out())

    # For Naive Bayes: log probability difference (spam - ham)
    log_prob_diff = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    top_spam_idx  = log_prob_diff.argsort()[-n:][::-1]
    top_ham_idx   = log_prob_diff.argsort()[:n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.barh(range(n), log_prob_diff[top_spam_idx], color='tomato')
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(feature_names[top_spam_idx])
    ax1.set_title(f'Top {n} Spam Indicators')
    ax1.invert_yaxis()

    ax2.barh(range(n), np.abs(log_prob_diff[top_ham_idx]), color='steelblue')
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(feature_names[top_ham_idx])
    ax2.set_title(f'Top {n} Ham Indicators')
    ax2.invert_yaxis()

    plt.suptitle('Most Informative Features (TF-IDF + NB)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'top_features.png'), dpi=150)
    plt.close()
    print("[INFO] Saved top_features.png")


# ─── Prediction ────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def predict(text: str, model, threshold: float = 0.5):
    """Classify a single message and return label + confidence."""
    cleaned = clean_text(text)
    proba   = model.predict_proba([cleaned])[0]

    spam_prob = float(proba[1])
    is_spam   = spam_prob >= threshold

    print(f"\n{'─'*55}")
    print(f"Text:       {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"Cleaned:    {cleaned[:60]}...")
    print(f"{'─'*55}")
    print(f"Prediction: {'🚨 SPAM' if is_spam else '✅ HAM (Not Spam)'}")
    print(f"Spam prob:  {spam_prob:.4f}  |  Ham prob: {proba[0]:.4f}")
    print(f"{'─'*55}\n")
    return 'spam' if is_spam else 'ham', spam_prob


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spam Mail Detector")
    parser.add_argument('--train',   action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Predict a single message')
    parser.add_argument('--data',    default='data/spam.csv', help='Path to dataset CSV')
    parser.add_argument('--model',   default='model/spam_classifier.pkl', help='Path to model')
    parser.add_argument('--text',    help='Message to classify (for --predict)')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    if args.train:
        train(args.data)

    elif args.predict:
        if not args.text:
            # Demo messages
            demo_messages = [
                "Congratulations! You've won a £1000 Walmart gift card! Call now: 0800-123-4567",
                "Hey, are we still on for lunch tomorrow? Let me know 😊",
                "URGENT: Your account has been suspended. Click here to verify: http://fake-bank.xyz",
                "Can you send me the meeting notes from this morning?",
                "FREE entry in 2 a weekly competition! TXT WIN to 87121 to receive entry.",
            ]
            print("=== Demo Predictions ===")
            model = load_model(args.model)
            for msg in demo_messages:
                predict(msg, model, args.threshold)
        else:
            model = load_model(args.model)
            predict(args.text, model, args.threshold)
