# Quick Start — Spam Mail Detector

## Install
```bash
pip install -r requirements.txt
```

## Train the model
1. Download SMS Spam Collection from https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
2. Place `spam.csv` in `data/`
3. Run: `python spam_classifier.py --train --data data/spam.csv`
4. Model saved to `model/spam_classifier.pkl`

## Predict a message
```bash
python spam_classifier.py --predict --model model/spam_classifier.pkl --text "Congratulations! You won a free iPhone!"
```

## Run demo (no text needed — shows 5 sample predictions)
```bash
python spam_classifier.py --predict --model model/spam_classifier.pkl
```
