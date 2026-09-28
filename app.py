"""
Spam Mail Detector — Gradio Web Application
Author: Santosh Narreddy
Deployable on Hugging Face Spaces or local browser.
"""

import os
import pickle
import gradio as gr
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Ensure a trained pipeline is available
MODEL_PATH = 'model/spam_classifier.pkl'

def get_pipeline():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass

    # Quick seed dataset to guarantee immediate functional demo out-of-the-box
    seed_texts = [
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt to 81010 claim.",
        "Congratulations! Your mobile number was awarded ,000,000 Walmart Gift Card! Click here.",
        "FREE MSG: Claim your free gift certificate right now by calling this number!",
        "SIX chances to win CASH! From 100 to 20,000 pounds. Text WIN to 87575.",
        "WINNER! As a valued network customer you have been selected to receive £900 reward. Call 09061701461.",
        "Hey Santosh, are we still meeting tomorrow for coffee at 4?",
        "Can you please review the pull request and merge when you get a chance?",
        "Looking forward to our sync call on Friday. Let me know if that time works.",
        "Don't forget to submit your weekly progress report by 5 PM today.",
        "The meeting notes from yesterday's architecture discussion are attached in the doc."
    ]
    seed_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ('nb', MultinomialNB(alpha=0.1))
    ])
    pipe.fit(seed_texts, seed_labels)
    os.makedirs('model', exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipe, f)
    return pipe

pipeline = get_pipeline()

def classify_text(message):
    if not message or not message.strip():
        return {}, "Please enter a message to classify."

    probs = pipeline.predict_proba([message])[0]
    ham_prob = float(probs[0])
    spam_prob = float(probs[1])

    result_dict = {
        "Legitimate (Ham)": ham_prob,
        "Spam / Phishing": spam_prob
    }

    if spam_prob >= 0.5:
        badge = f"### 🚨 Classified as SPAM ({spam_prob*100:.1f}% confidence)\n*Caution: This message exhibits patterns common in unsolicited or phishing communications.*"
    else:
        badge = f"### ✅ Classified as LEGITIMATE HAM ({ham_prob*100:.1f}% confidence)\n*This message looks like authentic personal or business communication.*"

    return result_dict, badge

examples = [
    ["Congratulations! You have been selected to win a free  Amazon gift card. Click here now!"],
    ["Hey, are we still meeting tomorrow at 3 PM to review the project code?"],
    ["URGENT: Your account has been temporarily suspended. Verify your credentials immediately."],
    ["Thanks for sending over the meeting notes. I will update the pull request shortly."]
]

demo = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(lines=4, placeholder="Enter email or SMS content here...", label="Message Text"),
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction Probabilities"),
        gr.Markdown(label="Verdict")
    ],
    examples=examples,
    title="📧 NLP Spam & Phishing Mail Detector",
    description="Classify incoming emails and SMS messages as Spam or Ham using TF-IDF vectorization and Naive Bayes machine learning.",
    article="**Author:** [Santosh Narreddy](https://github.com/santoshnarreddy) | **GitHub Repository:** [spam-mail-detector](https://github.com/santoshnarreddy/spam-mail-detector)",
    theme="default"
)

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860)
