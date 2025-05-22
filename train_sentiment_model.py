import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "I love this product",
    "Absolutely fantastic experience",
    "I'm very happy with it",
    "This is the worst thing ever",
    "I hate it so much",
    "Terrible and disappointing"
]
labels = [1, 1, 1, 0, 0, 0]

def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

texts = [preprocess(t) for t in texts]

# Vectorize
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train
model = LogisticRegression()
model.fit(X, labels)

# Save
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved!")
