import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Load data
if not os.path.exists("dataset.csv"):
    raise FileNotFoundError("dataset.csv not found. Place it in the same directory.")

df = pd.read_csv("dataset.csv")

# Validate columns
if "text" not in df.columns or "category" not in df.columns:
    raise ValueError("dataset.csv must have 'text' and 'category' columns.")

df.dropna(subset=["text", "category"], inplace=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["category"], test_size=0.2, random_state=42, stratify=df["category"]
)

# Train vectorizer and model separately
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB(alpha=0.1)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer separately
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\n✅ model.pkl and vectorizer.pkl saved successfully!")
