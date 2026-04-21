from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Training data
texts = [
    "python machine learning pandas",
    "deep learning tensorflow model",
    "java spring backend api",
    "react frontend javascript ui",
    "team management leadership project",
    "recruitment hiring employee hr"
]

labels = [
    "Data Scientist",
    "Data Scientist",
    "Developer",
    "Developer",
    "Manager",
    "HR"
]

# Convert text to features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Save files
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ DONE — model files created")