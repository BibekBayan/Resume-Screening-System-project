from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)  # Allow requests from HTML files opened in browser

base = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(base, "model.pkl"))
vectorizer = joblib.load(os.path.join(base, "vectorizer.pkl"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    classes = model.classes_

    # Build probabilities dict for all categories
    probabilities = {
        cls: round(float(p) * 100, 2)
        for cls, p in zip(classes, proba)
    }

    return jsonify({
        "category": prediction,
        "confidence": round(float(proba.max()) * 100, 2),
        "probabilities": probabilities
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
