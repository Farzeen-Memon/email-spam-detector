from flask import Flask, request, render_template
import pickle
import re

app = Flask(__name__)

# Load model and vectorizer
with open('spamnew_spam_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('spamnew_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        message = request.form['message']
        processed = preprocess_text(message)
        vect = vectorizer.transform([processed])
        pred = model.predict(vect)[0]
        prediction = 'Spam' if pred == 1 else 'Ham'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

