from flask import Flask, request, jsonify, render_template
import joblib
import re
import string
from nltk.stem import WordNetLemmatizer

app = Flask(__name__, template_folder="templates1")


model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("text")

        if user_text:
            cleaned = clean_text(user_text)
            vec = tfidf.transform([cleaned])
            pred = model.predict(vec)[0]
            result = "Positive 😊" if pred == 1 else "Negative 😡"

    return render_template("index1.html", result=result, user_text=user_text)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        user_text = request.args.get("text")
    else:
        data = request.get_json()
        user_text = None
        if data and "text" in data:
            user_text = data["text"]

    if not user_text:
        return jsonify({"error": "Please provide text"}), 400

    cleaned = clean_text(user_text)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]

    sentiment = "Positive" if pred == 1 else "Negative"

    return jsonify({
        "input_text": user_text,
        "sentiment": sentiment
    })

if __name__ == "__main__":
    app.run(debug=True)
