from flask import Flask, render_template, request
import joblib
import string

# Load the trained model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

# Cleaning function used during training
def clean_text(msg):
    if not isinstance(msg, str):
        return ""
    msg = msg.lower()
    msg = ''.join([char for char in msg if char not in string.punctuation])
    return msg.strip()

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        msg = request.form["message"]
        msg_clean = clean_text(msg)
        msg_vector = vectorizer.transform([msg_clean])
        result = model.predict(msg_vector)[0]

        if result.lower() == "ham":
            prediction = "✅ Safe Message"
        else:
            prediction = "❌ Potential Scam or Spam"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
