import joblib
from utils import preprocess_text

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def predict_spam(message):
    cleaned = preprocess_text(message)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    return "Spam" if prediction else "Ham"

# Example
if __name__ == "__main__":
    msg = input("Enter your SMS: ")
    print("Prediction:", predict_spam(msg))
