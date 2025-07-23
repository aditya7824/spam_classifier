# Spam Classifier

A machine learning web application that classifies SMS messages as Spam or Ham using Naive Bayes and TF-IDF vectorization.

## Project Files

```
spam_classifier/
├── main.py             # Model training script
├── spam_app.py         # Streamlit web application
├── predict.py          # Command-line prediction tool
├── utils.py            # Text preprocessing utilities
├── model.pkl           # Trained Naive Bayes model
├── vectorizer.pkl      # TF-IDF vectorizer
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Steps to run

1. git clone https://github.com/aditya7824/spam_classifier.git
2. cd spam_classifier
3. pip install -r requirements.txt
4. python main.py
5. python -m streamlit run spam_app.py
