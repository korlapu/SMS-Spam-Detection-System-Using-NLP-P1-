# SMS-Spam-Detection-System-Using-NLP-P1-
SMS Spam Detection System Using NLP (P1)
pip install pandas scikit-learn nltk

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('stopwords')
nltk.download('wordnet')
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.rename(columns={"v1": "label", "v2": "message"})[['label', 'message']]

# Encode labels (spam = 1, ham = 0)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize and remove stopwords
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

df['message'] = df['message'].apply(preprocess_text)
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.3, random_state=42)
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
def predict_spam(message):
    message = preprocess_text(message)
    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)
    return "Spam" if prediction[0] == 1 else "Not Spam"

test_message = "Congratulations! You've won a free ticket to Bahamas. Call now!"
print(f"Message: '{test_message}' is {predict_spam(test_message)}")
