
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("sentiment.csv")  # IMDB
df["sentiment"] = df["sentiment"].map({"negative":0, "positive":1})
print(df.head())

#text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["clean_review"] = df["review"].apply(clean_text)

#TF-IDF feature extraction
X = df["clean_review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

#naive bayes model
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

nb_pred = nb.predict(X_test_tfidf)
nb_acc = accuracy_score(y_test, nb_pred)

print("Naïve Bayes Accuracy:", nb_acc)

#logistic regression model
lr = LogisticRegression(max_iter=300)
lr.fit(X_train_tfidf, y_train)

lr_pred = lr.predict(X_test_tfidf)
lr_acc = accuracy_score(y_test, lr_pred)

print("Logistic Regression Accuracy:", lr_acc)

#confusion matrix for logistic regression
cm = confusion_matrix(y_test, lr_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Logistic Regression")
plt.show()

#LSTM model deep learning 
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

(X_train_lstm, y_train_lstm), (X_test_lstm, y_test_lstm) = imdb.load_data(num_words=10000)

X_train_lstm = pad_sequences(X_train_lstm, maxlen=200)
X_test_lstm = pad_sequences(X_test_lstm, maxlen=200)

model = Sequential()
model.add(Embedding(10000, 128, input_length=200))
model.add(LSTM(128))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train_lstm, y_train_lstm,
    epochs=3,
    batch_size=64,
    validation_split=0.2
)
lstm_loss, lstm_acc = model.evaluate(X_test_lstm, y_test_lstm)
print("LSTM Accuracy:", lstm_acc)

#accuracy comparison plot
models = ["Naïve Bayes", "Logistic Regression", "LSTM"]
accuracies = [nb_acc, lr_acc, lstm_acc]

plt.bar(models, accuracies)
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.show()

#sentiment chatbot deployment logic 
def sentiment_chatbot(text):
    text = clean_text(text)
    vec = tfidf.transform([text])
    pred = lr.predict(vec)[0]

    if pred == 1:
        return "That sounds positive :)"
    else:
        return "Im sorry to hear that :("

while True:
    user = input("You: ")
    if user.lower() in ["bye", "exit"]:
        break
    print("Bot:", sentiment_chatbot(user))
