import html
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def tfidf_features(train_texts, val_texts, test_texts, ngram_range=(1, 1),
                   max_features=None, min_df=1, max_df=1.0, stop_words=None):
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words=stop_words
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)
    x_test = vectorizer.transform(test_texts)
    return vectorizer, x_train, x_val, x_test


def train_classifier(x_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    return model


def predict(model, x):
    return model.predict(x)


def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }
