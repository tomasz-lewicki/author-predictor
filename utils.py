import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer

def accuracy(pred, truth):
    return np.sum(pred == truth)/truth.shape[0]

def read_dataset():
    # Reads the dataset from the /data directory, vectorizes data,
    # and converts feature vector X into tf-idf

    # read csv files
    train, test = pd.read_csv("./data/train.csv"), pd.read_csv("./data/test.csv")

    # Convert Create author name -> int mapping
    # We could've used sklearn.preprocessing.LabelEncoder for that as well.
    label_meaning = OrderedDict()
    for idx, l in enumerate(train.author.unique()):
        label_meaning[l] = idx

    y = np.array(list(map(label_meaning.get, train.author.values)))

    X_raw = train.text

    # Read features. I use TfidfVectorizer params from:
    # https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle

    vectorizer = TfidfVectorizer(
        min_df=3,
        max_features=None,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
        ngram_range=(1, 3),
        use_idf=1,
        smooth_idf=1,
        sublinear_tf=1,
        stop_words="english",
    )

    vectorizer.fit(X_raw)
    X_tfidf = vectorizer.transform(X_raw)

    return X_tfidf, y
