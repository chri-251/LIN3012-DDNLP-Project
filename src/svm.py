# from nltk import ngrams
# import nltk
# from nltk.corpus import stopwords
# import math
# import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle
import winsound


def read_labels(train_filename, test_filename):
    train_labels_file = open(train_filename, 'r')
    train_labels = train_labels_file.read().split("\n")[:-1]
    train_labels_file.close()

    test_labels_file = open(test_filename, 'r')
    test_labels = test_labels_file.read().split("\n")[:-1]
    test_labels_file.close()

    return train_labels, test_labels


def get_classification():
    mappings = open("../dataset/us/us_mapping.txt", 'r', encoding="utf-8")

    labels = []
    for line in mappings:
        split_line = line.split('\t')
        labels.append(split_line[1] + " " + split_line[2] + "(" + split_line[0] + ")")

    mappings.close()
    return labels


def svm_tfidf(save, train_tweets, test_tweets, train_emoji_labels, test_emoji_labels):
    # converts training and test data to TFIDF
    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_tweets)
    train_x_tfidf = vectorizer.transform(train_tweets)
    test_x_tfidf = vectorizer.transform(test_tweets)
    # print(vectorizer.vocabulary_)

    if save:
        print("Creating and fitting SVM classifier...")
        svc = svm.SVC(C=1.0, kernel='linear', degree=3)
        svc.fit(train_x_tfidf, train_emoji_labels)
        pickle.dump(svc, open('../models/svc.sav', 'wb'))
    else:
        print("Loading existing SVM classifier...")
        svc = pickle.load(open('../models/svc.sav', 'rb'))

    # predict the labels on validation dataset
    pred_emoji_labels = svc.predict(test_x_tfidf)
    emoji_labels = get_classification()

    print("\nAccuracy Score:", accuracy_score(test_emoji_labels, pred_emoji_labels))
    print('\nConfusion Matrix:\n', confusion_matrix(test_emoji_labels, pred_emoji_labels))
    print('\nClassification Report:\n', classification_report(test_emoji_labels, pred_emoji_labels,
                                                              target_names=emoji_labels))
    # labels=np.unique(pred_emoji_labels)


if __name__ == '__main__':
    # nltk.download()

    train_file = "../dataset/us/pre-processed data/train/text.txt"
    test_file = "../dataset/us/pre-processed data/test/text.txt"
    train_label_file = "../dataset/us/raw data/train/us_train.LABELS"
    test_label_file = "../dataset/us/raw data/test/us_test.LABELS"

    train_emoji_labels, test_emoji_labels = read_labels(train_label_file, test_label_file)

    # stores all test data tweets
    train_tweets = []
    with open(train_file, 'r', encoding='utf-8') as file:
        for line in file:
            train_tweets.append(line)

    # stores all test data tweets
    test_tweets = []
    with open(test_file, 'r', encoding='utf-8') as file:
        for line in file:
            test_tweets.append(line)

    svm_tfidf(True, train_tweets, test_tweets, train_emoji_labels, test_emoji_labels)

    winsound.Beep(1500, 500)

# further improvements:
# - different svm kernel/params
# - attempt simple bag-of-words & simple bag-of-ngrams both with tf-idf

# next step: tf-idf + svm + pos
