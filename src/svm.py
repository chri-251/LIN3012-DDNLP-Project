# from nltk import ngrams
# import nltk
# from nltk.corpus import stopwords
# import math
# import numpy
# import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# requires further pre-processing: removal of extra spaces, changing mentions to @user, etc.
def read_file(file_name):
    all_tweets = []

    training_txt = open(file_name, 'r', encoding='utf-8')

    for line in training_txt:
        lowercase_line = line.lower()
        lowercase_line = lowercase_line.rstrip("\n")

        # print(line,)  # The comma to suppress the extra new line char
        temp = lowercase_line.translate(str.maketrans('', '', string.punctuation))  # counts # and @ as punctuation
        temp = temp.strip()

        split_tweet = temp.split(' ')
        while "" in split_tweet:
            split_tweet.remove("")

        all_tweets.append(temp)

    training_txt.close()

    return all_tweets


def read_labels():
    train_labels_file = open("../dataset/es/train/es_train.LABELS", 'r')
    train_labels = train_labels_file.read().split("\n")[:-1]
    train_labels_file.close()

    test_labels_file = open("../dataset/es/test/es_test.labels", 'r')
    test_labels = test_labels_file.read().split("\n")[:-1]
    test_labels_file.close()

    return train_labels, test_labels


def get_classification():
    mappings = open("../dataset/es/es_mapping.txt", 'r', encoding="utf-8")

    labels = []

    for line in mappings:
        split_line = line.split('\t')
        labels.append(split_line[1] + " " + split_line[2] + "(" + split_line[0] + ")")

    mappings.close()
    return labels


if __name__ == '__main__':
    # nltk.download()

    test_file = '../dataset/es/test/es_test.text'
    train_file = '../dataset/es/train/es_train.TEXT'

    train_tweets = read_file(train_file)

    # stores all test data tweets
    test_tweets = []
    with open(test_file, 'r', encoding='utf-8') as file:
        for line in file:
            test_tweets.append(line)

    train_emoji_labels, test_emoji_labels = read_labels()

    # converts training and test data to TFIDF
    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_tweets)
    train_x_tfidf = vectorizer.transform(train_tweets)
    test_x_tfidf = vectorizer.transform(test_tweets)
    # print(vectorizer.vocabulary_)

    SVM = svm.SVC(C=1.0, kernel='linear', degree=3)
    SVM.fit(train_x_tfidf, train_emoji_labels)
    # predict the labels on validation dataset
    pred_emoji_labels = SVM.predict(test_x_tfidf)

    emoji_labels = get_classification()

    print("\nAccuracy Score:", accuracy_score(test_emoji_labels, pred_emoji_labels))
    print('\nConfusion Matrix:\n', confusion_matrix(test_emoji_labels, pred_emoji_labels))
    print('\nClassification Report:\n', classification_report(test_emoji_labels, pred_emoji_labels,
                                                              target_names=emoji_labels))

# improvements:
# - different svm kernel/params
# - attempt simple bag-of-words & simple bag-of-ngrams
# - attempt above with tfidf
