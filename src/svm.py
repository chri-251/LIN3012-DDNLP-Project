from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import winsound

# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from spacy.lang.en import English
from datetime import datetime


def read_labels(labels_filename):
    set_labels_file = open(labels_filename, 'r', encoding='utf-8')
    set_labels = set_labels_file.read().split("\n")
    set_labels_file.close()

    if set_labels[len(set_labels) - 1] == '':
        set_labels = set_labels[:-1]

    return set_labels


def simple_preprocessing(tweet_filename):
    set_tweets = []

    set_txt = open(tweet_filename, 'r', encoding='utf-8')

    for line in set_txt:
        prep_tweet = ''
        lowercase_line = line.lower()
        lowercase_line = lowercase_line.rstrip("\n")

        # print(line,)  # The comma to suppress the extra new line char
        lowercase_no_punc = lowercase_line.translate(str.maketrans('', '', string.punctuation))
        # removes # and @ as well

        for word in lowercase_no_punc.split():
            if len(word) >= 2 and word not in stopWords:
                prep_tweet += word + " "

        prep_tweet = prep_tweet.strip()  # removes extra spaces at the front or back of prep_tweet

        set_tweets.append(prep_tweet)

    set_txt.close()

    return set_tweets


def get_preprocessed_tweets(tweets_filename):
    set_tweets = []
    set_tweets_file = open(tweets_filename, 'r', encoding='utf-8')

    for line in set_tweets_file:
        set_tweets.append(line)

    set_tweets_file.close()

    return set_tweets


def spacy_tokenize(tweet):
    tokens = parser(tweet)
    tokens = [token.text + '_' + token.pos_ for token in tokens]
    return tokens


def svm_tfidf_pos_bigram(save, train_tweets, test_tweets, train_emoji_labels, test_emoji_labels):
    # train_tweets = train_tweets[:round((1/40)*len(train_tweets))]
    # train_emoji_labels = train_emoji_labels[:round((1 / 40) * len(train_emoji_labels))]

    print("\nFitting TF-IDF Vectorizer and transforming data...")
    # converts training and test data to TFIDF
    vectorizer = TfidfVectorizer(tokenizer=spacy_tokenize, ngram_range=(1, 2))
    vectorizer.fit(train_tweets)
    train_x_tfidf = vectorizer.transform(train_tweets)
    test_x_tfidf = vectorizer.transform(test_tweets)
    # print(vectorizer.vocabulary_)

    if save:
        print("\nCreating and fitting SVM classifier...")
        # svc = svm.SVC(C=1.0, kernel='linear', degree=3)
        svc = svm.LinearSVC()
        svc.fit(train_x_tfidf, train_emoji_labels)
        pickle.dump(svc, open('../models/svc_preprocessed_us.sav', 'wb'))
    else:
        print("\nLoading existing SVM classifier...")
        svc = pickle.load(open('../models/svc.sav', 'rb'))

    # predict the labels for test set
    pred_emoji_labels = svc.predict(test_x_tfidf)

    acc = accuracy_score(test_emoji_labels, pred_emoji_labels)
    results = classification_report(test_emoji_labels, pred_emoji_labels)

    print("\nAccuracy Score:", acc)
    print('\nConfusion Matrix:\n', confusion_matrix(test_emoji_labels, pred_emoji_labels))
    print('\nClassification Report:\n', results)

    with open("../results/svm " + datetime.now().strftime("%d-%m-%Y %H-%M-%S") + ".txt", "w+") as f:
        f.write(results + "\n" + "accuracy = " + str(acc))


if __name__ == '__main__':
    # nltk.download()

    parser = English()
    stopWords = stopwords.words("english")

    simple = False

    if simple:
        print("\nPerforming simple pre-processing...")
        train_tweets = simple_preprocessing('../dataset/us/raw data/train/us_train.text')
        train_tweets.extend(simple_preprocessing('../dataset/us/raw data/valid/us_valid.text'))
        test_tweets = simple_preprocessing('../dataset/us/raw data/test/us_test.text')
    else:
        print("\nReading data pre-processed with token swapping...")
        train_tweets = get_preprocessed_tweets('../dataset/us/pre-processed data/train/text.txt')
        train_tweets.extend(get_preprocessed_tweets('../dataset/us/pre-processed data/valid/text.txt'))
        test_tweets = get_preprocessed_tweets('../dataset/us/pre-processed data/test/text.txt')

    train_emoji_labels = read_labels('../dataset/us/raw data/train/us_train.labels')
    train_emoji_labels.extend(read_labels('../dataset/us/raw data/valid/us_valid.labels'))
    test_emoji_labels = read_labels('../dataset/us/raw data/test/us_test.labels')

    svm_tfidf_pos_bigram(True, train_tweets, test_tweets, train_emoji_labels, test_emoji_labels)

    winsound.Beep(1500, 500)

# definitely: change code to suit US and ES
