import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from spacy.lang.en import English
from spacy.lang.es import Spanish
from datetime import datetime
from preProcessData import getPreProcessData



# returns tweets as tokens having a POS tag
# observed from https://github.com/anbasile/emojiprediction/blob/master/nb_and_svm/notebooks/
def spacy_tokenize(tweet):
    tokens = parser(tweet)
    tokens = [token.text + '_' + token.pos_ for token in tokens]
    return tokens


# fits/loads SVM model, and converts all data such that it uses TF-IDF, POS tags and bi-grams
def svm_tfidf_pos_bigram(save, train_tweets, test_tweets, train_emoji_labels, test_emoji_labels):
    # used to shrink dataset to ensure code works
    # train_tweets = train_tweets[:round((1/40)*len(train_tweets))]
    # train_emoji_labels = train_emoji_labels[:round((1 / 40) * len(train_emoji_labels))]

    print("\nFitting TF-IDF Vectorizer and transforming data...")
    # converts training and test data accordingly
    vectorizer = TfidfVectorizer(tokenizer=spacy_tokenize, ngram_range=(1, 2))
    vectorizer.fit(train_tweets)
    train_x_tfidf = vectorizer.transform(train_tweets)
    test_x_tfidf = vectorizer.transform(test_tweets)

    if save:
        print("\nCreating and fitting SVM classifier...")
        # svc = svm.SVC(C=1.0, kernel='linear', degree=3)
        svc = svm.LinearSVC()
        svc.fit(train_x_tfidf, train_emoji_labels)
        pickle.dump(svc, open('../models/svc.sav', 'wb'))
    else:
        print("\nLoading existing SVM classifier...")
        svc = pickle.load(open('../models/svc.sav', 'rb'))

    # predict the labels for the test set
    pred_emoji_labels = svc.predict(test_x_tfidf)

    acc = accuracy_score(test_emoji_labels, pred_emoji_labels)
    results = classification_report(test_emoji_labels, pred_emoji_labels)
    print("\nAccuracy Score:", acc)
    print('\nConfusion Matrix:\n', confusion_matrix(test_emoji_labels, pred_emoji_labels))
    print('\nClassification Report:\n', results)

    with open("../results/svm " + datetime.now().strftime("%d-%m-%Y %H-%M-%S") + ".txt", "w+") as f:
        f.write(results + "\n" + "accuracy = " + str(acc))


if __name__ == '__main__':
    # sys.argv[1] --> language
    # sys.argv[2] --> simplePreProcessing
    # sys.argv[3] --> ForcePreProcessing
    # sys.argv[4] --> createNewModel

    languageAbbreviation = "us"
    simple = False
    forcePreProcessing = False
    save_model = False

    if len(sys.argv) >= 2:
        languageAbbreviation = sys.argv[1].lower()
        if languageAbbreviation == "us" or languageAbbreviation == "english":
            languageAbbreviation = "us"
        elif languageAbbreviation == "es" or languageAbbreviation == "spanish":
            languageAbbreviation = "es"
        else:
            print("Parser error: argv[1] needs to be either us or es")
            print("Using default value (us)")

        if len(sys.argv) >= 3:
            simple = sys.argv[2].lower()
            if simple == "false":
                simple = False
            elif simple == "true":
                simple = True
            else:
                print("Parser error: argv[2] needs to be either True or False")
                print("Using default value (False)")

            if len(sys.argv) >= 4:
                forcePreProcessing = sys.argv[3].lower()
                if forcePreProcessing == "false":
                    forcePreProcessing = False
                elif forcePreProcessing == "true":
                    forcePreProcessing = True
                else:
                    print("Parser error: argv[3] needs to be either True or False")
                    print("Using default value (False)")

                if len(sys.argv) >= 5:
                    save_model = sys.argv[4].lower()
                    if save_model == "false":
                        save_model = False
                    elif save_model == "true":
                        save_model = True
                    else:
                        print("Parser error: argv[4] needs to be either True or False")
                        print("Using default value (False)")

    if languageAbbreviation == "us":
        parser = English()
    else:
        parser = Spanish()

    train_tweets, train_labels, valid_tweets, valid_labels, test_tweets, test_labels = getPreProcessData(languageAbbreviation, forcePreProcessing, simple)

    train_tweets.extend(valid_tweets)
    train_labels.extend(valid_labels)

    svm_tfidf_pos_bigram(save_model, train_tweets, test_tweets, train_labels, test_labels)
