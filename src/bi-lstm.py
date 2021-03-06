import math
import numpy
import os
import sys

from datetime import datetime
from preProcessData import getPreProcessData
from sklearn import metrics
from tensorflow.keras.layers import Bidirectional, Embedding, Dense, LSTM
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Global Variable Definition
tokenizer = Tokenizer(filters='', lower=False)


def formatData(text, labels):

    lengthOfLongestWord = math.ceil(sum([len(tweet.split(" ")) for tweet in text]) / len(text))

    # Tokenize tweets
    tokenizer.fit_on_texts(text)
    text = tokenizer.texts_to_sequences(text)

    # Convert tweets into numpy array (This requires padding since sentences are not the same length)
    text = pad_sequences(text, maxlen=lengthOfLongestWord, padding="post")
    temp = []

    # Convert labels into a numpy array
    for label in labels:
        # Change for es
        current = numpy.zeros(numberOfDifferentLabels)
        current[int(label)] = 1
        temp.append(current)
    labels = numpy.asarray(temp)

    return text, labels, lengthOfLongestWord


if __name__ == "__main__":
    # sys.argv[1] --> language (Default = us)
    # sys.argv[2] --> simplePreProcessing (Default = False)
    # sys.argv[3] --> ForcePreProcessing (Default = False)
    # sys.argv[4] --> createNewModel (Default = False)
    # sys.argv[5] -->numberOfEpochs (Default = 1)

    # Handle command line inputs

    languageAbbreviation = "us"
    simple = False
    forcePreProcessing = False
    createNewModel = False
    numberOfEpochs = 1

    if len(sys.argv) >= 2:
        temp = sys.argv[1].lower()
        if temp == "us" or temp == "english":
            languageAbbreviation = "us"
        elif temp == "es" or temp == "spanish":
            languageAbbreviation = "es"
        else:
            print("Parser error: argv[1] needs to be either us or es")
            print("Using default value (us)")

        if len(sys.argv) >= 3:
            temp = sys.argv[2].lower()
            if temp == "false":
                simple = False
            elif temp == "true":
                simple = True
            else:
                print("Parser error: argv[2] needs to be either True or False")
                print("Using default value (False)")

            if len(sys.argv) >= 4:
                temp = sys.argv[3].lower()
                if temp == "false":
                    forcePreProcessing = False
                elif temp == "true":
                    forcePreProcessing = True
                else:
                    print("Parser error: argv[3] needs to be either True or False")
                    print("Using default value (False)")

                if len(sys.argv) >= 5:
                    temp = sys.argv[4].lower()
                    if temp == "false":
                        save_model = False
                    elif temp == "true":
                        save_model = True
                    else:
                        print("Parser error: argv[4] needs to be either True or False")
                        print("Using default value (False)")

                    if len(sys.argv) >= 6:
                        temp = sys.argv[4]
                        if not temp.isnumeric():
                            print("Parser error: argv[5] needs to be a whole positive number")
                            print("Using default value (1)")
                        else:
                            numberOfEpochs = temp

    # Variable Declaration
    trainText, trainLabels, validationText, validationLabels, testText, testLabels = getPreProcessData(languageAbbreviation, forcePreProcessing, simple)

    trainText = trainText[:round((1 / 100) * len(trainText))]
    trainLabels = trainLabels[:round((1 / 100) * len(trainLabels))]
    validationText = validationText[:round((1 / 100) * len(validationText))]
    validationLabels = validationLabels[:round((1 / 100) * len(validationLabels))]
    testText = testText[:round((1 / 100) * len(testText))]
    testLabels = testLabels[:round((1 / 100) * len(testLabels))]

    if languageAbbreviation == "us":
        numberOfDifferentLabels = 20
    else:
        numberOfDifferentLabels = 19

    if simple:
        modelPath = "../models/bi-lstm - " + languageAbbreviation + " - Simple PreProcessing"
    else:
        modelPath = "../models/bi-lstm - " + languageAbbreviation + " - Advanced PreProcessing"

    trainText.extend(validationText)
    trainLabels.extend(validationLabels)
    splitRatio = len(validationLabels) / len(trainLabels)

    del validationText, validationLabels

    trainText, trainLabels, lengthOfLongestWord = formatData(trainText, trainLabels)
    testText, testLabels, _ = formatData(testText, testLabels)

    if not os.path.isdir(modelPath) or createNewModel:

        print("Creating bi-lstm model, this is going to be a while")

        # Create weight matrix

        # First load GloVe vectors
        glove = open("../dataset/gloveVectors.txt", 'r', encoding="utf-8")
        embeddings = {}
        for line in glove:
            values = line.split(' ')
            embeddings[values[0]] = numpy.asarray([float(val) for val in values[1:]])
        glove.close()

        vocab = tokenizer.word_index
        weightMatrix = numpy.zeros((len(vocab) + 1, 200))

        for i, (myWord, myId) in enumerate(vocab.items()):
            if myWord in embeddings:
                weightMatrix[myId] = embeddings[myWord]

        # Create Bi-LSTM model
        model = Sequential()
        model.add(Embedding(len(vocab) + 1, 200, mask_zero=True, input_length=lengthOfLongestWord, trainable=True, weights=[weightMatrix]))
        model.add(Bidirectional(LSTM(128, dropout=0.2, return_sequences=True)))
        model.add(Bidirectional(LSTM(128, dropout=0.2)))
        model.add(Dense(numberOfDifferentLabels, "softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", Precision(), Recall()])

        # Fit model with data
        model.fit(trainText, trainLabels, epochs=numberOfEpochs, validation_split=splitRatio)
        model.save(modelPath)

    else:
        model = load_model(modelPath)

    # Evaluate model
    predictions = model.predict(testText)
    predictions = numpy.around(predictions)
    results = metrics.classification_report(testLabels, predictions)
    print(results)
    acc = str(metrics.accuracy_score(testLabels, predictions))
    print("accuracy = " + acc)
    with open("../results/bi-lstm " + str(numberOfEpochs) + " " + datetime.now().strftime("%d-%m-%Y %H-%M-%S") + ".txt", "w+") as f:
        f.write(results + "\n" + "accuracy = " + acc)

