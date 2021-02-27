import math
import numpy
import os
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

    # Variable Declaration
    languageAbbreviation = "us"
    trainText, trainLabels, validationText, validationLabels, testText, testLabels = getPreProcessData(languageAbbreviation, False, True)
    numberOfEpochs = 2
    continueTraining = True
    if languageAbbreviation == "us":
        numberOfDifferentLabels = 20
    else:
        numberOfDifferentLabels = 19

    trainText.extend(validationText)
    trainLabels.extend(validationLabels)
    splitRatio = len(validationLabels) / len(trainLabels)

    del validationText, validationLabels

    print("Resuming bi-lstm model training, this is going to be a while")

    trainText, trainLabels, lengthOfLongestWord = formatData(trainText, trainLabels)
    testText, testLabels, _ = formatData(testText, testLabels)
    model = load_model("../models/bi-lstm - " + languageAbbreviation)

    while True:

        # continueTraining = False

        # trainText = trainText[:round((1/10) * len(trainText))]
        # trainLabels = trainLabels[:round((1 / 10) * len(trainLabels))]
        # validationText = validationText[:round((1 / 10) * len(validationText))]
        # validationLabels = validationLabels[:round((1 / 10) * len(validationLabels))]
        # testText = testText[:round((1 / 10) * len(testText))]
        # testLabels = testLabels[:round((1 / 10) * len(testLabels))]
        if not os.path.isdir("../models/bi-lstm - " + languageAbbreviation):

            trainText.extend(validationText)
            trainLabels.extend(validationLabels)
            splitRatio = len(validationLabels) / len(trainLabels)

            del validationText, validationLabels

            print("Creating bi-lstm model, this is going to be a while")

            trainText, trainLabels, lengthOfLongestWord = formatData(trainText, trainLabels)
            testText, testLabels, _ = formatData(testText, testLabels)

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
            model.save("../models/bi-lstm - " + languageAbbreviation)

        elif continueTraining:

            model.fit(trainText, trainLabels, epochs=numberOfEpochs, validation_split=splitRatio, initial_epoch=numberOfEpochs-1)
            model.save("../models/bi-lstm - " + languageAbbreviation)
            model.save("../models/bi-lstm - " + languageAbbreviation + " (" + str(numberOfEpochs) + ")")

        else:
            testText, testLabels, _ = formatData(testText, testLabels)
            model = load_model("../models/bi-lstm - " + languageAbbreviation)

        # Evaluate model
        predictions = model.predict(testText)
        predictions = numpy.around(predictions)
        results = metrics.classification_report(testLabels, predictions)
        print(results)
        acc = str(metrics.accuracy_score(testLabels, predictions))
        print("accuracy = " + acc)
        with open("../results/bi-lstm " + str(numberOfEpochs) + " " + datetime.now().strftime("%d-%m-%Y %H-%M-%S") + ".txt", "w+") as f:
            f.write(results + "\n" + "accuracy = " + acc)
        numberOfEpochs += 1

