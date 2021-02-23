import math
import numpy
from preProcessData import getPreProcessData
from tensorflow.keras.layers import Bidirectional, Embedding, Dense, LSTM, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


if __name__ == "__main__":
    # Variable Declaration
    trainText, trainLabels, validationText, validationLabels, testText, testLabels = getPreProcessData("us")
    trainText.extend(validationText)
    trainLabels.extend(validationLabels)
    splitRatio = len(validationLabels) / len(trainLabels)

    del validationText, validationLabels

    lengthOfLongestWord = math.ceil(sum([len(tweet.split(" ")) for tweet in trainText]) / len(trainText))

    # Tokenize tweets
    print("Tokenize tweets")
    tokenizer = Tokenizer(filters='', lower=False)
    tokenizer.fit_on_texts(trainText)
    tweets = tokenizer.texts_to_sequences(trainText)

    # Convert tweets into numpy array (This requires padding since sentences are not the same length)
    print("Convert tokens into numpy array")
    tweets = pad_sequences(tweets, maxlen=lengthOfLongestWord, padding="post")
    temp = []

    # Convert labels into a numpy array
    print("Convert labels into a numpy array")
    for label in trainLabels:
        # Change for es
        current = numpy.zeros(20)
        current[int(label)] = 1
        temp.append(current)
    labels = numpy.asarray(temp)

    # Create weight matrix

    # First load GloVe vectors
    print("First load GloVe vectors")
    glove = open("../dataset/gloveVectors.txt", 'r', encoding="utf-8")
    embeddings = {}
    for line in glove:
        values = line.split(' ')
        embeddings[values[0]] = numpy.asarray([float(val) for val in values[1:]])
    glove.close()

    print("Create weight matrix")
    vocab = tokenizer.word_index
    weightMatrix = numpy.zeros((len(vocab) + 1, 300))

    for i, (myWord, myId) in enumerate(vocab.items()):
        if myWord in embeddings:
            weightMatrix[myId] = embeddings[myWord]
        print(str(math.floor((i / len(vocab.items())) * 100)) + "% complete")

    # Create LSTM model
    print("Create LSTM model")
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, 300, mask_zero=True, input_length=lengthOfLongestWord, trainable=True,
                        weights=[weightMatrix]))
    model.add(Bidirectional(LSTM(128, dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, dropout=0.2)))
    model.add(Dense(20, "softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Fit model with data
    model.fit(tweets, labels, epochs=1, validation_split=splitRatio)
