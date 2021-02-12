import math
import numpy
from tensorflow.keras.layers import Bidirectional, Embedding, Dense, LSTM, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
# from keras.initializers import Constant
# from keras.layers import Bidirectional, Embedding, Dense, LSTM, SpatialDropout1D
# from keras.layers.wrappers import Bidirectional
# from keras.models import Sequential
# from keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords

if __name__ == "__main__":
    # Variable Declaration
    print("Variable Declaration")
    TextFile = open("../dataset/us/train/us_train.TEXT", 'r', encoding="utf-8")
    LabelsFile = open("../dataset/us/train/us_train.LABELS", 'r')
    text = TextFile.read().split("\n")[:-1]
    labels = LabelsFile.read().split("\n")[:-1]
    stopWords = stopwords.words("english")
    tweets = []

    TextFile.close()
    LabelsFile.close()

    if len(text) != len(labels):
        print("Error: Number of tweets not equal to number of number of labels")
        exit(-2)

    # Remove unneeded words
    print("Remove unneeded words")
    for tweet in text:
        temp = ""
        for word in tweet.split(" "):
            if len(word) != 0:
                # Remove Stopwords and @handles
                if word not in stopWords and word[0] != '@':
                    # # Remove #
                    # if word[0] == '#':
                    #     temp += word[1:] + " "
                    # else:
                    temp += word + " "
        tweets.append(temp)
    lengthOfLongestWord = math.ceil(sum([len(s.split(" ")) for s in tweets]) / len(tweets))

    # Tokenize tweets
    print("Tokenize tweets")
    tokenizer = Tokenizer(lower=True, split=" ")
    tokenizer.fit_on_texts(tweets)
    tweets = tokenizer.texts_to_sequences(tweets)

    # Convert tweets into numpy array (This requires padding since sentences are not the same length)
    print("Convert tweets into numpy array")
    tweets = pad_sequences(tweets, maxlen=lengthOfLongestWord, padding="post")
    temp = []

    # Convert labels into a numpy array
    print("Convert labels into a numpy array")
    for label in labels:
        # Change for es
        current = numpy.zeros(20)
        current[int(label)] = 1
        temp.append(current)
    labels = numpy.asarray(temp)

    # Create weight matrix

    # First load GloVe vectors
    print("First load GloVe vectors")
    glove = open("../glove.840B.300d.txt", 'r', encoding="utf-8")
    embeddings = {}
    for line in glove:
        values = line.split(' ')
        embeddings[values[0]] = numpy.asarray([float(val) for val in values[1:]])
    glove.close()

    print("Create weight matrix")
    vocab = tokenizer.word_index
    weightMatrix = numpy.zeros((len(vocab) + 1, 300))

    for i, (word, myId) in enumerate(vocab.items()):
        if word in embeddings:
            weightMatrix[myId] = embeddings[word]
        print(str(math.floor((i / len(vocab.items())) * 100)) + "% complete")

    # Create LSTM model
    print("Create LSTM model")
    # embeddingLayer = Embedding(len(vocab) + 1, 300, weights=[weightMatrix], input_length=lengthOfLongestWord, trainable=True,  mask_zero=True)
    model = Sequential()
    model.add(Embedding(len(vocab) + 1, 300, mask_zero=True, input_length=lengthOfLongestWord, trainable=True, weights=[weightMatrix]))
    model.add(Bidirectional(LSTM(128, dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, dropout=0.2)))
    model.add(Dense(20, "softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # model.fit(x, y, epochs=5, validation_split=0.25)
    # score, acc = model.evaluate(xTest, yTest)

    # trainSet = pandas.DataFrame(list(zip(text, labels)))
