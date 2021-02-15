import math
import numpy
import re
from Naked.toolshed.shell import muterun_rb
from tensorflow.keras.layers import Bidirectional, Embedding, Dense, LSTM, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords

def asciiEmojiChecker(listOfEmojis)


def getData(textPath, labelPath):
    stopWords = stopwords.words("english")
    TextFile = open(textPath, 'r', encoding="utf-8")
    LabelsFile = open(labelPath, 'r')
    text = TextFile.read().split("\n")[:-1]
    labels = LabelsFile.read().split("\n")[:-1]
    tweets = []

    TextFile.close()
    LabelsFile.close()

    # Different regex parts
    eyes = "[8:=;]"
    nose = "['`\-]?"
    smilePattern = re.compile("/#[8:=;]#['`\-]?[)d]+|[)d]+#['`\-]?#[8:=;]/i,")

    if len(text) != len(labels):
        print("Error: Number of tweets not equal to number of number of labels")
        exit(-2)

    for tweet in text:
        temp = ""
        for word in tweet.split(" "):
            #Remove empty strings and stopwords
            if len(word) != 0 and word not in stopWords:
                # URL Token Check
                if word.startswith("http"):
                    entered = False
                    for i in range(len(word)):
                        if not (word[i].isdigit() or word[i].isalpha() or word[i] in "-._~:/?#[]@!$&'()*+,;="):
                            entered = True
                            break
                    temp += "<URL> "
                    if entered:
                        word = word[i:]
                    else:
                        word = ""
                # User Token Check
                elif word[0] == '@':
                    temp += "<USER> "
                    continue
                nowContinue = False
                # Smile Token Check
                wordHolder = word
                for face in [":‑)", ":)", ":-]", ":]", ":-3", ":3", ":->", ":>", "8-)", "8)", ":-}", ":}", ":o)", ":c)", ":^)", "=]", "=)", ":-))", ":'‑)", ":')", "^_^", "(°o°)", "(^_^)/", "(^O^)/", "(^O^)/", "(^o^)/", "(^^)/", "(≧∇≦)/", "(/◕ヮ◕)/", "(^o^)丿", "∩(·ω·)∩", "(·ω·)", "^ω^", "\(~o~)/", "\(^o^)/", "\(-o-)/", "ヽ(^。^)ノ", "ヽ(^o^)丿", "(*^0^*)", "(●＾o＾●)", "(＾ｖ＾)", "(＾ｕ＾)", "(＾◇＾)", "( ^)o(^ )", "(^O^)", "(^o^)", "(^○^)", ")^o^(", "(*^▽^*)", "(✿◠‿◠)", "( ﾟヮﾟ)", "ヽ(´▽`)/", "^ㅂ^"]:
                    wordHolder.replace(face, "")
                    if wordHolder == "":
                        temp += "<SMILE>"
                        nowContinue = True
                        break
                if nowContinue:
                    continue
                # LolFace Token Check
                wordHolder = word
                for face in [":‑D", ":D", "8‑D", "8D", "x‑D", "xD", "X‑D", "XD", "=D", "=3", "B^D", "c:", "C:",">^_^<", "<^!^>", "^/^", "(*^_^*)", "§^.^§", "(^<^)", "(^.^)", "(^ム^)", "(^·^)", "(^.^)", "(^_^.)", "(^_^)", "(^^)", "(^J^)", "(*^.^*)", "^_^", "(#^.^#)", "(^—^)", "(*^^)", "(^^)", "(^_^)", "(’-’*)", "(^v^)", "(^▽^)", "(・∀・)", "(´∀`)", "(⌒▽⌒)"]:
                    wordHolder.replace(face, "")
                    if wordHolder == "":
                        temp += "<LOLFACE>"
                        nowContinue = True
                        break
                if nowContinue:
                    continue
                # SadFace Token Check
                wordHolder = word
                for face in [":‑(", ":(", ":‑c", ":c", ":‑<", ":<", ":‑[", ":[", ":{", ";(", "D‑':", ":'‑(", ":'(", "('_')", "(/_;)", "(T_T)", "(;_;)", "(;_;", "(;_:)", "(;O;)", "(:_;)", "(ToT)", "(T▽T)", ";_;", ";-;", ";n;", "(._.)","(´；ω；`)", "( つ Д `)"]:
                    wordHolder.replace(face, "")
                    if wordHolder == "":
                        temp += "<SADFACE>"
                        nowContinue = True
                        break
                if nowContinue:
                    continue
                # NeutralFace Token Check
                wordHolder = word
                for face in [":‑|", ":|", "(-_-)", "(`_ゝ`)"]:
                    wordHolder.replace(face, "")
                    if wordHolder == "":
                        temp += "<NEUTRALFACE>"
                        nowContinue = True
                        break
                if nowContinue:
                    continue
                # Heart token check
                wordHolder = word
                wordHolder.replace("<3", "")
                if wordHolder == "":
                    temp += "<HEART>"
                    continue
                # Number token check
                if word.replace('.','',1).isdigit():
                    temp += "<NUMBER>"
                    continue
                # Hashtag token check
                

        tweets.append(temp)
    lengthOfLongestWord = math.ceil(sum([len(s.split(" ")) for s in tweets]) / len(tweets))
    exit()
    return text, labels, lengthOfLongestWord


if __name__ == "__main__":
    # Variable Declaration
    text, labels, _ = getData("../dataset/us/train/us_train.TEXT", "../dataset/us/train/us_train.LABELS")
    validationText, validationLabels, _ = getData("../dataset/us/valid/us_valid.TEXT",
                                                  "../dataset/us/valid/us_valid.LABELS")
    tweets = []

    text.extend(validationText)
    labels.extend(validationLabels)
    splitRatio = len(validationLabels) / len(labels)
    del validationText, validationLabels

    # Remove unneeded words
    print("Remove unneeded words")

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
    model.fit(tweets, labels, epochs=5, validation_split=splitRatio)
    # score, acc = model.evaluate(xTest, yTest)

    # trainSet = pandas.DataFrame(list(zip(text, labels)))
