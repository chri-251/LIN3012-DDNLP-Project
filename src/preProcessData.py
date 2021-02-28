import enchant
import os
import re
import string
from language_detector import detect_language
from nltk.corpus import stopwords

# Global Variable Declaration
global dictionary
global stopWords
global languageAbbreviation


def isWord(word):
    if languageAbbreviation == "us":
        if dictionary.check(word):
            return True
    else:
        if detect_language(word) == "Spanish":
            return True
    return False


def generateDicts():
    global dictionary
    global stopWords

    if languageAbbreviation == "us":
        dictionary = enchant.Dict("en_US")
        stopWords = stopwords.words("english")
    else:
        stopWords = stopwords.words("spanish")


def getCharacterClass(character):
    if character.isdigit():
        return "digit"
    elif character.isalpha():
        return "letter"
    else:
        return "other"


def preProcessWord(word, handled=False):
    # Remove empty strings and stopwords
    if len(word) >= 2 and word.lower() not in stopWords:

        # URL Token Check
        if word.startswith("http") or word.startswith("www"):
            for i in range(len(word)):
                if not (word[i].isdigit() or word[i].isalpha() or word[i] in " \<>{}_"):
                    word = preProcessWord(word[i:])
                    if word != 0:
                        return "<url> " + word
                    return "<url>"
            return "<url>"

        # User Token Check
        if word[0] == '@':
            if word == '@':
                return word
            return "<user>"

        # Number token check
        if word.replace('.', '', 1).isdigit():
            return "<number>"

        # Hashtag token check
        if word[0] == '#':
            # CamelCase check
            word = word[1:]
            words = re.findall(r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))", word)
            if len(words) > 1:
                temp = "<hashtag> "
                for word in words:
                    processedWord = preProcessWord(word)
                    if processedWord != 0:
                        temp += processedWord + " "
                if temp[0] == '#':
                    return temp[:-1]
                else:
                    return temp

            # Underscore check
            words = word.split('_')
            if len(words) > 1:
                temp = "<hashtag> "
                for word in words:
                    processedWord = preProcessWord(word)
                    if processedWord != 0:
                        temp += processedWord + " "
                if temp[0] == '#':
                    return temp[:-1]
                else:
                    return temp
            processedWord = preProcessWord(word)
            if processedWord == 0:
                return "<hashtag>"
            return "<hashtag> " + processedWord

        # All Caps token check
        # N.B: Second check needs to be added due to certain unicode characters
        if word.isupper() and word.upper() != word.lower():
            processedWord = preProcessWord(word.lower())
            if processedWord == 0:
                return "<allcaps>"
            return "<allcaps> " + processedWord

        # Handle Punctuation and numbers
        current = getCharacterClass(word[0])
        words = [""]
        for i, c in enumerate(word):
            currentClass = getCharacterClass(c)
            if current == currentClass:
                words[-1] += c
            else:
                words.append(c)
            current = currentClass

        if len(words) != 1 and not handled:
            toBeReturned = ""
            for word in words:
                word = preProcessWord(word, True)
                if word != 0:
                    toBeReturned += word + " "
            return toBeReturned

        if any(p in word for p in string.punctuation):

            # Smile Token Check
            wordHolder = word
            for face in [":‑)", ":)", ":-]", ":]", ":-3", ":3", ":->", ":>", "8-)", "8)", ":-}", ":}", ":o)", ":c)",
                         ":^)", "=]", "=)", ":-))", ":'‑)", ":')", "^_^", "(°o°)", "(^_^)/", "(^O^)/", "(^O^)/",
                         "(^o^)/", "(^^)/", "(≧∇≦)/", "(/◕ヮ◕)/", "(^o^)丿", "∩(·ω·)∩", "(·ω·)", "^ω^", "\(~o~)/",
                         "\(^o^)/", "\(-o-)/", "ヽ(^。^)ノ", "ヽ(^o^)丿", "(*^0^*)", "(●＾o＾●)", "(＾ｖ＾)", "(＾ｕ＾)",
                         "(＾◇＾)", "( ^)o(^ )", "(^O^)", "(^o^)", "(^○^)", ")^o^(", "(*^▽^*)", "(✿◠‿◠)", "( ﾟヮﾟ)",
                         "ヽ(´▽`)/", "^ㅂ^"]:
                wordHolder = wordHolder.replace(face, "")
                if wordHolder == "":
                    return "<smile>"

            # LolFace Token Check
            wordHolder = word
            for face in [":‑D", ":D", "8‑D", "8D", "x‑D", "xD", "X‑D", "XD", "=D", "=3", "B^D", "c:", "C:", ">^_^<",
                         "<^!^>", "^/^", "(*^_^*)", "§^.^§", "(^<^)", "(^.^)", "(^ム^)", "(^·^)", "(^.^)", "(^_^.)",
                         "(^_^)", "(^^)", "(^J^)", "(*^.^*)", "^_^", "(#^.^#)", "(^—^)", "(*^^)", "(^^)", "(^_^)",
                         "(’-’*)", "(^v^)", "(^▽^)", "(・∀・)", "(´∀`)", "(⌒▽⌒)"]:
                wordHolder = wordHolder.replace(face, "")
                if wordHolder == "":
                    return "<lolface>"

            # SadFace Token Check
            wordHolder = word
            for face in [":‑(", ":(", ":‑c", ":c", ":‑<", ":<", ":‑[", ":[", ":{", ";(", "D‑':", ":'‑(", ":'(", "('_')",
                         "(/_;)", "(T_T)", "(;_;)", "(;_;", "(;_:)", "(;O;)", "(:_;)", "(ToT)", "(T▽T)", ";_;", ";-;",
                         ";n;", "(._.)", "(´；ω；`)", "( つ Д `)"]:
                wordHolder = wordHolder.replace(face, "")
                if wordHolder == "":
                    return "<sadface>"

            # NeutralFace Token Check
            wordHolder = word
            for face in [":‑|", ":|", "(-_-)", "(`_ゝ`)"]:
                wordHolder = wordHolder.replace(face, "")
                if wordHolder == "":
                    return "<neutralface>"

            # Heart token check
            if word.replace("<3", "") == "":
                return "<heart>"

            # Repeat token check
            if len(word) != 1 and word[-1] == word[-2]:
                punctuation = word[-1]
                reversedWord = word[::-1]
                temp = ""
                for i in range(len(reversedWord)):
                    if reversedWord[i] == punctuation:
                        temp = temp[:i] + temp[i + 1:]
                    else:
                        break
                return temp[::-1].lower() + " " + punctuation + " <repeat>"

        # Elong token check
        if len(word) > 2 and word[-1] == word[-2] and not isWord(word):
            repeatedCharacter = word[-1]
            reversedWord = word[::-1]
            for i in range(len(reversedWord)):
                if reversedWord[i] != repeatedCharacter:
                    word = reversedWord[i:][::-1] + repeatedCharacter
                    if isWord(word + repeatedCharacter):
                        processedWord = preProcessWord(word + repeatedCharacter)
                        if processedWord != 0:
                            return processedWord + " <elong>"
                    else:
                        processedWord = preProcessWord(word)
                        if processedWord != 0:
                            return processedWord + " <elong>"
        return word.lower()
    return 0


def getData(name, preProcessedPath, rawPath, labelPath, ForcePreProcess):

    # Get labels
    labelsFile = open(labelPath)
    labels = labelsFile.read().split("\n")[:-1]

    if not ForcePreProcess and os.path.exists(preProcessedPath):
        # Get Tweets
        textFile = open(preProcessedPath, encoding="utf-8")
        tweets = textFile.read().split("\n")[:-1]

        if len(tweets) != len(labels):
            print("Error occurred while getting " + name + " data: Number of tweets not equal to number of number of labels")
            exit(-2)

        textFile.close()
        labelsFile.close()
    else:
        print("---------------------")
        print("Pre-processing " + name + " data")
        print("---------------------")

        tweets = []
        textFile = open(rawPath, encoding="utf-8")
        text = textFile.read().split('\n')[:-1]

        if len(text) != len(labels):
            print("Error occurred while getting " + name + " data: Number of tweets not equal to number of number of labels")
            exit(-2)

        for i, tweet in enumerate(text):
            punctuationLength = 0
            punctuation = ""
            newTweet = ""
            current = tweet.split(" ")
            for j, word in enumerate(current):
                isPunctuation = True
                if word == '':
                    isPunctuation = False
                else:
                    for c in word:
                        if c not in string.punctuation:
                            isPunctuation = False
                            break
                if isPunctuation:
                    punctuation += word
                    punctuationLength += 1
                elif len(punctuation) != 0:
                    newTweet = ' '.join(current[:j - punctuationLength]) + " " + punctuation + " " + ' '.join(
                        current[j:])
                    punctuation = ""
            if newTweet != "":
                tweet = newTweet

            temp = ""
            for word in tweet.split(" "):
                processed = preProcessWord(word)
                if processed != 0:
                    temp += processed + " "
            temp = re.sub(' +', ' ', temp).strip()
            # Handle case when all words in tweet are stopwords
            if len(temp) == 0:
                tweets.append(" ")
            else:
                tweets.append(temp)
            print("Pre-processing " + name + " data: " + str(round((i / len(text)) * 100)) + "% complete")
        with open(preProcessedPath, "w+", encoding="utf-8") as f:
            for tweet in tweets:
                f.write(tweet + '\n')

    return tweets, labels


def simple_preprocessing(tweet_filename, labels_filename):
    set_tweets = []

    set_txt = open(tweet_filename, 'r', encoding='utf-8')
    set_labels_file = open(labels_filename, 'r', encoding='utf-8')

    # create list with labels
    set_labels = set_labels_file.read().split("\n")
    # if the last saved label is empty (so, text file ends with a new line), remove it from the list of labels
    if set_labels[len(set_labels) - 1] == '':
        set_labels = set_labels[:-1]

    # create list with pre-processed tweets
    for line in set_txt:
        prep_tweet = ''
        lowercase_line = line.lower()
        lowercase_line = lowercase_line.rstrip("\n")

        lowercase_no_punc = lowercase_line.translate(str.maketrans('', '', string.punctuation))
        # removes # and @ as well

        for word in lowercase_no_punc.split():
            if len(word) >= 2 and word not in stopWords:
                prep_tweet += word + " "

        prep_tweet = prep_tweet.strip()  # removes extra spaces at the front or back of prep_tweet

        set_tweets.append(prep_tweet)

    set_txt.close()
    set_labels_file.close()

    return set_tweets, set_labels


def getPreProcessData(language, ForcePreProcess=False, SimplePreProcessing=False):
    global languageAbbreviation
    languageAbbreviation = language

    generateDicts()

    os.chdir("../dataset/" + languageAbbreviation)

    if SimplePreProcessing:
        trainData, trainLabels = simple_preprocessing('raw data/train/' + languageAbbreviation + '_train.text',
                                                      'raw data/train/' + languageAbbreviation + '_train.labels')
        validData, validLabels = simple_preprocessing('raw data/valid/' + languageAbbreviation + '_valid.text',
                                                      'raw data/valid/' + languageAbbreviation + '_valid.labels')
        testData, testLabels = simple_preprocessing('raw data/test/' + languageAbbreviation + '_test.text',
                                                    'raw data/test/' + languageAbbreviation + '_test.labels')
    else:
        trainData, trainLabels = getData("train", "pre-processed data/train/text.txt", "raw data/train/" + languageAbbreviation +"_train.TEXT", "raw data/train/" + languageAbbreviation +"_train.LABELS", ForcePreProcess)
        validData, validLabels = getData("validation", "pre-processed data/valid/text.txt", "raw data/valid/" + languageAbbreviation +"_valid.TEXT", "raw data/valid/" + languageAbbreviation +"_valid.LABELS", ForcePreProcess)
        testData, testLabels = getData("test", "pre-processed data/test/text.txt", "raw data/test/" + languageAbbreviation +"_test.TEXT", "raw data/test/" + languageAbbreviation +"_test.LABELS", ForcePreProcess)

    os.chdir("../../src")

    return trainData, trainLabels, validData, validLabels, testData, testLabels
