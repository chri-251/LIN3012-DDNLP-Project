import json


def preprocessData(languageAbbreviation, path, savePath):
    # Variable Declaration
    text = []
    labels = []
    noLabelsCount = 0
    if languageAbbreviation == "us":
        dictOfChars = {'❤': 0, '😍': 1, '😂': 2, '💕': 3, '🔥': 4, '😊': 5, '😎': 6, '✨': 7, '💙': 8, '😘': 9, '📷': 10, '🇺🇸': 11, '☀': 12, '💜': 13, '😉': 14, '💯': 15, '😁': 16, '🎄': 17, '📸': 18, '😜': 19}
    else:
        dictOfChars = {'❤': 0, '😍': 1, '😂': 2, '💕': 3, '😊': 4, '😘': 5, '💪': 6, '😉': 7, '👌': 8, '🇪🇸': 9, '😎': 10,
                       '💙': 11, '💜': 12, '😜': 13, '💞': 14, '✨': 15, '🎶': 16, '💘': 17, '😁': 18}

    with open(path, encoding="utf-8") as file:
        tweets = file.read().split("\n")
        for i in range(len(tweets)):
            currentText = ''
            currentLabel = ''
            for character in json.loads(tweets[i])["text"]:
                if character in ['❤', '😍', '😂', '💕', '🔥', '😊', '😎', '✨', '💙', '😘', '📷', '🇺🇸', '☀', '💜',
                                 '😉', '💯', '😁', '🎄', '📸', '😜']:
                    if currentLabel == '':
                        currentLabel = character
                        labels.append(dictOfChars[character])
                    if currentLabel != character:
                        print("Error: Different labels detected in tweet #" + str(i))
                        exit(-1)
                else:
                    # \n check
                    if character == '\n':
                        currentText += " "
                    else:
                        currentText += character
            if currentLabel == '':
                noLabelsCount += 1
            else:
                text.append(currentText)

    # Clean data
    for i in range(len(text)):
        # Remove any spaces from the start of string
        text[i].strip()

        # Remove last link
        if text[i].split()[-1].startswith("https://t.co/"):
            text[i] = text[i].rsplit(' ', 1)[0]

    if len(text) != len(labels):
        print("Error: Number of tweets not equal to number of number of labels")
        exit(-2)

    print(str(noLabelsCount) + " tweets found without a label")
    print(str(len(labels)) + " tweets found with a label")

    with open(savePath + ".TEXT", 'w', encoding="utf-8") as f:
        for i in text:
            f.write("%s\n" % i)
    print("TEXT file successfully generated")

    with open(savePath + ".LABELS", 'w') as f:
        for label in labels:
            f.write("%s\n" % label)
    print("LABELS file successfully generated")


if __name__ == "__main__":
    print("-------- Pre-processing us train data --------")
    preprocessData("us", "dataset/us/train/trainOriginal.txt", "dataset/us/train/us_train")
    print("-------- Pre-processing es train data --------")
    preprocessData("es", "dataset/es/train/trainOriginal.txt", "dataset/es/train/es_train")
