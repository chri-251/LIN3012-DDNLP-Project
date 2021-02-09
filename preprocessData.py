import json

if __name__ == "__main__":

    # Variable Declaration
    text = []
    labels = []
    noLabelsCount = 0

    with open("dataset/us/train/tweet_by_ID_08_2_2021__11_57_05.txt", encoding='utf-8') as file:
        tweets = file.read().split("\n")
        for i in range(len(tweets)):
            currentText = ''
            currentLabel = ''
            for character in json.loads(tweets[i])["text"]:
                if character in ['❤', '😍', '😂', '💕', '🔥', '😊', '😎', '✨', '💙', '😘', '📷', '🇺🇸', '☀', '💜', '😉', '💯', '😁', '🎄', '📸', '😜']:
                    if currentLabel == '':
                        currentLabel = character
                        labels.append(character)
                    if currentLabel != character:
                        print("Error: Different labels detected in tweet #" + str(i))
                        exit(-1)
                else:
                    currentText += character
            if currentLabel == '':
                noLabelsCount += 1
            else:
                text.append(currentText)
    print(noLabelsCount)
    print(len(text))
    print(len(labels))

