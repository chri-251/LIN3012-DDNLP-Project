import sys

if __name__ == "__main__":
    # argv[1] = name of model
    # argv[2] = sentence

    while True:
        if len(sys.argv) < 3:
            while True:
                print("Please pick a model")
                print("1) LSTM")
                print("2) Bi-LSTM")
                print("3) SVM")
                print("0) Exit")
                model = input().lower().strip()
                if model == '1' or model == "lstm":
                    model = 1
                    break
                elif model == '2' or model == "bi-lstm" or model == "bilstm":
                    model = 2
                    break
                elif model == '3' or model == "svm":
                    model = 3
                    break
                else:
                    print("Please enter a valid input")
            sentence = input("Enter your sentence: ")
        else:
            sentence = sys.argv[2]
            if sys.argv[1] == '1' or sys.argv[1] == "lstm":
                model = 1
                break
            elif sys.argv[1] == '2' or sys.argv[1] == "bi-lstm" or sys.argv[1] == "bilstm":
                model = 2
                break
            elif sys.argv[1] == '3' or sys.argv[1] == "svm":
                model = 3
                break
            else:
                print("-----------------")
                print("Parser Error")
                print("-----------------")

    if model == 1:
        print("Doing lstm")
    elif model == 2:
        print("Doing bi-lstm")
    elif model == 3:
        print("Doing svm")

