Data-Driven NLP: Multilingual EMOJI Prediction Multilingual EMOJI prediction
By Christina Barbara (0037900L) and Charlton Sammut (0490200L)

-In the documentation folder one can find the documentation, the plagiarism form and the compiled results generated through our experiments.

-In the dataset folder one can find all the data used by our scripts.
Can be downloaded from: https://drive.google.com/file/d/19XzfaGeZKyd0xFP1e7cn8Z7brU3ERNwq/view?usp=sharing

-In the models folder one can find the 8 created models.
Can be downloaded from: https://drive.google.com/file/d/14Kl0icBFk3IfmuyMMkVEs5ZtoRexNo2P/view?usp=sharing

-In the results folder one can find the automatically generated results created by the models.

-In the source folder one can find all the code written for this Project.

--Before running any scripts the python >= 3.8 needs to be installed with the following packages:
---language-detector >= 5.0.2 (pip install language-detector)
---nltk >= 3.5 (pip install nltk)
---numpy >= 1.19.5(pip install numpy)
---pyenchant >= 3.2.0 (pip install pyenchant)
---scikit-learn >= 0.24.1 (pip install scikit – learn)
---spacy >= 2.3.5 (pip install spacy)
---tensorflow >= 2.4.1 (pip install tensorflow)

--bi-lstm.py --> Is used to build and evaluate our Bi-LSTM model:
---sys.argv[1] --> language
----Either “us” or “es” (Default = us)
---sys.argv[2] --> simplePreProcessing
----Either “True” or “False” (Default = False)
----If True, simple pre-processing will take place
---sys.argv[3] --> ForcePreProcessing 
----Either “True” or “False” (Default = False)
----If True, then all pre-processed data will be removed and pre-processing will re-occur.
---sys.argv[4] --> createNewModel 
----Either “True” or “False” (Default = False)
----If True,  a new model will be trained
---sys.argv[5] -->numberOfEpochs
----Needs to be a positive integer (Default = 1)

---Example bi-lstm.py input: "python bi-lstm.py us False False False 1"

--extractData.py --> Used to extract the training tweets and labels from the meta data.

--preProcessData.py --> Handles the preprocessing of the data

--svm.py --> Is used to build and evaluate our SVM model:
---Takes in the same inputs as bi-lstm.py without the 5th argument (numberOfEpochs)
---Example Input: "python svm.py es False False False"



