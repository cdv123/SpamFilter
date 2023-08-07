from simplifyDataset import loadTrainingData
from simplifyDataset import loadTestData
from collections import Counter
import numpy as np
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
trainingData = loadTestData()
mostCommonWords = []
allTrainingWords = []
spamWordsSet = set()
spamWordsList = []
hamWordsSet = set()
hamWordsList = []
with open("./Dataset/test-data.csv") as file:
    trainingDataCSV = list(csv.reader(file))
    for row in range(len(trainingData)):
        rowWords = trainingData[row].split(" ")
        if trainingDataCSV[row][0] == "spam":
            for i in rowWords:
                if i !='':
                    spamWordsList.append(i)
        else:
            for i in rowWords:
                if i !='':
                    hamWordsList.append(i)
allTrainingWords = hamWordsList+spamWordsList
word_count = Counter(allTrainingWords)
word_count = word_count.most_common(5000)
mostCommonWords = []
for i in range(0,len(word_count)):
    mostCommonWords.append(word_count[i][0])
wordProbSpam = {}
wordProbHam = {}
for i in mostCommonWords:
    wordProbSpam[i] = (spamWordsList.count(i)+1)/(spamWordsList.count(i)+1+hamWordsList.count(i)+1)
    wordProbHam[i] = (hamWordsList.count(i)+1)/(spamWordsList.count(i)+1+hamWordsList.count(i)+1)
testData = loadTrainingData()
correctCount = 0
incorrectCount = 0
isSpam = True
with open("./Dataset/training-data.csv") as file:
    testingDataCSV = list(csv.reader(file))
    for i in range(len(testData)):
        hamProb = 0.87
        spamProb = 0.13
        for j in testData[i]:
            if wordProbHam.get(j) == None:  
                wordProbHam[j] = 1/2
                wordProbSpam[j] = 1/2
            hamProb*=wordProbHam[j]
            spamProb*=wordProbSpam[j]
        if hamProb > spamProb:
            isSpam = False
        else:
            isSpam = True
        if isSpam == True and testingDataCSV[i][0] == "spam" or isSpam == False and testingDataCSV[i][0] == "ham":
            correctCount+=1
        else:
            incorrectCount+=1
accuracy = correctCount/(incorrectCount+correctCount)
#random comment
print(accuracy)