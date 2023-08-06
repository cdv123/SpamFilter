from simplifyDataset import loadTrainingData
from simplifyDataset import loadTestData
from simplifyDataset import loadSMS
from collections import Counter
import numpy as np
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
trainingData,spamList = loadSMS('SMSSpamCollection.csv')
mostCommonWords = []
allTrainingWords = []
spamWordsSet = set()
spamWordsList = []
hamWordsSet = set()
hamWordsList = []
spamCount = 0
length = len(trainingData)
hamCount = 0
correctCount =0
for row in range(length):
    rowWords = trainingData[row].split(" ")
    if spamList[row] == "spam":
        for i in rowWords:
            if i !='':
                spamCount+=1
                spamWordsList.append(i)
    else:
        for i in rowWords:
            if i !='':
                hamCount+=1
                hamWordsList.append(i)

# with open("./Dataset/training-data.csv") as file:
#     trainingDataCSV = list(csv.reader(file))
#     length =len(trainingData)
#     for row in range(len(trainingData)):
#         rowWords = trainingData[row].split(" ")
#         if trainingDataCSV[row][0] == "spam":
#             for i in rowWords:
#                 if i !='':
#                     spamCount+=1
#                     spamWordsList.append(i)
#         else:
#             for i in rowWords:
#                 if i !='':
#                     hamCount+=1
#                     hamWordsList.append(i)

allTrainingWords = hamWordsList+spamWordsList
vocab = set(allTrainingWords)
word_count = Counter(allTrainingWords)
word_count = word_count.most_common(1000)
mostCommonWords = []
for i in range(0,len(word_count)):
    mostCommonWords.append(word_count[i][0])
wordProbSpam = {}
wordProbHam = {}
for i in mostCommonWords:
    wordProbSpam[i] = (spamWordsList.count(i))/(spamCount+len(vocab))
    wordProbHam[i] = (hamWordsList.count(i))/(hamCount+len(vocab))
print({k: v for k, v in sorted(wordProbSpam.items(), key=lambda item: item[1])})
print({k: v for k, v in sorted(wordProbHam.items(), key=lambda item: item[1])})
incorrectCount = 0
isSpam = True
testData,spamTestList = loadSMS('SMSTest.csv')
# for i in range(len(testData))
# with open("./Dataset/test-data.csv") as file:
#     testingDataCSV = list(csv.reader(file))
for i in range(len(testData)):
    hamProb = 0.85
    spamProb = 0.15
    row = testData[i].split(" ")
    for j in row:
        if j != '':
            if wordProbHam.get(j) == None or wordProbHam.get(j) == 0:  
                wordProbHam[j] = 1/2
                wordProbSpam[j] = 1/2
            hamProb*=wordProbHam[j]
            spamProb*= wordProbSpam[j]
    if hamProb > spamProb:
        isSpam = False
    else:
        isSpam = True
    if (isSpam == True and spamTestList[i] == "spam") or (isSpam == False and spamTestList[i] == "ham"):
        correctCount+=1
    else:
        incorrectCount+=1
accuracy = correctCount/(incorrectCount+correctCount)
print(accuracy)