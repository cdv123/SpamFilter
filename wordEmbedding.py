from simplifyDataset import stopwords,loadSMS2
import numpy as np
#Steps
#Read text
#Preprocess text
#Create (x,y) data points
#Create one hot encoded (X,Y) matrices
#train neural network
#neural network should have one input layer, one hidden layer and one output layer
#extract weights from input layer
import math
def makeAllPairs(trainingData,window):
    wordPairs=[]
    text = []
    for row in trainingData:
        row_words = row.split()
        for i in range(len(row_words)):
            pairList = makeRowPairs(window,row_words,i)
            wordPairs+= pairList
        for i in range(len(row_words)):
            text.append(row_words[i])
    return wordPairs,text
def makeRowPairs(window,row_words,index):
    j = 1
    pairList = []
    while index-j>=0 and j<=window:
        pairList.append((row_words[index],row_words[index-j]))
        j+=1
    j = 1
    while index+j<len(row_words) and j<= window:
        pairList.append((row_words[index],row_words[index+j]))
        j+=1
    return pairList
def uniqueWordDict(text):
    text = list(set(text))
    wordDict = {}
    j=0
    for i in text:
        wordDict[i] = j
        j+=1
    return wordDict
def samplingRate(frequency):
    prob = 0.001/frequency
    multiplier = math.sqrt(frequency/0.001)+1
    prob*=multiplier
    return prob
def makeMatrices(wordDict,pairs):
    focusMatrix = []
    contextMatrix = []
    wordCount = len(wordDict)
    for words in pairs:
        focusWordIndex = wordDict[words[0]]
        contextWordIndex = wordDict[words[1]]
        focusRow = np.zeros(wordCount)
        focusRow[focusWordIndex] = 1
        contextRow = np.zeros(wordCount)
        contextRow[contextWordIndex] = 1
        focusMatrix.append(focusRow)
        contextMatrix.append(contextRow)
    return focusMatrix,contextMatrix
trainingData,spamData = loadSMS2("SMSVal.txt")
window = 2
pairs,text = makeAllPairs(trainingData,window)
wordDict = uniqueWordDict(text)
print(len(wordDict))
focusMatrix,contextMatrix = makeMatrices(wordDict,pairs)