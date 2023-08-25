import numpy as np
import math
import random
from oneHot import oneHotEncode
from oneHot import getMostCommonWords
from simplifyDataset import loadSMS
from simplifyDataset import convertSpamToBinary
def neuralNetwork(trainingData, spamData,lr):
    vectorSize = len(trainingData[0])
    #initialise weights
    w = np.random.rand(vectorSize)*lr
    #initialise m and v for adam optimizer
    m = np.zeros(vectorSize)
    v = np.zeros(vectorSize)
    #initialise bias
    b = random.random()

    # shuffle the lists with same order
    zipped = list(zip(trainingData, spamData))
    random.shuffle(zipped)
    trainingData, spamData = zip(*zipped)
    #train step - 
    for epoch in range(100):
        for x in range(len(trainingData)):
            a = getOutput(trainingData[x],w,b)
            a = sigmoid(a)
            loss = BCE(spamData[x],a,vectorSize)
            gradw,gradb = computeGradient(trainingData[x],spamData[x],a,vectorSize)
            w,b = updateWeights(w,b,gradw,gradb,lr)
    return w,b

def sigmoid(x):
    return (1/(1+(math.exp(-x))))

def BCE(y,logit,m):
    loss = np.multiply(-y,np.log(logit))-np.multiply((1-y),np.log(1-logit))
    loss = np.sum(loss)*1/m
    return loss
def getOutput(x,w,b):
    return (np.matmul(x,w)+b)

def computeGradient(x,y,a,vectorSize):
    gradx = (1/vectorSize)*np.dot(a-y,x)
    gradb = (1/vectorSize)*np.sum(a-y)
    return gradx,gradb
def updateWeights(w,b,dw,db,lr):
    w = w-lr*dw
    b = b-lr*db
    return w,b

def adamOptim(w,lr,m,v,g):
    epsilon = 1*(10**-8)
    m = 0.99*m + (1-0.99)*g
    v = 0.999*v + (1-0.999)*(g^2)
    mhat = m/(1-0.99)
    vhat = m/(1-0.999)
    w = w-((lr)/math.sqrt(vhat)+epsilon)*mhat
    return m,v,w
def testNetwork(testData,spamTest,w,b):
    outputs = []
    correct = 0
    incorrect = 0
    for i in testData:
        outputs.append(getOutput(i,w,b))
    predictions = []
    for i in range(len(outputs)):
        if sigmoid(outputs[i])>0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    for i in range(len(predictions)):
        if predictions[i] == spamTest[i]:
            correct+=1
        else:
            incorrect+=1
    accuarcy = correct/(correct+incorrect)
    return accuarcy
trainingData, spamData = loadSMS('SMSSpamCollection.csv')
spamData = convertSpamToBinary(spamData)
vector_size = 300
mostCommonWords = getMostCommonWords(trainingData,vector_size)
oneHotTrain = list(oneHotEncode(trainingData,mostCommonWords).values())
testData, spamTest = loadSMS('SMSTest.csv')
oneHotTest = list(oneHotEncode(testData,mostCommonWords).values())
spamTest = convertSpamToBinary(spamTest)
weights,bias = neuralNetwork(oneHotTrain,spamData,0.9)
# print(weights,bias)
print(testNetwork(oneHotTest,spamTest,weights,bias))

